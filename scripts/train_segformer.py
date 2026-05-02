from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX, LoveDAConfig, build_dataloaders, save_json, set_seed
from loveda_project.losses import CriterionConfig, SegmentationCriterion, compute_class_weights
from loveda_project.metrics import SegmentationMeter, save_confusion_matrix_plot, save_metrics_json
from loveda_project.modeling import SegformerBuildConfig, build_segformer_model

try:
    import wandb
except ImportError:
    wandb = None


CLASS_NAME_TO_ID = {class_name: class_id for class_id, class_name in CLASS_NAMES.items()}


def resolve_crop_target_classes(class_names: Sequence[str]) -> tuple[int, ...]:
    class_ids: list[int] = []
    for class_name in class_names:
        key = class_name.strip().lower()
        if key.isdigit():
            class_id = int(key)
            if class_id not in CLASS_NAMES:
                valid_names = ", ".join(CLASS_NAME_TO_ID)
                raise ValueError(f"Unknown crop target class '{class_name}'. Valid names: {valid_names}")
        elif key in CLASS_NAME_TO_ID:
            class_id = CLASS_NAME_TO_ID[key]
        else:
            valid_names = ", ".join(CLASS_NAME_TO_ID)
            raise ValueError(f"Unknown crop target class '{class_name}'. Valid names: {valid_names}")
        class_ids.append(class_id)
    return tuple(class_ids)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SegFormer baseline on LoveDA")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/day3_day5")
    parser.add_argument("--variant", type=str, default="segformer-b0", choices=["segformer-b0", "segformer-b1", "segformer-b2"])
    parser.add_argument("--pretrained", action="store_true", help="Load HuggingFace pretrained weights if available")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--aug-preset", type=str, default="basic", choices=["basic", "strong"])
    parser.add_argument("--class-aware-crop", action="store_true", help="Prefer crops containing target classes")
    parser.add_argument(
        "--crop-target-classes",
        nargs="+",
        default=["road", "barren", "forest"],
        help="LoveDA class names or ids to target when --class-aware-crop is enabled",
    )
    parser.add_argument("--crop-min-pixels", type=int, default=1024)
    parser.add_argument("--crop-tries", type=int, default=20)
    parser.add_argument("--class-aware-crop-prob", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument(
        "--scheduler-type",
        type=str,
        default="warmup+cosine",
        choices=["none", "cosine-only", "warmup+cosine"],
    )
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=2)

    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--loss-name", type=str, default="ce", choices=["ce", "ce_dice", "focal", "lovasz", "ce_lovasz"])
    parser.add_argument("--dice-weight", type=float, default=0.25)
    parser.add_argument("--lovasz-weight", type=float, default=0.5)
    parser.add_argument(
        "--class-weight-mode",
        type=str,
        default="none",
        choices=["none", "inverse", "effective", "median"],
    )
    parser.add_argument("--class-stats", type=str, default=None)
    parser.add_argument("--class-weight-beta", type=float, default=0.9999)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save epoch_NNN.pth every N epochs; use 0 to disable periodic epoch checkpoints.",
    )
    parser.add_argument(
        "--train-scenes",
        nargs="+",
        default=["urban", "rural"],
        choices=["urban", "rural"],
    )
    parser.add_argument(
        "--val-scenes",
        nargs="+",
        default=["urban", "rural"],
        choices=["urban", "rural"],
    )

    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="loveda-segformer")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B logging mode",
    )

    args = parser.parse_args(argv)
    if args.crop_min_pixels <= 0:
        parser.error("--crop-min-pixels must be > 0")
    if args.crop_tries <= 0:
        parser.error("--crop-tries must be > 0")
    if not 0.0 <= args.class_aware_crop_prob <= 1.0:
        parser.error("--class-aware-crop-prob must be between 0 and 1")
    if args.lovasz_weight < 0:
        parser.error("--lovasz-weight must be >= 0")

    try:
        args.crop_target_class_ids = resolve_crop_target_classes(args.crop_target_classes)
    except ValueError as exc:
        parser.error(str(exc))

    if args.class_aware_crop and not any(class_id != IGNORE_INDEX for class_id in args.crop_target_class_ids):
        parser.error("--crop-target-classes must include at least one non-ignore class")

    return args


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
) -> Dict[str, object]:
    model.eval()
    meter = SegmentationMeter(num_classes=num_classes, ignore_index=IGNORE_INDEX, class_names=CLASS_NAMES)
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="val", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        outputs = model(pixel_values=images)
        logits = outputs.logits
        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)

        loss = criterion(logits, masks)
        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        meter.update(logits, masks)

    summary = meter.compute()
    return {
        "val_loss": total_loss / max(total_samples, 1),
        "mean_iou": summary.mean_iou,
        "pixel_accuracy": summary.pixel_accuracy,
        "per_class_iou": summary.per_class_iou,
        "confusion_matrix": summary.confusion_matrix,
    }


def iter_per_class_iou(per_class_iou, class_names, ignore_index):
    if isinstance(per_class_iou, dict):
        for class_name, class_iou in per_class_iou.items():
            yield class_name, float(class_iou)
    else:
        for class_idx, class_iou in enumerate(per_class_iou):
            if class_idx == ignore_index:
                continue
            yield class_names[class_idx], float(class_iou)


def load_class_counts(class_stats_path: str | Path, class_names: Dict[int, str]) -> torch.Tensor:
    with Path(class_stats_path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_counts = payload.get("counts", payload)
    counts = torch.zeros(max(class_names) + 1, dtype=torch.float32)
    for class_id, class_name in class_names.items():
        if class_name not in raw_counts:
            raise ValueError(f"Missing class '{class_name}' in class stats: {class_stats_path}")
        counts[class_id] = float(raw_counts[class_name])
    return counts

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    steps_per_epoch: int,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
):
    if scheduler_type == "none":
        return None

    total_steps = steps_per_epoch * epochs
    if total_steps <= 0:
        raise ValueError("total_steps must be > 0")

    if scheduler_type == "cosine-only":
        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=min_lr,
        )

    if scheduler_type == "warmup+cosine":
        warmup_steps = steps_per_epoch * warmup_epochs

        # 避免 warmup_steps >= total_steps 導致 scheduler 出問題
        if warmup_steps >= total_steps:
            warmup_steps = max(total_steps - 1, 0)

        if warmup_steps > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=max(1, total_steps - warmup_steps),
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )

        return CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps),
            eta_min=min_lr,
        )

    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    grad_accum_steps: int = 1,
) -> float:
    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be >= 1")

    model.train()
    total_loss = 0.0
    total_samples = 0
    num_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        if amp_enabled and scaler is not None:
            with torch.autocast(device_type="cuda"):
                outputs = model(pixel_values=images)
                logits = outputs.logits
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = criterion(logits, masks)
            scaler.scale(loss / grad_accum_steps).backward()
            should_step = batch_idx % grad_accum_steps == 0 or batch_idx == num_batches
            if should_step:
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = model(pixel_values=images)
            logits = outputs.logits
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks)
            (loss / grad_accum_steps).backward()
            should_step = batch_idx % grad_accum_steps == 0 or batch_idx == num_batches
            if should_step:
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.save_every < 0:
        raise ValueError("--save-every must be >= 0")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    config = LoveDAConfig(
        root=args.root,
        patch_size=args.patch_size,
        train_scenes=tuple(args.train_scenes),
        val_scenes=tuple(args.val_scenes),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=False,
        seed=args.seed,
        aug_preset=args.aug_preset,
        class_aware_crop=args.class_aware_crop,
        crop_target_classes=tuple(args.crop_target_class_ids),
        crop_min_pixels=args.crop_min_pixels,
        crop_tries=args.crop_tries,
        class_aware_crop_prob=args.class_aware_crop_prob,
    )
    _, loaders = build_dataloaders(config)

    device = torch.device(args.device)
    num_classes = len(CLASS_NAMES)
    class_weights = None
    if args.class_weight_mode != "none":
        if args.class_stats is None:
            raise ValueError("--class-stats is required when --class-weight-mode is not 'none'")
        class_counts = load_class_counts(args.class_stats, CLASS_NAMES)
        class_weights = compute_class_weights(
            class_counts=class_counts,
            mode=args.class_weight_mode,
            ignore_index=IGNORE_INDEX,
            beta=args.class_weight_beta,
        )
        class_weight_payload = {
            CLASS_NAMES[class_id]: float(class_weights[class_id].item())
            for class_id in sorted(CLASS_NAMES)
        }
        save_json(
            {
                "mode": args.class_weight_mode,
                "beta": args.class_weight_beta,
                "class_stats": args.class_stats,
                "weights": class_weight_payload,
            },
            output_dir / "class_weights.json",
        )
        print(f"Using {args.class_weight_mode} class weights:")
        for class_id in sorted(CLASS_NAMES):
            print(f"  {CLASS_NAMES[class_id]}: {class_weights[class_id].item():.6f}")

    model = build_segformer_model(
        SegformerBuildConfig(
            variant=args.variant,
            num_labels=num_classes,
            ignore_index=IGNORE_INDEX,
            pretrained=args.pretrained,
        )
    ).to(device)

    criterion = SegmentationCriterion(
        CriterionConfig(
            num_classes=num_classes,
            ignore_index=IGNORE_INDEX,
            loss_name=args.loss_name,
            dice_weight=args.dice_weight,
            lovasz_weight=args.lovasz_weight,
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
        )
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    steps_per_epoch = math.ceil(len(loaders["train"]) / args.grad_accum_steps)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler_type,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )

    run_config = vars(args).copy()
    save_json(run_config, output_dir / "run_config.json")

    def _sanitize_metric_name(name: str) -> str:
        return name.replace(" ", "_").replace("/", "_").lower()

    run = None
    if args.use_wandb:
        if wandb is None:
            raise ImportError(
                "You passed --use-wandb, but wandb is not installed. "
                "Please run: pip install wandb"
            )

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=str(output_dir),
            config=run_config,
            mode=args.wandb_mode,
        )

        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        run.define_metric("best/*", step_metric="epoch")

    best_miou = -1.0
    history = []

    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            scaler=scaler if args.amp and device.type == "cuda" else None,
            amp_enabled=args.amp and device.type == "cuda",
            grad_accum_steps=args.grad_accum_steps,
        )
        val_metrics = evaluate(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        epoch_time = time.time() - t0

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["val_loss"],
            "mean_iou": val_metrics["mean_iou"],
            "pixel_accuracy": val_metrics["pixel_accuracy"],
            "per_class_iou": val_metrics["per_class_iou"],
            "epoch_time_sec": epoch_time,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_log)

        current_lr = optimizer.param_groups[0]["lr"]
        if run is not None:
            log_dict = {
                "epoch": epoch,
                "train/loss": float(train_loss),
                "train/lr": float(current_lr),
                "train/epoch_time_sec": float(epoch_time),
                "val/loss": float(val_metrics["val_loss"]),
                "val/mean_iou": float(val_metrics["mean_iou"]),
                "val/pixel_accuracy": float(val_metrics["pixel_accuracy"]),
            }

            per_class_iou = val_metrics["per_class_iou"]
            for class_name, class_iou in iter_per_class_iou(per_class_iou, CLASS_NAMES, IGNORE_INDEX):
                safe_name = _sanitize_metric_name(class_name)
                log_dict[f"val/per_class_iou/{safe_name}"] = class_iou            

            run.log(log_dict)

        print(
            f"[epoch {epoch:03d}] "
            f"lr={current_lr:.8f} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['val_loss']:.4f} "
            f"mIoU={val_metrics['mean_iou']:.4f} "
            f"pixel_acc={val_metrics['pixel_accuracy']:.4f} "
            f"time={epoch_time:.1f}s"
        )

        save_json({"history": history}, output_dir / "history.json")

        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "args": run_config,
                },
                ckpt_dir / f"epoch_{epoch:03d}.pth",
            )

        if val_metrics["mean_iou"] > best_miou:
            best_miou = float(val_metrics["mean_iou"])
            best_payload = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "args": run_config,
                "best_mean_iou": best_miou,
            }
            torch.save(best_payload, ckpt_dir / "best_model.pth")

            class_names_wo_ignore = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES) if i != IGNORE_INDEX]
            cm_wo_ignore = val_metrics["confusion_matrix"][1:, 1:]
            save_confusion_matrix_plot(
                confusion_matrix=cm_wo_ignore,
                class_names=class_names_wo_ignore,
                out_path=output_dir / "best_confusion_matrix.png",
                normalize=True,
            )

            from loveda_project.metrics import MetricSummary

            save_metrics_json(
                MetricSummary(
                    mean_iou=val_metrics["mean_iou"],
                    pixel_accuracy=val_metrics["pixel_accuracy"],
                    per_class_iou=val_metrics["per_class_iou"],
                    confusion_matrix=val_metrics["confusion_matrix"],
                ),
                output_dir / "best_metrics.json",
            )

            if run is not None:
                run.summary["best/epoch"] = int(epoch)
                run.summary["best/mean_iou"] = float(val_metrics["mean_iou"])
                run.summary["best/pixel_accuracy"] = float(val_metrics["pixel_accuracy"])
                run.summary["best/val_loss"] = float(val_metrics["val_loss"])

                for class_name, class_iou in iter_per_class_iou(val_metrics["per_class_iou"], CLASS_NAMES, IGNORE_INDEX):
                    safe_name = _sanitize_metric_name(class_name)
                    run.summary[f"best/per_class_iou/{safe_name}"] = class_iou

                best_cm_path = output_dir / "best_confusion_matrix.png"
                if best_cm_path.exists():
                    run.log(
                        {
                            "epoch": epoch,
                            "val/best_confusion_matrix": wandb.Image(str(best_cm_path)),
                        }
                    )

    print(f"Finished. Best val mIoU = {best_miou:.4f}")
    print(f"Outputs saved to: {output_dir.resolve()}")

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
