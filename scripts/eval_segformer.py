from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX, LoveDAConfig, build_dataloaders
from loveda_project.inference import SegformerInferenceWrapper, load_segformer_from_checkpoint
from loveda_project.metrics import MetricSummary, SegmentationMeter, save_confusion_matrix_plot, save_metrics_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen SegFormer checkpoint on LoveDA")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/eval_smoke")
    parser.add_argument("--variant", type=str, default=None, choices=["segformer-b0", "segformer-b1"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--val-scenes",
        nargs="+",
        default=["urban", "rural"],
        choices=["urban", "rural"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    config = LoveDAConfig(
        root=args.root,
        patch_size=args.patch_size,
        train_scenes=tuple(args.val_scenes),
        val_scenes=tuple(args.val_scenes),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=False,
    )
    _, loaders = build_dataloaders(config)
    loader = loaders["val"]

    model = load_segformer_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        num_labels=len(CLASS_NAMES),
        ignore_index=IGNORE_INDEX,
        variant=args.variant,
    )
    predictor = SegformerInferenceWrapper(model)
    meter = SegmentationMeter(num_classes=len(CLASS_NAMES), ignore_index=IGNORE_INDEX, class_names=CLASS_NAMES)

    entropy_sum = 0.0
    entropy_count = 0
    samples_seen = 0

    for batch in tqdm(loader, desc="eval"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if args.max_samples is not None:
            remaining = args.max_samples - samples_seen
            if remaining <= 0:
                break
            images = images[:remaining]
            masks = masks[:remaining]

        outputs = predictor.predict(pixel_values=images, target_size=masks.shape[-2:])
        meter.update(outputs.logits, masks)
        entropy_sum += float(outputs.entropy.sum().item())
        entropy_count += int(outputs.entropy.numel())
        samples_seen += int(images.shape[0])

    summary = meter.compute()
    save_metrics_json(
        MetricSummary(
            mean_iou=summary.mean_iou,
            pixel_accuracy=summary.pixel_accuracy,
            per_class_iou=summary.per_class_iou,
            confusion_matrix=summary.confusion_matrix,
        ),
        output_dir / "metrics.json",
    )

    class_names_wo_ignore = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES) if i != IGNORE_INDEX]
    save_confusion_matrix_plot(
        confusion_matrix=summary.confusion_matrix[1:, 1:],
        class_names=class_names_wo_ignore,
        out_path=output_dir / "confusion_matrix.png",
        normalize=True,
    )

    report = {
        "checkpoint": str(Path(args.checkpoint)),
        "num_samples": samples_seen,
        "mean_iou": summary.mean_iou,
        "pixel_accuracy": summary.pixel_accuracy,
        "per_class_iou": summary.per_class_iou,
        "mean_entropy": entropy_sum / max(entropy_count, 1),
    }
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluated {samples_seen} samples")
    print(f"mIoU={summary.mean_iou:.4f} pixel_acc={summary.pixel_accuracy:.4f}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
