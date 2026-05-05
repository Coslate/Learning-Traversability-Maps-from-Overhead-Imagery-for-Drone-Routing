from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX, LoveDAConfig, build_dataloaders
from loveda_project.inference import (
    SegformerEnsembleInferenceWrapper,
    SegformerInferenceWrapper,
    load_segformer_from_checkpoint,
)
from loveda_project.metrics import MetricSummary, SegmentationMeter, save_confusion_matrix_plot, save_metrics_json
from loveda_project.modeling import SEGFORMER_VARIANTS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a frozen SegFormer checkpoint on LoveDA")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint", type=str)
    checkpoint_group.add_argument("--checkpoints", nargs="+", type=str)
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/eval_smoke")
    parser.add_argument("--variant", type=str, default=None, choices=sorted(SEGFORMER_VARIANTS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tta", type=str, default="none", choices=["none", "sliding"])
    parser.add_argument("--window-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--scales", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--ensemble-weights",
        nargs="+",
        type=float,
        default=None,
        help="Optional model-level weights for --checkpoints; length must match checkpoint count",
    )
    parser.add_argument(
        "--per-class-ensemble-weights",
        type=str,
        default=None,
        help="Optional JSON file with per-class ensemble weights for --checkpoints",
    )
    parser.add_argument(
        "--val-scenes",
        nargs="+",
        default=["urban", "rural"],
        choices=["urban", "rural"],
    )
    args = parser.parse_args(argv)
    if args.checkpoints is not None and args.variant is not None:
        parser.error("--variant can only be used with --checkpoint; ensemble checkpoints use their saved variants")
    if args.ensemble_weights is not None and args.per_class_ensemble_weights is not None:
        parser.error("Use either --ensemble-weights or --per-class-ensemble-weights, not both")
    if args.ensemble_weights is not None:
        if args.checkpoints is None:
            parser.error("--ensemble-weights requires --checkpoints")
        if len(args.ensemble_weights) != len(args.checkpoints):
            parser.error("--ensemble-weights length must match the number of checkpoints")
        if any(not math.isfinite(weight) for weight in args.ensemble_weights):
            parser.error("--ensemble-weights must be finite")
        if any(weight < 0 for weight in args.ensemble_weights):
            parser.error("--ensemble-weights must be non-negative")
        if sum(args.ensemble_weights) <= 0:
            parser.error("--ensemble-weights must have a positive sum")
    if args.per_class_ensemble_weights is not None and args.checkpoints is None:
        parser.error("--per-class-ensemble-weights requires --checkpoints")
    return args


def checkpoint_paths_from_args(args: argparse.Namespace) -> list[str]:
    return args.checkpoints if args.checkpoints is not None else [args.checkpoint]


def load_per_class_ensemble_weights(
    path: str | Path,
    *,
    num_checkpoints: int,
    class_names: dict[int, str] = CLASS_NAMES,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Load a [num_classes, num_checkpoints] per-class ensemble weight matrix."""
    weight_path = Path(path)
    with weight_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("per-class ensemble weights JSON must contain an object")

    expected = {
        name: class_id
        for class_id, name in class_names.items()
        if class_id != ignore_index
    }
    unknown_names = sorted(set(payload) - set(expected))
    if unknown_names:
        raise ValueError(f"Unknown class names in per-class ensemble weights: {unknown_names}")
    missing_names = sorted(set(expected) - set(payload))
    if missing_names:
        raise ValueError(f"Missing class names in per-class ensemble weights: {missing_names}")

    weights = torch.ones(len(class_names), num_checkpoints, dtype=torch.float32)
    for class_name, class_id in expected.items():
        raw_values = payload[class_name]
        if not isinstance(raw_values, list):
            raise ValueError(f"Per-class weights for {class_name!r} must be a list")
        if len(raw_values) != num_checkpoints:
            raise ValueError(
                f"Per-class weights for {class_name!r} must have length {num_checkpoints}"
            )
        class_weights = torch.as_tensor(raw_values, dtype=torch.float32)
        if not bool(torch.isfinite(class_weights).all()):
            raise ValueError(f"Per-class weights for {class_name!r} must be finite")
        if bool((class_weights < 0).any()):
            raise ValueError(f"Per-class weights for {class_name!r} must be non-negative")
        if float(class_weights.sum().item()) <= 0.0:
            raise ValueError(f"Per-class weights for {class_name!r} must have a positive sum")
        weights[class_id] = class_weights
    return weights


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

    checkpoint_paths = checkpoint_paths_from_args(args)
    predictors = []
    for checkpoint_path in checkpoint_paths:
        model = load_segformer_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            num_labels=len(CLASS_NAMES),
            ignore_index=IGNORE_INDEX,
            variant=args.variant if len(checkpoint_paths) == 1 else None,
        )
        predictors.append(SegformerInferenceWrapper(model))

    per_class_weights = None
    if args.per_class_ensemble_weights is not None:
        per_class_weights = load_per_class_ensemble_weights(
            args.per_class_ensemble_weights,
            num_checkpoints=len(checkpoint_paths),
        )

    predictor = (
        predictors[0]
        if len(predictors) == 1
        else SegformerEnsembleInferenceWrapper(
            predictors,
            weights=args.ensemble_weights,
            per_class_weights=per_class_weights,
        )
    )
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

        if args.tta == "sliding":
            outputs = predictor.predict_multiscale_sliding(
                pixel_values=images,
                window_size=args.window_size,
                stride=args.stride,
                scales=args.scales,
            )
        else:
            outputs = predictor.predict(pixel_values=images, target_size=masks.shape[-2:])
        meter.update(outputs.probabilities, masks)
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
        "checkpoint": str(Path(checkpoint_paths[0])) if len(checkpoint_paths) == 1 else None,
        "checkpoints": [str(Path(path)) for path in checkpoint_paths],
        "ensemble_size": len(checkpoint_paths),
        "num_samples": samples_seen,
        "mean_iou": summary.mean_iou,
        "pixel_accuracy": summary.pixel_accuracy,
        "per_class_iou": summary.per_class_iou,
        "mean_entropy": entropy_sum / max(entropy_count, 1),
        "tta": args.tta,
        "window_size": args.window_size,
        "stride": args.stride,
        "scales": args.scales,
        "ensemble_weights": args.ensemble_weights,
        "per_class_ensemble_weights": str(Path(args.per_class_ensemble_weights))
        if args.per_class_ensemble_weights is not None
        else None,
    }
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluated {samples_seen} samples")
    print(f"mIoU={summary.mean_iou:.4f} pixel_acc={summary.pixel_accuracy:.4f}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
