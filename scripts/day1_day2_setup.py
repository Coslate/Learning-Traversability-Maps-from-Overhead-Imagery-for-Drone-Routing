from __future__ import annotations

import argparse
from pathlib import Path

from loveda_project.data import (
    LoveDAConfig,
    build_dataloaders,
    build_scene_datasets,
    compute_class_histogram,
    save_class_histogram_plot,
    save_json,
    save_sample_grid,
    set_seed,
    summarize_domains,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Day 1-2 LoveDA data pipeline setup")
    parser.add_argument("--root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="./outputs/day1_day2", help="Output directory")
    parser.add_argument("--patch-size", type=int, default=512, help="Train/val patch size")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for sanity-check loaders")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--num-vis", type=int, default=20, help="Number of sample visualizations to save")
    parser.add_argument("--download", action="store_true", help="Attempt TorchGeo download")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--train-scenes",
        nargs="+",
        default=["urban", "rural"],
        choices=["urban", "rural"],
        help="Train scenes to include",
    )
    parser.add_argument(
        "--val-scenes",
        nargs="+",
        default=["urban", "rural"],
        choices=["urban", "rural"],
        help="Val scenes to include",
    )
    parser.add_argument(
        "--hist-max-samples",
        type=int,
        default=None,
        help="Optional cap for histogram computation to speed up inspection",
    )
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    config = LoveDAConfig(
        root=args.root,
        patch_size=args.patch_size,
        train_scenes=tuple(args.train_scenes),
        val_scenes=tuple(args.val_scenes),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=args.download,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Building scene datasets...")
    scene_datasets = build_scene_datasets(config)

    print("[2/5] Saving domain split summary...")
    domain_summary = summarize_domains(scene_datasets)
    save_json(domain_summary, output_dir / "domain_split_summary.json")

    print("[3/5] Building sanity-check dataloaders...")
    _, loaders = build_dataloaders(config)
    train_batch = next(iter(loaders["train"]))
    val_batch = next(iter(loaders["val"]))
    loader_shapes = {
        "train_batch_image_shape": list(train_batch["image"].shape),
        "train_batch_mask_shape": list(train_batch["mask"].shape),
        "val_batch_image_shape": list(val_batch["image"].shape),
        "val_batch_mask_shape": list(val_batch["mask"].shape),
    }
    save_json(loader_shapes, output_dir / "loader_shape_summary.json")

    print("[4/5] Computing train class histogram...")
    train_dataset = loaders["train"].dataset
    hist = compute_class_histogram(train_dataset, max_samples=args.hist_max_samples)
    save_json(hist, output_dir / "train_class_histogram_train.json")
    save_class_histogram_plot(hist, output_dir / "train_class_histogram_train.png")

    val_dataset = loaders["val"].dataset
    hist = compute_class_histogram(val_dataset, max_samples=args.hist_max_samples)
    save_json(hist, output_dir / "val_class_histogram_val.json")
    save_class_histogram_plot(hist, output_dir / "val_class_histogram_val.png")

    print("[5/5] Saving baseline visualizations...")
    save_sample_grid(train_dataset, output_dir / "samples" / "train", num_vis=args.num_vis, seed=args.seed)
    save_sample_grid(train_dataset, output_dir / "samples" / "val", num_vis=args.num_vis, seed=args.seed)

    print("Done.")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
