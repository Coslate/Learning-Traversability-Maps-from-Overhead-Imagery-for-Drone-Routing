from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchgeo.datasets import LoveDA
from tqdm import tqdm

from loveda_project.data import CLASS_NAMES
from loveda_project.transforms import EnsureTensorTypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LoveDA train pixel counts for class-balanced losses")
    parser.add_argument("--root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs/class_stats")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--train-scenes",
        nargs="+",
        default=["urban", "rural"],
        choices=["urban", "rural"],
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def _default_output_path(output_dir: str | Path, train_scenes: list[str]) -> Path:
    scene_key = "_".join(train_scenes)
    return Path(output_dir) / f"{scene_key}.json"


def main() -> None:
    args = parse_args()
    out_path = Path(args.output) if args.output is not None else _default_output_path(args.output_dir, args.train_scenes)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = torch.zeros(max(CLASS_NAMES) + 1, dtype=torch.int64)
    transform = EnsureTensorTypes()
    samples_counted = 0

    for scene in args.train_scenes:
        dataset = LoveDA(
            root=args.root,
            split="train",
            scene=[scene],
            transforms=transform,
            download=args.download,
        )
        remaining = None if args.max_samples is None else max(args.max_samples - samples_counted, 0)
        length = len(dataset) if remaining is None else min(len(dataset), remaining)
        if length == 0:
            break

        for idx in tqdm(range(length), desc=f"train/{scene}"):
            mask = dataset[idx]["mask"].reshape(-1)
            unique = set(mask.unique().tolist())
            unknown = sorted(class_id for class_id in unique if class_id not in CLASS_NAMES)
            if unknown:
                raise ValueError(f"Unknown class ids in {scene} sample {idx}: {unknown}")
            counts += torch.bincount(mask, minlength=counts.numel()).cpu()
            samples_counted += 1

    payload = {
        "split": "train",
        "scenes": args.train_scenes,
        "num_samples_counted": samples_counted,
        "counts": {CLASS_NAMES[class_id]: int(counts[class_id].item()) for class_id in sorted(CLASS_NAMES)},
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote class stats to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
