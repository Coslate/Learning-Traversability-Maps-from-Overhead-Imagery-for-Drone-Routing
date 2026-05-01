from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchgeo.datasets import LoveDA

from loveda_project.transforms import AddMetadata, ComposeDict, build_train_transforms, build_val_transforms, denormalize_image

IGNORE_INDEX = 0
CLASS_NAMES: Dict[int, str] = {
    0: "ignore",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}

CLASS_COLORS: Dict[int, Sequence[int]] = {
    0: (0, 0, 0),
    1: (180, 180, 180),
    2: (220, 20, 60),
    3: (255, 215, 0),
    4: (30, 144, 255),
    5: (160, 82, 45),
    6: (34, 139, 34),
    7: (255, 140, 0),
}


@dataclass
class LoveDAConfig:
    root: str = "./data"
    patch_size: int = 512
    train_scenes: Sequence[str] = ("urban", "rural")
    val_scenes: Sequence[str] = ("urban", "rural")
    batch_size: int = 4
    num_workers: int = 0
    download: bool = False
    seed: int = 0
    aug_preset: str = "basic"
    class_aware_crop: bool = False
    crop_target_classes: Sequence[int] = ()
    crop_min_pixels: int = 1024
    crop_tries: int = 20
    class_aware_crop_prob: float = 0.5


class WrappedLoveDAScene(Dataset):
    """LoveDA scene dataset with per-scene metadata.

    Each instance wraps exactly one split + one scene so we can preserve domain labels
    cleanly after concatenation.
    """

    def __init__(
        self,
        root: str,
        split: str,
        scene: str,
        patch_size: int,
        download: bool = False,
        aug_preset: str = "basic",
        class_aware_crop: bool = False,
        crop_target_classes: Sequence[int] = (),
        crop_min_pixels: int = 1024,
        crop_tries: int = 20,
        class_aware_crop_prob: float = 0.5,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        if scene not in {"urban", "rural"}:
            raise ValueError(f"Unsupported scene: {scene}")

        base_transform = (
            build_train_transforms(
                patch_size,
                aug_preset=aug_preset,
                class_aware_crop=class_aware_crop,
                crop_target_classes=crop_target_classes,
                crop_min_pixels=crop_min_pixels,
                crop_tries=crop_tries,
                class_aware_crop_prob=class_aware_crop_prob,
                ignore_index=IGNORE_INDEX,
            )
            if split == "train"
            else build_val_transforms(patch_size)
        )
        transform = ComposeDict([base_transform, AddMetadata(split=split, scene=scene)])
        self.dataset = LoveDA(root=root, split=split, scene=[scene], transforms=transform, download=download)
        self.split = split
        self.scene = scene

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample = self.dataset[index]
        sample["filename"] = Path(self.dataset.files[index]["image"]).stem
        return sample



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def build_scene_datasets(config: LoveDAConfig) -> Dict[str, Dict[str, WrappedLoveDAScene]]:
    scene_datasets: Dict[str, Dict[str, WrappedLoveDAScene]] = {"train": {}, "val": {}}

    for scene in config.train_scenes:
        scene_datasets["train"][scene] = WrappedLoveDAScene(
            root=config.root,
            split="train",
            scene=scene,
            patch_size=config.patch_size,
            download=config.download,
            aug_preset=config.aug_preset,
            class_aware_crop=config.class_aware_crop,
            crop_target_classes=config.crop_target_classes,
            crop_min_pixels=config.crop_min_pixels,
            crop_tries=config.crop_tries,
            class_aware_crop_prob=config.class_aware_crop_prob,
        )

    for scene in config.val_scenes:
        scene_datasets["val"][scene] = WrappedLoveDAScene(
            root=config.root,
            split="val",
            scene=scene,
            patch_size=config.patch_size,
            download=config.download,
            aug_preset=config.aug_preset,
        )

    return scene_datasets



def build_concat_dataset(scene_datasets: Dict[str, WrappedLoveDAScene]) -> Dataset:
    datasets = list(scene_datasets.values())
    if len(datasets) == 1:
        return datasets[0]
    return ConcatDataset(datasets)



def _collate_samples(batch: List[dict]) -> dict:
    images = torch.stack([item["image"] for item in batch], dim=0)
    masks = torch.stack([item["mask"] for item in batch], dim=0)
    split_names = [item.get("split_name", "unknown") for item in batch]
    scene_names = [item.get("scene_name", "unknown") for item in batch]
    return {
        "image": images,
        "mask": masks,
        "split_name": split_names,
        "scene_name": scene_names,
    }



def build_dataloaders(config: LoveDAConfig) -> tuple[Dict[str, Dict[str, WrappedLoveDAScene]], Dict[str, DataLoader]]:
    scene_datasets = build_scene_datasets(config)
    train_dataset = build_concat_dataset(scene_datasets["train"])
    val_dataset = build_concat_dataset(scene_datasets["val"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_samples,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_samples,
    )

    return scene_datasets, {"train": train_loader, "val": val_loader}



def summarize_domains(scene_datasets: Dict[str, Dict[str, WrappedLoveDAScene]]) -> dict:
    summary: dict = {"train": {}, "val": {}}
    for split, split_datasets in scene_datasets.items():
        total = 0
        for scene, dataset in split_datasets.items():
            count = len(dataset)
            total += count
            summary[split][scene] = count
        summary[split]["total"] = total
    return summary



def compute_class_histogram(dataset: Dataset, max_samples: int | None = None) -> dict:
    counts = {class_id: 0 for class_id in CLASS_NAMES}
    total_pixels = 0

    length = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(length):
        sample = dataset[idx]
        mask = sample["mask"].reshape(-1)
        
        assert all(class_id in CLASS_NAMES for class_id in sample["mask"].unique().tolist()), f"class_id out of range."

        bincount = torch.bincount(mask, minlength=max(CLASS_NAMES) + 1)
        for class_id in CLASS_NAMES:
            class_count = int(bincount[class_id].item())
            counts[class_id] += class_count
            if class_id != IGNORE_INDEX:
                total_pixels += class_count

    proportions = {}
    denom = max(total_pixels, 1)
    for class_id, count in counts.items():
        proportions[class_id] = 0.0 if class_id == IGNORE_INDEX else count / denom

    return {
        "counts": {CLASS_NAMES[k]: int(v) for k, v in counts.items()},
        "proportions_excluding_ignore": {CLASS_NAMES[k]: float(v) for k, v in proportions.items()},
        "num_samples_counted": length,
    }



def mask_to_rgb(mask: torch.Tensor) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    h, w = mask_np.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb in CLASS_COLORS.items():
        color[mask_np == class_id] = np.asarray(rgb, dtype=np.uint8)
    return color



def sample_to_figure(sample: dict, alpha: float = 0.45):
    image = denormalize_image(sample["image"]).permute(1, 2, 0).detach().cpu().numpy()
    image = np.clip(image, 0.0, 1.0)
    mask_rgb = mask_to_rgb(sample["mask"])
    overlay = ((1 - alpha) * (image * 255.0) + alpha * mask_rgb).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(image)
    axes[0].set_title("image")
    axes[1].imshow(mask_rgb)
    axes[1].set_title("mask")
    axes[2].imshow(overlay)
    axes[2].set_title("overlay")
    for ax in axes:
        ax.axis("off")

    split_name = sample.get("split_name", "unknown")
    scene_name = sample.get("scene_name", "unknown")
    fig.suptitle(f"split={split_name} | scene={scene_name}")
    return fig



def save_sample_grid(dataset: Dataset, out_dir: str | Path, num_vis: int = 20, seed: int = 0) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(num_vis, len(indices))]

    saved_paths: List[Path] = []
    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        filename = sample.get("filename", f"{idx:06d}")
        fig = sample_to_figure(sample)
        split_name = sample.get("split_name", "unknown")
        scene_name = sample.get("scene_name", "unknown")
        out_path = out_dir / f"sample_{rank:03d}_{split_name}_{scene_name}_{filename}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)
    return saved_paths



def save_class_histogram_plot(hist: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class_names = [name for name in hist["counts"].keys() if name != "ignore"]
    values = [hist["counts"][name] for name in class_names]

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(class_names, values)
    ax.set_title("LoveDA train pixel histogram")
    ax.set_ylabel("pixel count")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



def save_json(obj: dict, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
