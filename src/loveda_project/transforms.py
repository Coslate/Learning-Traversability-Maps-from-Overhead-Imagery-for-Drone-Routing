from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, MutableMapping, Union

import torch
from torchvision.transforms import functional as TF

SampleValue = Union[torch.Tensor, str]
Sample = MutableMapping[str, SampleValue]


class ComposeDict:
    """Compose paired transforms operating on a sample dict.

    Expected sample format:
        {
            "image": Tensor[C, H, W],
            "mask": Tensor[H, W],
            ...
        }
    """

    def __init__(self, transforms: Iterable[Callable[[Sample], Sample]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, sample: Sample) -> Sample:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class EnsureTensorTypes:
    """Canonicalize dtypes for LoveDA samples.

    - image -> float32 in [0, 1]
    - mask  -> int64
    """

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        mask = sample["mask"]

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)   # already float32 in [0,1] if it is PIL/uint8
        else:
            image = image.detach().clone().float()

            # value is [0, 255], normalized to [0, 1]
            if image.max() > 1.0:
                image = image / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        else:
            mask = mask.detach().clone()
        mask = mask.long()

        sample["image"] = image
        sample["mask"] = mask
        return sample

@dataclass
class RandomCropPair:
    size: int

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        mask = sample["mask"]
        _, h, w = image.shape

        crop_h = min(self.size, h)
        crop_w = min(self.size, w)

        if h == crop_h and w == crop_w:
            return sample

        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)

        sample["image"] = TF.crop(image, top, left, crop_h, crop_w)
        sample["mask"] = TF.crop(mask.unsqueeze(0), top, left, crop_h, crop_w).squeeze(0)
        return sample


@dataclass
class CenterCropPair:
    size: int

    def __call__(self, sample: Sample) -> Sample:
        image = sample["image"]
        mask = sample["mask"]
        _, h, w = image.shape
        crop_h = min(self.size, h)
        crop_w = min(self.size, w)

        sample["image"] = TF.center_crop(image, [crop_h, crop_w])
        sample["mask"] = TF.center_crop(mask.unsqueeze(0), [crop_h, crop_w]).squeeze(0)
        return sample


@dataclass
class RandomHorizontalFlipPair:
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample["image"] = TF.hflip(sample["image"])
            sample["mask"] = TF.hflip(sample["mask"].unsqueeze(0)).squeeze(0)
        return sample


@dataclass
class RandomVerticalFlipPair:
    p: float = 0.5

    def __call__(self, sample: Sample) -> Sample:
        if random.random() < self.p:
            sample["image"] = TF.vflip(sample["image"])
            sample["mask"] = TF.vflip(sample["mask"].unsqueeze(0)).squeeze(0)
        return sample


@dataclass
class NormalizeImage:
    mean: List[float]
    std: List[float]

    def __call__(self, sample: Sample) -> Sample:
        sample["image"] = TF.normalize(sample["image"], mean=self.mean, std=self.std)
        return sample


class AddMetadata:
    """Attach simple metadata fields to each sample."""

    def __init__(self, split: str, scene: str) -> None:
        self.split = split
        self.scene = scene

    def __call__(self, sample: Sample) -> Sample:
        sample["split_name"] = self.split
        sample["scene_name"] = self.scene
        return sample


DEFAULT_MEAN = [0.485, 0.456, 0.406]
DEFAULT_STD = [0.229, 0.224, 0.225]


def build_train_transforms(patch_size: int) -> ComposeDict:
    return ComposeDict(
        [
            EnsureTensorTypes(),
            RandomCropPair(size=patch_size),
            RandomHorizontalFlipPair(p=0.5),
            RandomVerticalFlipPair(p=0.5),
            NormalizeImage(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )

def build_val_transforms(patch_size: int) -> ComposeDict:
    return ComposeDict(
        [
            EnsureTensorTypes(),
            CenterCropPair(size=patch_size),
            NormalizeImage(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )



def denormalize_image(image: torch.Tensor, mean: List[float] | None = None, std: List[float] | None = None) -> torch.Tensor:
    mean = mean or DEFAULT_MEAN
    std = std or DEFAULT_STD
    out = image.detach().clone()
    mean_t = torch.tensor(mean, dtype=out.dtype, device=out.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=out.dtype, device=out.device).view(-1, 1, 1)
    out = out * std_t + mean_t
    return out.clamp(0.0, 1.0)
