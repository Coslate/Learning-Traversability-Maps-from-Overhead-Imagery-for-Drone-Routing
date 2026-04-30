from __future__ import annotations

import random

import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from loveda_project.transforms import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    ComposeDict,
    EnsureTensorTypes,
    NormalizeImage,
    PadToSizePair,
    RandomColorJitterImage,
    RandomCropPair,
    RandomGaussianBlurImage,
    RandomHorizontalFlipPair,
    RandomScalePair,
    RandomVerticalFlipPair,
    build_train_transforms,
)


def _sample(size: int = 8) -> dict:
    image = torch.linspace(0, 1, steps=3 * size * size, dtype=torch.float32).reshape(3, size, size)
    mask = torch.arange(size * size, dtype=torch.long).reshape(size, size) % 8
    return {"image": image, "mask": mask}


def test_color_jitter_changes_only_image() -> None:
    random.seed(7)
    sample = _sample()
    original_image = sample["image"].clone()
    original_mask = sample["mask"].clone()

    transformed = RandomColorJitterImage(
        brightness=0.3,
        contrast=0.0,
        saturation=0.0,
        hue=0.0,
        p=1.0,
    )(sample)

    assert not torch.allclose(transformed["image"], original_image)
    assert torch.equal(transformed["mask"], original_mask)


def test_gaussian_blur_changes_only_image() -> None:
    sample = {
        "image": torch.zeros(3, 9, 9, dtype=torch.float32),
        "mask": torch.arange(81, dtype=torch.long).reshape(9, 9) % 8,
    }
    sample["image"][:, 4, 4] = 1.0
    original_image = sample["image"].clone()
    original_mask = sample["mask"].clone()

    transformed = RandomGaussianBlurImage(kernel_size=5, sigma=(1.0, 1.0), p=1.0)(sample)

    assert not torch.allclose(transformed["image"], original_image)
    assert torch.equal(transformed["mask"], original_mask)


def test_random_scale_pair_uses_bilinear_for_image_and_nearest_for_mask() -> None:
    sample = _sample(size=4)
    expected_image = TF.resize(
        sample["image"],
        [8, 8],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    expected_mask = TF.resize(
        sample["mask"].unsqueeze(0),
        [8, 8],
        interpolation=InterpolationMode.NEAREST,
    ).squeeze(0)

    random.seed(3)
    transformed = RandomScalePair(scale_range=(2.0, 2.0))(sample)

    assert transformed["image"].shape == (3, 8, 8)
    assert transformed["mask"].shape == (8, 8)
    assert torch.allclose(transformed["image"], expected_image)
    assert torch.equal(transformed["mask"], expected_mask)
    assert set(transformed["mask"].unique().tolist()).issubset(set(range(8)))


def test_pad_to_size_pair_pads_image_and_mask_symmetrically() -> None:
    image = torch.ones(3, 2, 3, dtype=torch.float32)
    mask = torch.full((2, 3), 5, dtype=torch.long)
    transformed = PadToSizePair(size=5, image_fill=0.25, mask_fill=99)({"image": image, "mask": mask})

    assert transformed["image"].shape == (3, 5, 5)
    assert transformed["mask"].shape == (5, 5)
    assert torch.allclose(transformed["image"][:, 1:3, 1:4], image)
    assert torch.equal(transformed["mask"][1:3, 1:4], mask)
    assert torch.all(transformed["image"][:, 0, :] == 0.25)
    assert torch.all(transformed["mask"][0, :] == 99)


def test_strong_aug_pads_scaled_down_samples_to_patch_size(monkeypatch) -> None:
    sample = _sample(size=8)
    monkeypatch.setattr(random, "uniform", lambda lower, upper: lower)

    random.seed(17)
    transformed = build_train_transforms(8, aug_preset="strong")(sample)

    assert transformed["image"].shape == (3, 8, 8)
    assert transformed["mask"].shape == (8, 8)


def test_aug_preset_basic_matches_original_train_pipeline() -> None:
    sample_a = _sample(size=10)
    sample_b = {"image": sample_a["image"].clone(), "mask": sample_a["mask"].clone()}
    manual = ComposeDict(
        [
            EnsureTensorTypes(),
            RandomCropPair(size=6),
            RandomHorizontalFlipPair(p=0.5),
            RandomVerticalFlipPair(p=0.5),
            NormalizeImage(mean=DEFAULT_MEAN, std=DEFAULT_STD),
        ]
    )

    random.seed(11)
    expected = manual(sample_a)
    random.seed(11)
    actual = build_train_transforms(6, aug_preset="basic")(sample_b)

    assert torch.allclose(actual["image"], expected["image"])
    assert torch.equal(actual["mask"], expected["mask"])


def test_aug_preset_strong_is_deterministic_with_seeded_rng() -> None:
    sample_a = _sample(size=12)
    sample_b = {"image": sample_a["image"].clone(), "mask": sample_a["mask"].clone()}

    random.seed(13)
    first = build_train_transforms(6, aug_preset="strong")(sample_a)
    random.seed(13)
    second = build_train_transforms(6, aug_preset="strong")(sample_b)

    assert first["image"].shape == (3, 6, 6)
    assert first["mask"].shape == (6, 6)
    assert torch.allclose(first["image"], second["image"])
    assert torch.equal(first["mask"], second["mask"])
