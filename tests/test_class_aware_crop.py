from __future__ import annotations

import random

import pytest
import torch

from loveda_project import data as data_module
from loveda_project.data import IGNORE_INDEX, LoveDAConfig, _collate_samples
from loveda_project.transforms import (
    ClassAwareRandomCropPair,
    RandomCropPair,
    build_train_transforms,
)
from scripts.train_segformer import parse_args


def _sample(size: int = 8) -> dict:
    mask = torch.ones(size, size, dtype=torch.long)
    image = torch.stack(
        [
            mask.float(),
            torch.arange(size * size, dtype=torch.float32).reshape(size, size),
            torch.zeros(size, size, dtype=torch.float32),
        ],
        dim=0,
    )
    return {"image": image, "mask": mask}


def test_class_aware_crop_keeps_candidate_with_enough_target_pixels(monkeypatch) -> None:
    sample = _sample(size=8)
    sample["mask"][2:6, 2:6] = 5
    sample["image"][0] = sample["mask"].float()

    values = iter([2, 2])
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda lower, upper: next(values))

    transformed = ClassAwareRandomCropPair(
        size=4,
        target_classes=[5],
        min_pixels=16,
        num_tries=1,
        p=1.0,
        ignore_index=IGNORE_INDEX,
    )(sample)

    assert transformed["image"].shape == (3, 4, 4)
    assert transformed["mask"].shape == (4, 4)
    assert int((transformed["mask"] == 5).sum().item()) == 16


def test_class_aware_crop_falls_back_when_target_absent(monkeypatch) -> None:
    sample = _sample(size=6)
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda lower, upper: lower)

    transformed = ClassAwareRandomCropPair(
        size=4,
        target_classes=[5],
        min_pixels=1,
        num_tries=1,
        p=1.0,
        ignore_index=IGNORE_INDEX,
    )(sample)

    assert transformed["image"].shape == (3, 4, 4)
    assert transformed["mask"].shape == (4, 4)
    assert int((transformed["mask"] == 5).sum().item()) == 0


def test_class_aware_crop_probability_bypass_still_returns_fixed_size(monkeypatch) -> None:
    sample = _sample(size=3)
    monkeypatch.setattr(random, "random", lambda: 1.0)

    transformed = ClassAwareRandomCropPair(
        size=4,
        target_classes=[5],
        min_pixels=1,
        num_tries=1,
        p=0.5,
        ignore_index=IGNORE_INDEX,
    )(sample)

    assert transformed["image"].shape == (3, 4, 4)
    assert transformed["mask"].shape == (4, 4)


def test_class_aware_crop_preserves_image_mask_alignment(monkeypatch) -> None:
    sample = _sample(size=8)
    sample["mask"][3:7, 1:5] = 5
    sample["image"][0] = sample["mask"].float()

    values = iter([3, 1])
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda lower, upper: next(values))

    transformed = ClassAwareRandomCropPair(
        size=4,
        target_classes=[5],
        min_pixels=16,
        num_tries=1,
        p=1.0,
        ignore_index=IGNORE_INDEX,
    )(sample)

    assert torch.equal(transformed["image"][0].long(), transformed["mask"])


def test_ignore_index_does_not_count_as_target_pixels(monkeypatch) -> None:
    sample = _sample(size=8)
    sample["mask"][:, :] = 1
    sample["mask"][:4, :4] = IGNORE_INDEX
    sample["mask"][4:, 4:] = 5
    sample["image"][0] = sample["mask"].float()

    values = iter([0, 0, 4, 4])
    monkeypatch.setattr(random, "random", lambda: 0.0)
    monkeypatch.setattr(random, "randint", lambda lower, upper: next(values))

    transformed = ClassAwareRandomCropPair(
        size=4,
        target_classes=[IGNORE_INDEX, 5],
        min_pixels=16,
        num_tries=2,
        p=1.0,
        ignore_index=IGNORE_INDEX,
    )(sample)

    assert torch.all(transformed["mask"] == 5)


def test_build_train_transforms_inserts_class_aware_crop_and_is_seeded() -> None:
    transform = build_train_transforms(
        4,
        aug_preset="basic",
        class_aware_crop=True,
        crop_target_classes=[5],
        crop_min_pixels=4,
        crop_tries=3,
        class_aware_crop_prob=1.0,
        ignore_index=IGNORE_INDEX,
    )

    assert any(isinstance(item, ClassAwareRandomCropPair) for item in transform.transforms)
    assert not any(type(item) is RandomCropPair for item in transform.transforms)

    sample_a = _sample(size=8)
    sample_b = {"image": sample_a["image"].clone(), "mask": sample_a["mask"].clone()}
    sample_a["mask"][2:6, 2:6] = 5
    sample_b["mask"][2:6, 2:6] = 5
    sample_a["image"][0] = sample_a["mask"].float()
    sample_b["image"][0] = sample_b["mask"].float()

    random.seed(31)
    first = transform(sample_a)
    random.seed(31)
    second = build_train_transforms(
        4,
        aug_preset="basic",
        class_aware_crop=True,
        crop_target_classes=[5],
        crop_min_pixels=4,
        crop_tries=3,
        class_aware_crop_prob=1.0,
        ignore_index=IGNORE_INDEX,
    )(sample_b)

    assert torch.allclose(first["image"], second["image"])
    assert torch.equal(first["mask"], second["mask"])


def test_train_cli_rejects_unknown_crop_target_class() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--crop-target-classes", "road", "not-a-class"])


def test_train_cli_accepts_crop_target_class_names_in_any_order() -> None:
    args = parse_args(["--crop-target-classes", "forest", "road", "barren"])

    assert args.crop_target_class_ids == (6, 3, 5)


def test_transformed_samples_stack_in_collate() -> None:
    transform = build_train_transforms(
        4,
        aug_preset="basic",
        class_aware_crop=True,
        crop_target_classes=[5],
        crop_min_pixels=1,
        crop_tries=2,
        class_aware_crop_prob=1.0,
        ignore_index=IGNORE_INDEX,
    )
    sample_a = _sample(size=8)
    sample_b = _sample(size=8)
    sample_a["mask"][2:6, 2:6] = 5
    sample_b["mask"][1:5, 1:5] = 5

    random.seed(41)
    transformed_a = transform(sample_a)
    random.seed(43)
    transformed_b = transform(sample_b)
    batch = _collate_samples([transformed_a, transformed_b])

    assert batch["image"].shape == (2, 3, 4, 4)
    assert batch["mask"].shape == (2, 4, 4)


def test_loveda_config_passes_class_aware_crop_to_train_dataset(monkeypatch) -> None:
    class FakeLoveDA:
        def __init__(self, root, split, scene, transforms, download) -> None:
            self.root = root
            self.split = split
            self.scene = scene
            self.transforms = transforms
            self.download = download
            self.files = [{"image": f"{split}_{scene[0]}.png"}]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int) -> dict:
            return self.transforms(_sample(size=6))

    monkeypatch.setattr(data_module, "LoveDA", FakeLoveDA)
    config = LoveDAConfig(
        root="./fake",
        patch_size=4,
        train_scenes=("urban",),
        val_scenes=("urban",),
        class_aware_crop=True,
        crop_target_classes=(5,),
        crop_min_pixels=1,
        crop_tries=2,
        class_aware_crop_prob=1.0,
    )

    scene_datasets = data_module.build_scene_datasets(config)
    train_pipeline = scene_datasets["train"]["urban"].dataset.transforms.transforms[0]
    val_pipeline = scene_datasets["val"]["urban"].dataset.transforms.transforms[0]

    assert any(isinstance(item, ClassAwareRandomCropPair) for item in train_pipeline.transforms)
    assert not any(isinstance(item, ClassAwareRandomCropPair) for item in val_pipeline.transforms)
