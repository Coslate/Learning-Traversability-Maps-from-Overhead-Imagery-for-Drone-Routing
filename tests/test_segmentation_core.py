from __future__ import annotations

import pytest
import torch

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX
from loveda_project.losses import MulticlassDiceLoss
from loveda_project.metrics import SegmentationMeter
from loveda_project.modeling import SegformerBuildConfig, build_segformer_model
from loveda_project.transforms import EnsureTensorTypes


def test_ensure_tensor_types_normalizes_image_and_mask() -> None:
    sample = {
        "image": torch.tensor(
            [
                [[0, 255], [128, 64]],
                [[255, 0], [32, 16]],
                [[10, 20], [30, 40]],
            ],
            dtype=torch.uint8,
        ),
        "mask": torch.tensor([[0, 1], [2, 3]], dtype=torch.uint8),
    }

    transformed = EnsureTensorTypes()(sample)

    assert transformed["image"].dtype == torch.float32
    assert transformed["mask"].dtype == torch.int64
    assert torch.all(transformed["image"] >= 0.0)
    assert torch.all(transformed["image"] <= 1.0)
    assert transformed["image"][0, 0, 1].item() == pytest.approx(1.0)


def test_dice_loss_ignores_ignore_index_pixels() -> None:
    target = torch.tensor([[[IGNORE_INDEX, 1]]], dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [[0.0, -4.0]],
                [[0.0, 4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    changed_ignored_pixel = logits.clone()
    changed_ignored_pixel[:, :, 0, 0] = torch.tensor([100.0, -100.0])

    loss_fn = MulticlassDiceLoss(num_classes=2, ignore_index=IGNORE_INDEX)

    assert loss_fn(logits, target).item() == pytest.approx(
        loss_fn(changed_ignored_pixel, target).item()
    )


def test_segmentation_meter_ignores_class_zero_and_computes_iou() -> None:
    target = torch.tensor([[[0, 1, 1], [2, 2, 2]]], dtype=torch.long)
    pred = torch.tensor([[[2, 1, 2], [2, 1, 2]]], dtype=torch.long)
    logits = torch.nn.functional.one_hot(pred, num_classes=3).permute(0, 3, 1, 2).float()

    meter = SegmentationMeter(
        num_classes=3,
        ignore_index=IGNORE_INDEX,
        class_names={0: "ignore", 1: "class1", 2: "class2"},
    )
    meter.update(logits, target)
    summary = meter.compute()

    assert summary.confusion_matrix[0].sum() == 0
    assert summary.per_class_iou["class1"] == pytest.approx(1.0 / 3.0)
    assert summary.per_class_iou["class2"] == pytest.approx(0.5)
    assert summary.mean_iou == pytest.approx((1.0 / 3.0 + 0.5) / 2.0)


def test_segformer_model_builds_eight_class_head_without_pretrained_weights() -> None:
    model = build_segformer_model(
        SegformerBuildConfig(
            variant="segformer-b0",
            num_labels=len(CLASS_NAMES),
            ignore_index=IGNORE_INDEX,
            pretrained=False,
        )
    )

    assert model.config.num_labels == len(CLASS_NAMES)
    assert model.config.semantic_loss_ignore_index == IGNORE_INDEX
    assert model.decode_head.classifier.out_channels == len(CLASS_NAMES)


def test_segformer_model_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported variant"):
        build_segformer_model(
            SegformerBuildConfig(
                variant="segformer-b9",
                num_labels=len(CLASS_NAMES),
                ignore_index=IGNORE_INDEX,
                pretrained=False,
            )
        )
