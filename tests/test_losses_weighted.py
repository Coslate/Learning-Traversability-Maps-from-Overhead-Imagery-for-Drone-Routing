from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from loveda_project.losses import (
    CriterionConfig,
    MulticlassDiceLoss,
    SegmentationCriterion,
    compute_class_weights,
    focal_cross_entropy_loss,
)


def test_inverse_class_weights_match_hand_computed_values() -> None:
    counts = torch.tensor([100.0, 2.0, 6.0])

    weights = compute_class_weights(counts, mode="inverse", ignore_index=0)

    assert weights is not None
    assert weights[0].item() == 0.0
    assert weights[1].item() == pytest.approx(8.0 / (2.0 * 2.0))
    assert weights[2].item() == pytest.approx(8.0 / (2.0 * 6.0))


def test_effective_class_weights_are_normalized_over_non_ignore_classes() -> None:
    counts = torch.tensor([100.0, 2.0, 6.0])
    beta = 0.9999
    raw = torch.tensor(
        [
            (1.0 - beta) / (1.0 - beta**2.0),
            (1.0 - beta) / (1.0 - beta**6.0),
        ],
        dtype=torch.float64,
    )
    expected = raw * (2.0 / raw.sum())

    weights = compute_class_weights(counts, mode="effective", ignore_index=0, beta=beta)

    assert weights is not None
    assert weights[0].item() == 0.0
    assert weights[1].item() == pytest.approx(float(expected[0]))
    assert weights[2].item() == pytest.approx(float(expected[1]))
    assert weights.sum().item() == pytest.approx(2.0)


@pytest.mark.parametrize("mode", ["inverse", "effective", "median"])
def test_ignore_class_weight_is_zero_for_weighted_modes(mode: str) -> None:
    counts = torch.tensor([100.0, 2.0, 6.0, 10.0])

    weights = compute_class_weights(counts, mode=mode, ignore_index=0)

    assert weights is not None
    assert weights[0].item() == 0.0


def test_median_class_weights_match_hand_computed_values() -> None:
    counts = torch.tensor([100.0, 2.0, 6.0, 10.0])

    weights = compute_class_weights(counts, mode="median", ignore_index=0)

    assert weights is not None
    assert weights.tolist() == pytest.approx([0.0, 3.0, 1.0, 0.6])


def test_focal_loss_downweights_easy_examples() -> None:
    logits = torch.tensor(
        [
            [
                [[-5.0, 5.0]],
                [[5.0, -5.0]],
            ]
        ]
    )
    target = torch.tensor([[[1, 1]]], dtype=torch.long)

    loss = focal_cross_entropy_loss(logits, target, gamma=2.0, ignore_index=0, reduction="none")

    assert loss[0, 0, 0].item() < loss[0, 0, 1].item()


def test_focal_loss_with_inverse_weights_matches_elementwise_formula() -> None:
    logits = torch.tensor(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 2.0]],
                [[3.0, 1.0]],
            ]
        ]
    )
    target = torch.tensor([[[1, 2]]], dtype=torch.long)
    weights = compute_class_weights(torch.tensor([50.0, 2.0, 6.0]), mode="inverse", ignore_index=0)
    assert weights is not None

    actual = focal_cross_entropy_loss(
        logits,
        target,
        gamma=2.0,
        weight=weights,
        ignore_index=0,
        reduction="none",
    )
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    expected = torch.pow(1.0 - pt, 2.0) * ce * weights[target]

    assert torch.allclose(actual, expected)


def test_weighted_ce_criterion_matches_torch_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 2.0]],
                [[3.0, 1.0]],
            ]
        ]
    )
    target = torch.tensor([[[1, 2]]], dtype=torch.long)
    weights = compute_class_weights(torch.tensor([50.0, 2.0, 6.0]), mode="inverse", ignore_index=0)
    assert weights is not None

    criterion = SegmentationCriterion(
        CriterionConfig(num_classes=3, ignore_index=0, loss_name="ce", class_weights=weights)
    )

    assert criterion(logits, target).item() == pytest.approx(
        F.cross_entropy(logits, target, weight=weights, ignore_index=0).item()
    )


def test_existing_ce_and_ce_dice_paths_match_unweighted_formulas() -> None:
    logits = torch.tensor(
        [
            [
                [[1.0, 0.0]],
                [[0.0, 2.0]],
                [[3.0, 1.0]],
            ]
        ]
    )
    target = torch.tensor([[[1, 2]]], dtype=torch.long)
    ce_criterion = SegmentationCriterion(CriterionConfig(num_classes=3, ignore_index=0, loss_name="ce"))
    ce_dice_criterion = SegmentationCriterion(
        CriterionConfig(num_classes=3, ignore_index=0, loss_name="ce_dice", dice_weight=0.25)
    )
    dice = MulticlassDiceLoss(num_classes=3, ignore_index=0)

    expected_ce = F.cross_entropy(logits, target, ignore_index=0)
    expected_ce_dice = expected_ce + 0.25 * dice(logits, target)

    assert ce_criterion(logits, target).item() == pytest.approx(expected_ce.item())
    assert ce_dice_criterion(logits, target).item() == pytest.approx(expected_ce_dice.item())
