from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from loveda_project.losses import (
    CriterionConfig,
    MulticlassDiceLoss,
    SegmentationCriterion,
    flatten_probabilities,
    lovasz_grad,
    lovasz_softmax_loss,
)
from scripts.train_segformer import parse_args


def test_lovasz_grad_matches_hand_computed_jaccard_steps() -> None:
    sorted_foreground = torch.tensor([1.0, 0.0, 1.0])

    grad = lovasz_grad(sorted_foreground)

    assert grad.tolist() == pytest.approx([0.5, 1.0 / 6.0, 1.0 / 3.0])
    assert grad.sum().item() == pytest.approx(1.0)


def test_flatten_probabilities_drops_ignore_pixels() -> None:
    probabilities = torch.tensor(
        [
            [
                [[0.9, 0.1]],
                [[0.1, 0.9]],
            ]
        ]
    )
    target = torch.tensor([[[0, 1]]], dtype=torch.long)

    flat_probs, flat_target = flatten_probabilities(probabilities, target, ignore_index=0)

    assert flat_probs.shape == (1, 2)
    assert flat_target.tolist() == [1]
    assert flat_probs[0].tolist() == pytest.approx([0.1, 0.9])


def test_lovasz_softmax_loss_is_small_for_nearly_perfect_predictions() -> None:
    logits = torch.tensor(
        [
            [
                [[-12.0, -12.0]],
                [[12.0, -12.0]],
                [[-12.0, 12.0]],
            ]
        ]
    )
    target = torch.tensor([[[1, 2]]], dtype=torch.long)

    loss = lovasz_softmax_loss(logits, target, ignore_index=0)

    assert loss.item() < 1e-5


def test_lovasz_softmax_loss_ignores_ignore_index_pixels() -> None:
    target = torch.tensor([[[0, 1]]], dtype=torch.long)
    logits = torch.tensor(
        [
            [
                [[0.0, -4.0]],
                [[0.0, 4.0]],
                [[0.0, -4.0]],
            ]
        ],
        dtype=torch.float32,
    )
    changed_ignored_pixel = logits.clone()
    changed_ignored_pixel[:, :, 0, 0] = torch.tensor([-100.0, -100.0, 100.0])

    assert lovasz_softmax_loss(logits, target, ignore_index=0).item() == pytest.approx(
        lovasz_softmax_loss(changed_ignored_pixel, target, ignore_index=0).item()
    )


def test_lovasz_softmax_loss_returns_zero_when_all_pixels_are_ignored() -> None:
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    target = torch.zeros(1, 2, 2, dtype=torch.long)

    loss = lovasz_softmax_loss(logits, target, ignore_index=0)
    loss.backward()

    assert loss.item() == pytest.approx(0.0)
    assert torch.all(logits.grad == 0)


def test_lovasz_criterion_matches_direct_lovasz_loss() -> None:
    logits = torch.tensor(
        [
            [
                [[1.0, -0.5]],
                [[0.0, 2.0]],
                [[-1.0, 0.5]],
            ]
        ]
    )
    target = torch.tensor([[[1, 2]]], dtype=torch.long)

    criterion = SegmentationCriterion(CriterionConfig(num_classes=3, ignore_index=0, loss_name="lovasz"))

    assert criterion(logits, target).item() == pytest.approx(
        lovasz_softmax_loss(logits, target, ignore_index=0).item()
    )


def test_ce_lovasz_criterion_matches_weighted_sum_formula() -> None:
    logits = torch.tensor(
        [
            [
                [[1.0, -0.5]],
                [[0.0, 2.0]],
                [[-1.0, 0.5]],
            ]
        ]
    )
    target = torch.tensor([[[1, 2]]], dtype=torch.long)
    class_weights = torch.tensor([0.0, 2.0, 0.5])
    criterion = SegmentationCriterion(
        CriterionConfig(
            num_classes=3,
            ignore_index=0,
            loss_name="ce_lovasz",
            lovasz_weight=0.25,
            class_weights=class_weights,
        )
    )

    expected = F.cross_entropy(logits, target, weight=class_weights, ignore_index=0)
    expected = expected + 0.25 * lovasz_softmax_loss(logits, target, ignore_index=0)

    assert criterion(logits, target).item() == pytest.approx(expected.item())


def test_ce_dice_lovasz_criterion_matches_weighted_sum_formula() -> None:
    logits = torch.tensor(
        [
            [
                [[1.0, -0.5]],
                [[0.0, 2.0]],
                [[-1.0, 0.5]],
            ]
        ]
    )
    target = torch.tensor([[[1, 2]]], dtype=torch.long)
    class_weights = torch.tensor([0.0, 2.0, 0.5])
    criterion = SegmentationCriterion(
        CriterionConfig(
            num_classes=3,
            ignore_index=0,
            loss_name="ce_dice_lovasz",
            dice_weight=0.25,
            lovasz_weight=0.3,
            class_weights=class_weights,
        )
    )
    dice = MulticlassDiceLoss(num_classes=3, ignore_index=0)

    expected = F.cross_entropy(logits, target, weight=class_weights, ignore_index=0)
    expected = expected + 0.25 * dice(logits, target)
    expected = expected + 0.3 * lovasz_softmax_loss(logits, target, ignore_index=0)

    assert criterion(logits, target).item() == pytest.approx(expected.item())


def test_train_cli_accepts_lovasz_losses_and_weight() -> None:
    args = parse_args(["--loss-name", "ce_lovasz", "--lovasz-weight", "0.75"])

    assert args.loss_name == "ce_lovasz"
    assert args.lovasz_weight == pytest.approx(0.75)


def test_train_cli_accepts_ce_dice_lovasz_with_dice_and_lovasz_weights() -> None:
    args = parse_args(
        [
            "--loss-name",
            "ce_dice_lovasz",
            "--dice-weight",
            "0.25",
            "--lovasz-weight",
            "0.3",
        ]
    )

    assert args.loss_name == "ce_dice_lovasz"
    assert args.dice_weight == pytest.approx(0.25)
    assert args.lovasz_weight == pytest.approx(0.3)


def test_train_cli_rejects_negative_lovasz_weight() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--loss-name", "ce_lovasz", "--lovasz-weight", "-0.1"])
