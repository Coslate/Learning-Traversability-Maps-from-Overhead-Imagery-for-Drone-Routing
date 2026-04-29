from __future__ import annotations

import math

import pytest
import torch

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX
from loveda_project.inference import SegformerInferenceWrapper, semantic_entropy
from loveda_project.modeling import SegformerBuildConfig, build_segformer_model


def _build_random_b0() -> torch.nn.Module:
    return build_segformer_model(
        SegformerBuildConfig(
            variant="segformer-b0",
            num_labels=len(CLASS_NAMES),
            ignore_index=IGNORE_INDEX,
            pretrained=False,
        )
    )


def test_inference_outputs_are_upsampled_to_target_size() -> None:
    model = _build_random_b0()
    images = torch.randn(1, 3, 64, 64)

    outputs = SegformerInferenceWrapper(model).predict(images, target_size=(33, 35))

    assert outputs.logits.shape == (1, len(CLASS_NAMES), 33, 35)
    assert outputs.probabilities.shape == outputs.logits.shape
    assert outputs.masks.shape == (1, 33, 35)
    assert outputs.entropy.shape == (1, 33, 35)


def test_inference_softmax_sums_to_one_per_pixel() -> None:
    model = _build_random_b0()
    images = torch.randn(1, 3, 64, 64)

    outputs = SegformerInferenceWrapper(model).predict(images)
    sums = outputs.probabilities.sum(dim=1)

    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_entropy_is_bounded_by_number_of_classes() -> None:
    probabilities = torch.full((1, len(CLASS_NAMES), 2, 2), 1.0 / len(CLASS_NAMES))

    entropy = semantic_entropy(probabilities)

    assert torch.all(entropy >= 0.0)
    assert torch.all(entropy <= math.log(len(CLASS_NAMES)) + 1e-6)
    assert entropy[0, 0, 0].item() == pytest.approx(math.log(len(CLASS_NAMES)))


def test_inference_uses_no_grad_and_restores_training_state() -> None:
    model = _build_random_b0()
    model.train()
    images = torch.randn(1, 3, 64, 64, requires_grad=True)

    outputs = SegformerInferenceWrapper(model).predict(images)

    assert model.training is True
    assert outputs.logits.requires_grad is False
    assert outputs.probabilities.requires_grad is False
    assert outputs.entropy.requires_grad is False


def test_inference_preserves_eval_state() -> None:
    model = _build_random_b0()
    model.eval()
    images = torch.randn(1, 3, 64, 64)

    SegformerInferenceWrapper(model).predict(images)

    assert model.training is False
