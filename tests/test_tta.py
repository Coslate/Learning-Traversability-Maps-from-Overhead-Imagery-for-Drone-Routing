from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from loveda_project.inference import (
    SegformerInferenceWrapper,
    build_gaussian_weight_mask,
)


class IdentityLogitsModel(nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        return SimpleNamespace(logits=pixel_values[:, :2])


class LocalColumnLogitsModel(nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        batch_size, _, height, width = pixel_values.shape
        columns = torch.arange(width, device=pixel_values.device, dtype=pixel_values.dtype)
        columns = columns.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return SimpleNamespace(logits=torch.cat([columns, -columns], dim=1))


def test_sliding_window_smaller_tile_matches_single_pass_inference() -> None:
    model = IdentityLogitsModel()
    wrapper = SegformerInferenceWrapper(model)
    image = torch.randn(1, 2, 3, 4)

    single_pass = wrapper.predict(image)
    sliding = wrapper.predict_sliding(image, window_size=8, stride=4)

    assert torch.allclose(sliding.probabilities, single_pass.probabilities)
    assert torch.equal(sliding.masks, single_pass.masks)


def test_sliding_window_uses_gaussian_weighted_probability_aggregation() -> None:
    model = LocalColumnLogitsModel()
    wrapper = SegformerInferenceWrapper(model)
    image = torch.zeros(1, 1, 3, 4)

    output = wrapper.predict_sliding(image, window_size=3, stride=1)

    sigma = 3 / 4
    weights = build_gaussian_weight_mask(3, 3, sigma=sigma, dtype=image.dtype)
    left_probs = torch.softmax(model(image[:, :, :, 0:3]).logits, dim=1)
    right_probs = torch.softmax(model(image[:, :, :, 1:4]).logits, dim=1)

    expected_sum = torch.zeros_like(output.probabilities)
    expected_weight = torch.zeros(1, 1, 3, 4)
    expected_sum[:, :, :, 0:3] += left_probs * weights
    expected_weight[:, :, :, 0:3] += weights
    expected_sum[:, :, :, 1:4] += right_probs * weights
    expected_weight[:, :, :, 1:4] += weights
    expected = expected_sum / expected_weight

    assert torch.allclose(output.probabilities, expected, atol=1e-6)


def test_multiscale_sliding_preserves_output_size_and_class_dimension() -> None:
    model = LocalColumnLogitsModel()
    wrapper = SegformerInferenceWrapper(model)
    image = torch.zeros(2, 1, 5, 7)

    output = wrapper.predict_multiscale_sliding(
        image,
        window_size=4,
        stride=2,
        scales=[0.75, 1.0, 1.25],
    )

    assert output.probabilities.shape == (2, 2, 5, 7)
    assert output.logits.shape == (2, 2, 5, 7)
    assert output.masks.shape == (2, 5, 7)
    assert output.entropy.shape == (2, 5, 7)
    assert torch.allclose(
        output.probabilities.sum(dim=1),
        torch.ones(2, 5, 7),
        atol=1e-6,
    )
