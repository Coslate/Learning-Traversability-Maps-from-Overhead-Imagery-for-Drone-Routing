from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from loveda_project.inference import (
    SegformerEnsembleInferenceWrapper,
    SegformerInferenceOutput,
    SegformerInferenceWrapper,
    average_logit_maps,
    average_probability_maps,
    build_gaussian_weight_mask,
    output_from_logits,
    semantic_entropy,
)
from scripts.eval_segformer import parse_args


class FixedLogitPredictor:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits

    def _output(self) -> SegformerInferenceOutput:
        return output_from_logits(self.logits)

    def predict(self, pixel_values: torch.Tensor, target_size=None) -> SegformerInferenceOutput:
        return self._output()

    def predict_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
        aggregation: str = "prob-mean",
    ) -> SegformerInferenceOutput:
        return self._output()

    def predict_multiscale_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
        scales,
        aggregation: str = "prob-mean",
    ) -> SegformerInferenceOutput:
        return self._output()


class IdentityLogitsModel(nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        return SimpleNamespace(logits=pixel_values[:, :2])


class LocalColumnLogitsModel(nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        batch_size, _, height, width = pixel_values.shape
        columns = torch.arange(width, device=pixel_values.device, dtype=pixel_values.dtype)
        columns = columns.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return SimpleNamespace(logits=torch.cat([columns, -columns], dim=1))


def test_average_logit_maps_matches_weighted_raw_logit_average() -> None:
    logits_a = torch.tensor([[[[4.0]], [[0.0]]]])
    logits_b = torch.tensor([[[[0.0]], [[0.0]]]])

    averaged_logits = average_logit_maps([logits_a, logits_b], weights=[3.0, 1.0])

    expected_logits = (3.0 * logits_a + logits_b) / 4.0
    assert torch.allclose(averaged_logits, expected_logits)
    assert torch.allclose(output_from_logits(averaged_logits).probabilities, torch.softmax(expected_logits, dim=1))


def test_logit_mean_is_not_probability_mean_on_calibration_sensitive_case() -> None:
    logits_a = torch.tensor([[[[4.0]], [[0.0]]]])
    logits_b = torch.tensor([[[[0.0]], [[0.0]]]])
    probabilities_a = torch.softmax(logits_a, dim=1)
    probabilities_b = torch.softmax(logits_b, dim=1)

    probability_mean = average_probability_maps([probabilities_a, probabilities_b])
    logit_mean = output_from_logits(average_logit_maps([logits_a, logits_b])).probabilities

    assert torch.allclose(logit_mean, torch.softmax((logits_a + logits_b) / 2.0, dim=1))
    assert not torch.allclose(logit_mean, probability_mean)
    assert logit_mean[:, 0] > probability_mean[:, 0]


def test_logit_mean_with_per_class_weights_matches_hand_computed_average() -> None:
    logits_a = torch.tensor([[[[3.0]], [[0.0]], [[-1.0]]]])
    logits_b = torch.tensor([[[[0.0]], [[4.0]], [[1.0]]]])
    per_class_weights = torch.tensor(
        [
            [3.0, 1.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    averaged_logits = average_logit_maps(
        [logits_a, logits_b],
        per_class_weights=per_class_weights,
    )

    expected = torch.empty_like(logits_a)
    expected[:, 0] = (3.0 * logits_a[:, 0] + logits_b[:, 0]) / 4.0
    expected[:, 1] = (logits_a[:, 1] + 3.0 * logits_b[:, 1]) / 4.0
    expected[:, 2] = (logits_a[:, 2] + logits_b[:, 2]) / 2.0
    assert torch.allclose(averaged_logits, expected)
    assert torch.allclose(output_from_logits(averaged_logits).probabilities.sum(dim=1), torch.ones(1, 1, 1))


def test_ensemble_logit_mean_output_uses_final_logits_for_probabilities_mask_and_entropy() -> None:
    logits_a = torch.tensor([[[[4.0, 0.0]], [[0.0, 1.0]]]])
    logits_b = torch.tensor([[[[0.0, 0.0]], [[0.0, 3.0]]]])
    ensemble = SegformerEnsembleInferenceWrapper(
        [
            FixedLogitPredictor(logits_a),
            FixedLogitPredictor(logits_b),
        ],
        weights=[3.0, 1.0],
        aggregation="logit-mean",
    )

    output = ensemble.predict(torch.zeros(1, 1, 1, 2))
    expected_logits = average_logit_maps([logits_a, logits_b], weights=[3.0, 1.0])
    expected_probabilities = torch.softmax(expected_logits, dim=1)

    assert torch.allclose(output.logits, expected_logits)
    assert torch.allclose(output.probabilities, expected_probabilities)
    assert torch.equal(output.masks, torch.argmax(expected_probabilities, dim=1))
    assert torch.allclose(output.entropy, semantic_entropy(expected_probabilities))


def test_sliding_window_logit_aggregation_uses_gaussian_weighted_raw_logits() -> None:
    model = LocalColumnLogitsModel()
    wrapper = SegformerInferenceWrapper(model)
    image = torch.zeros(1, 1, 3, 4)

    output = wrapper.predict_sliding(image, window_size=3, stride=1, aggregation="logit-mean")

    sigma = 3 / 4
    weights = build_gaussian_weight_mask(3, 3, sigma=sigma, dtype=image.dtype)
    left_logits = model(image[:, :, :, 0:3]).logits
    right_logits = model(image[:, :, :, 1:4]).logits

    expected_sum = torch.zeros_like(output.logits)
    expected_weight = torch.zeros(1, 1, 3, 4)
    expected_sum[:, :, :, 0:3] += left_logits * weights
    expected_weight[:, :, :, 0:3] += weights
    expected_sum[:, :, :, 1:4] += right_logits * weights
    expected_weight[:, :, :, 1:4] += weights
    expected_logits = expected_sum / expected_weight

    assert torch.allclose(output.logits, expected_logits, atol=1e-6)
    assert torch.allclose(output.probabilities, torch.softmax(expected_logits, dim=1), atol=1e-6)


def test_logit_sliding_single_window_matches_single_pass_logits() -> None:
    model = IdentityLogitsModel()
    wrapper = SegformerInferenceWrapper(model)
    image = torch.randn(1, 2, 3, 4)

    single_pass = wrapper.predict(image)
    sliding = wrapper.predict_sliding(image, window_size=8, stride=4, aggregation="logit-mean")

    assert torch.allclose(sliding.logits, single_pass.logits)
    assert torch.allclose(sliding.probabilities, single_pass.probabilities)


def test_multiscale_logit_sliding_preserves_output_size_and_class_dimension() -> None:
    model = LocalColumnLogitsModel()
    wrapper = SegformerInferenceWrapper(model)
    image = torch.zeros(2, 1, 5, 7)

    output = wrapper.predict_multiscale_sliding(
        image,
        window_size=4,
        stride=2,
        scales=[0.75, 1.0, 1.25],
        aggregation="logit-mean",
    )

    assert output.logits.shape == (2, 2, 5, 7)
    assert output.probabilities.shape == (2, 2, 5, 7)
    assert output.masks.shape == (2, 5, 7)
    assert output.entropy.shape == (2, 5, 7)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(2, 5, 7), atol=1e-6)


def test_single_checkpoint_logit_mean_path_matches_ordinary_prediction() -> None:
    model = IdentityLogitsModel()
    wrapper = SegformerInferenceWrapper(model)
    image = torch.randn(1, 2, 3, 4)

    ordinary = wrapper.predict(image)
    logit_path = wrapper.predict_multiscale_sliding(
        image,
        window_size=8,
        stride=4,
        scales=[1.0],
        aggregation="logit-mean",
    )

    assert torch.allclose(logit_path.logits, ordinary.logits)
    assert torch.allclose(logit_path.probabilities, ordinary.probabilities)
    assert torch.equal(logit_path.masks, ordinary.masks)


def test_parse_args_accepts_logit_mean_aggregation() -> None:
    args = parse_args(["--checkpoint", "single.pth", "--ensemble-aggregation", "logit-mean"])

    assert args.ensemble_aggregation == "logit-mean"


def test_parse_args_defaults_to_probability_mean_aggregation() -> None:
    args = parse_args(["--checkpoint", "single.pth"])

    assert args.ensemble_aggregation == "prob-mean"


def test_parse_args_rejects_unknown_aggregation() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "single.pth", "--ensemble-aggregation", "not-real"])
