from __future__ import annotations

import json

import pytest
import torch

from loveda_project.data import CLASS_NAMES, IGNORE_INDEX
from loveda_project.inference import (
    SegformerEnsembleInferenceWrapper,
    SegformerInferenceOutput,
    average_probability_maps,
    output_from_probabilities,
    semantic_entropy,
)
from scripts.eval_segformer import load_per_class_ensemble_weights, parse_args


class FixedProbabilityPredictor:
    def __init__(self, probabilities: torch.Tensor) -> None:
        self.probabilities = probabilities

    def _output(self) -> SegformerInferenceOutput:
        return output_from_probabilities(self.probabilities)

    def predict(self, pixel_values: torch.Tensor, target_size=None) -> SegformerInferenceOutput:
        return self._output()

    def predict_sliding(self, pixel_values: torch.Tensor, window_size: int, stride: int) -> SegformerInferenceOutput:
        return self._output()

    def predict_multiscale_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
        scales,
    ) -> SegformerInferenceOutput:
        return self._output()


def _valid_per_class_payload(num_checkpoints: int = 2) -> dict[str, list[float]]:
    return {
        class_name: [1.0] * num_checkpoints
        for class_id, class_name in CLASS_NAMES.items()
        if class_id != IGNORE_INDEX
    }


def test_weighted_probability_maps_match_hand_computed_average() -> None:
    probs_a = torch.tensor([[[[0.8, 0.2]], [[0.2, 0.8]]]], dtype=torch.float32)
    probs_b = torch.tensor([[[[0.4, 0.6]], [[0.6, 0.4]]]], dtype=torch.float32)

    averaged = average_probability_maps([probs_a, probs_b], weights=[2.0, 1.0])

    expected = (2.0 * probs_a + probs_b) / 3.0
    assert torch.allclose(averaged, expected)
    assert torch.allclose(averaged.sum(dim=1), torch.ones(1, 1, 2))


def test_unweighted_probability_maps_still_match_equal_mean() -> None:
    probs_a = torch.tensor([[[[0.9]], [[0.1]]]], dtype=torch.float32)
    probs_b = torch.tensor([[[[0.3]], [[0.7]]]], dtype=torch.float32)

    averaged = average_probability_maps([probs_a, probs_b])

    assert torch.allclose(averaged, (probs_a + probs_b) / 2.0)


def test_per_class_weighted_probability_maps_match_hand_computed_average() -> None:
    probs_a = torch.tensor([[[[0.7]], [[0.2]], [[0.1]]]], dtype=torch.float32)
    probs_b = torch.tensor([[[[0.1]], [[0.7]], [[0.2]]]], dtype=torch.float32)
    per_class_weights = torch.tensor(
        [
            [3.0, 1.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )

    averaged = average_probability_maps(
        [probs_a, probs_b],
        per_class_weights=per_class_weights,
    )

    expected = torch.empty_like(probs_a)
    expected[:, 0] = (3.0 * probs_a[:, 0] + probs_b[:, 0]) / 4.0
    expected[:, 1] = (probs_a[:, 1] + 3.0 * probs_b[:, 1]) / 4.0
    expected[:, 2] = (probs_a[:, 2] + probs_b[:, 2]) / 2.0
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(averaged, expected)
    assert torch.allclose(averaged.sum(dim=1), torch.ones(1, 1, 1))


def test_weighted_ensemble_output_entropy_uses_final_probabilities() -> None:
    probs_a = torch.tensor([[[[0.9, 0.1]], [[0.1, 0.9]]]], dtype=torch.float32)
    probs_b = torch.tensor([[[[0.3, 0.7]], [[0.7, 0.3]]]], dtype=torch.float32)
    ensemble = SegformerEnsembleInferenceWrapper(
        [
            FixedProbabilityPredictor(probs_a),
            FixedProbabilityPredictor(probs_b),
        ],
        weights=[3.0, 1.0],
    )

    output = ensemble.predict(torch.zeros(1, 1, 1, 2))
    expected_probabilities = average_probability_maps([probs_a, probs_b], weights=[3.0, 1.0])

    assert torch.allclose(output.probabilities, expected_probabilities)
    assert torch.equal(output.masks, torch.argmax(expected_probabilities, dim=1))
    assert torch.allclose(output.entropy, semantic_entropy(expected_probabilities))


def test_per_class_ensemble_can_favor_different_predictors_by_class() -> None:
    probs_a = torch.tensor([[[[0.8]], [[0.1]], [[0.1]]]], dtype=torch.float32)
    probs_b = torch.tensor([[[[0.1]], [[0.8]], [[0.1]]]], dtype=torch.float32)
    ensemble = SegformerEnsembleInferenceWrapper(
        [
            FixedProbabilityPredictor(probs_a),
            FixedProbabilityPredictor(probs_b),
        ],
        per_class_weights=[
            [5.0, 1.0],
            [1.0, 5.0],
            [1.0, 1.0],
        ],
    )

    output = ensemble.predict(torch.zeros(1, 1, 1, 1))

    assert output.probabilities[:, 0] > output.probabilities[:, 2]
    assert output.probabilities[:, 1] > output.probabilities[:, 2]


@pytest.mark.parametrize(
    ("weights", "match"),
    [
        ([1.0], "length"),
        ([1.0, -1.0], "non-negative"),
        ([0.0, 0.0], "positive sum"),
        ([1.0, float("inf")], "finite"),
        ([1.0, float("nan")], "finite"),
    ],
)
def test_model_level_weights_are_validated(weights: list[float], match: str) -> None:
    probs_a = torch.ones(1, 2, 1, 1)
    probs_b = torch.ones(1, 2, 1, 1)

    with pytest.raises(ValueError, match=match):
        average_probability_maps([probs_a, probs_b], weights=weights)


@pytest.mark.parametrize(
    ("per_class_weights", "match"),
    [
        ([[1.0, 1.0]], "shape"),
        ([[1.0, -1.0], [1.0, 1.0]], "non-negative"),
        ([[1.0, 1.0], [0.0, 0.0]], "positive weight sum"),
        ([[1.0, float("inf")], [1.0, 1.0]], "finite"),
    ],
)
def test_per_class_weights_are_validated(per_class_weights: list[list[float]], match: str) -> None:
    probs_a = torch.ones(1, 2, 1, 1)
    probs_b = torch.ones(1, 2, 1, 1)

    with pytest.raises(ValueError, match=match):
        average_probability_maps([probs_a, probs_b], per_class_weights=per_class_weights)


def test_rejects_model_and_per_class_weights_together() -> None:
    probs_a = torch.ones(1, 2, 1, 1)
    probs_b = torch.ones(1, 2, 1, 1)

    with pytest.raises(ValueError, match="either ensemble weights or per-class"):
        average_probability_maps(
            [probs_a, probs_b],
            weights=[1.0, 1.0],
            per_class_weights=[[1.0, 1.0], [1.0, 1.0]],
        )


def test_parse_args_accepts_model_level_ensemble_weights() -> None:
    args = parse_args(
        [
            "--checkpoints",
            "a.pth",
            "b.pth",
            "--ensemble-weights",
            "1.0",
            "1.5",
        ]
    )

    assert args.ensemble_weights == [1.0, 1.5]


def test_parse_args_accepts_per_class_weight_path() -> None:
    args = parse_args(
        [
            "--checkpoints",
            "a.pth",
            "b.pth",
            "--per-class-ensemble-weights",
            "weights.json",
        ]
    )

    assert args.per_class_ensemble_weights == "weights.json"


@pytest.mark.parametrize(
    "argv",
    [
        ["--checkpoint", "single.pth", "--ensemble-weights", "1.0"],
        ["--checkpoints", "a.pth", "b.pth", "--ensemble-weights", "1.0"],
        ["--checkpoints", "a.pth", "b.pth", "--ensemble-weights", "1.0", "-1.0"],
        ["--checkpoints", "a.pth", "b.pth", "--ensemble-weights", "0.0", "0.0"],
        [
            "--checkpoints",
            "a.pth",
            "b.pth",
            "--ensemble-weights",
            "1.0",
            "1.0",
            "--per-class-ensemble-weights",
            "weights.json",
        ],
    ],
)
def test_parse_args_rejects_invalid_weight_options(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_load_per_class_ensemble_weights_accepts_valid_json(tmp_path) -> None:
    payload = _valid_per_class_payload(num_checkpoints=2)
    payload["barren"] = [2.0, 0.5]
    weight_path = tmp_path / "weights.json"
    weight_path.write_text(json.dumps(payload), encoding="utf-8")

    weights = load_per_class_ensemble_weights(weight_path, num_checkpoints=2)

    assert weights.shape == (len(CLASS_NAMES), 2)
    assert torch.allclose(weights[IGNORE_INDEX], torch.ones(2))
    assert torch.allclose(weights[5], torch.tensor([2.0, 0.5]))


@pytest.mark.parametrize(
    ("mutate_payload", "match"),
    [
        (lambda payload: payload.pop("road"), "Missing class names"),
        (lambda payload: payload.update({"not_a_class": [1.0, 1.0]}), "Unknown class names"),
        (lambda payload: payload.update({"road": [1.0]}), "length 2"),
        (lambda payload: payload.update({"road": [1.0, -1.0]}), "non-negative"),
        (lambda payload: payload.update({"road": [0.0, 0.0]}), "positive sum"),
        (lambda payload: payload.update({"road": [1.0, float("nan")]}), "finite"),
    ],
)
def test_load_per_class_ensemble_weights_rejects_invalid_json(
    tmp_path,
    mutate_payload,
    match: str,
) -> None:
    payload = _valid_per_class_payload(num_checkpoints=2)
    mutate_payload(payload)
    weight_path = tmp_path / "weights.json"
    weight_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_per_class_ensemble_weights(weight_path, num_checkpoints=2)
