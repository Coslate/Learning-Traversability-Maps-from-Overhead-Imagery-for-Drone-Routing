from __future__ import annotations

import sys

import pytest
import torch

from loveda_project.inference import (
    SegformerEnsembleInferenceWrapper,
    SegformerInferenceOutput,
    average_probability_maps,
    output_from_probabilities,
    semantic_entropy,
)
from scripts.eval_segformer import checkpoint_paths_from_args, parse_args


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


def test_average_probability_maps_averages_classwise_per_pixel() -> None:
    probs_a = torch.tensor([[[[0.8, 0.2]], [[0.2, 0.8]]]], dtype=torch.float32)
    probs_b = torch.tensor([[[[0.4, 0.6]], [[0.6, 0.4]]]], dtype=torch.float32)

    averaged = average_probability_maps([probs_a, probs_b])

    expected = torch.tensor([[[[0.6, 0.4]], [[0.4, 0.6]]]], dtype=torch.float32)
    assert torch.allclose(averaged, expected)
    assert torch.allclose(averaged.sum(dim=1), torch.ones(1, 1, 2))


def test_ensemble_output_shapes_and_entropy_use_averaged_probabilities() -> None:
    probs_a = torch.tensor([[[[0.9, 0.1]], [[0.1, 0.9]]]], dtype=torch.float32)
    probs_b = torch.tensor([[[[0.3, 0.7]], [[0.7, 0.3]]]], dtype=torch.float32)
    ensemble = SegformerEnsembleInferenceWrapper(
        [
            FixedProbabilityPredictor(probs_a),
            FixedProbabilityPredictor(probs_b),
        ]
    )

    output = ensemble.predict(torch.zeros(1, 1, 1, 2))
    expected_probabilities = average_probability_maps([probs_a, probs_b])

    assert output.probabilities.shape == (1, 2, 1, 2)
    assert output.masks.shape == (1, 1, 2)
    assert output.entropy.shape == (1, 1, 2)
    assert torch.allclose(output.probabilities, expected_probabilities)
    assert torch.equal(output.masks, torch.argmax(expected_probabilities, dim=1))
    assert torch.allclose(output.entropy, semantic_entropy(expected_probabilities))


def test_ensemble_rejects_empty_predictor_list() -> None:
    with pytest.raises(ValueError, match="at least one predictor"):
        SegformerEnsembleInferenceWrapper([])


def test_ensemble_rejects_incompatible_class_dimensions() -> None:
    probs_a = torch.ones(1, 2, 1, 1)
    probs_b = torch.ones(1, 3, 1, 1)

    with pytest.raises(ValueError, match="same class dimension"):
        average_probability_maps([probs_a, probs_b])


def test_parse_args_keeps_single_checkpoint_path(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_segformer.py", "--checkpoint", "single.pth"])

    args = parse_args()

    assert args.checkpoint == "single.pth"
    assert args.checkpoints is None
    assert checkpoint_paths_from_args(args) == ["single.pth"]


def test_parse_args_accepts_multiple_checkpoints(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_segformer.py", "--checkpoints", "a.pth", "b.pth"])

    args = parse_args()

    assert args.checkpoint is None
    assert args.checkpoints == ["a.pth", "b.pth"]
    assert checkpoint_paths_from_args(args) == ["a.pth", "b.pth"]


def test_parse_args_rejects_empty_checkpoints(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["eval_segformer.py", "--checkpoints"])

    with pytest.raises(SystemExit):
        parse_args()


def test_parse_args_rejects_variant_override_for_ensemble(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["eval_segformer.py", "--checkpoints", "a.pth", "b.pth", "--variant", "segformer-b2"],
    )

    with pytest.raises(SystemExit):
        parse_args()
