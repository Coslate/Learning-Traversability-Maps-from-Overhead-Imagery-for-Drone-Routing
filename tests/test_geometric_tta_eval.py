from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from loveda_project.inference import (
    SegformerEnsembleInferenceWrapper,
    SegformerInferenceOutput,
    SegformerInferenceWrapper,
    apply_geometric_tta_view,
    average_logit_maps,
    average_probability_maps,
    geometric_tta_views,
    invert_geometric_tta_view,
    output_from_logits,
    output_from_probabilities,
    predict_with_geometric_tta,
    semantic_entropy,
)
from scripts.eval_segformer import parse_args


class LocalColumnLogitsModel(nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        batch_size, _, height, width = pixel_values.shape
        columns = torch.arange(width, device=pixel_values.device, dtype=pixel_values.dtype)
        columns = columns.view(1, 1, 1, width).expand(batch_size, 1, height, width)
        return SimpleNamespace(logits=torch.cat([columns, -columns], dim=1))


class IdentityLogitsModel(nn.Module):
    def forward(self, pixel_values: torch.Tensor):
        return SimpleNamespace(logits=pixel_values[:, :2])


class FixedProbabilityPredictor:
    def __init__(self, probabilities: torch.Tensor) -> None:
        self.probabilities = probabilities

    def _output(self) -> SegformerInferenceOutput:
        return output_from_probabilities(self.probabilities)

    def predict(self, pixel_values: torch.Tensor, target_size=None) -> SegformerInferenceOutput:
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


class RecordingSlidingPredictor:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, pixel_values: torch.Tensor, target_size=None) -> SegformerInferenceOutput:
        return output_from_logits(torch.cat([pixel_values[:, :1], -pixel_values[:, :1]], dim=1))

    def predict_multiscale_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
        scales,
        aggregation: str = "prob-mean",
    ) -> SegformerInferenceOutput:
        self.calls.append(
            {
                "shape": tuple(pixel_values.shape),
                "window_size": window_size,
                "stride": stride,
                "scales": tuple(scales),
                "aggregation": aggregation,
            }
        )
        logits = torch.cat([pixel_values[:, :1], -pixel_values[:, :1]], dim=1)
        return output_from_logits(logits)


def test_geometric_views_restore_non_symmetric_tensor() -> None:
    tensor = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)

    for mode in ["none", "hflip", "hvflip", "rot90", "d4"]:
        for view in geometric_tta_views(mode):
            restored = invert_geometric_tta_view(apply_geometric_tta_view(tensor, view), view)
            assert torch.equal(restored, tensor)


def test_rot90_views_swap_rectangular_shape_and_restore_it() -> None:
    tensor = torch.arange(6, dtype=torch.float32).view(1, 1, 2, 3)
    side_rotations = [view for view in geometric_tta_views("rot90") if view.rotation_k % 2 == 1]

    for view in side_rotations:
        transformed = apply_geometric_tta_view(tensor, view)
        restored = invert_geometric_tta_view(transformed, view)

        assert transformed.shape[-2:] == (3, 2)
        assert restored.shape[-2:] == (2, 3)
        assert torch.equal(restored, tensor)


def test_d4_expands_to_eight_deterministic_views_with_identity_first() -> None:
    views = geometric_tta_views("d4")

    assert len(views) == 8
    assert views[0].name == "rot0"
    assert views[0].rotation_k == 0
    assert views[0].hflip is False
    assert len({view.name for view in views}) == 8


def test_parse_args_accepts_geometric_tta_modes() -> None:
    for mode in ["none", "hflip", "hvflip", "rot90", "d4"]:
        args = parse_args(["--checkpoint", "single.pth", "--geometric-tta", mode])
        assert args.geometric_tta == mode


def test_parse_args_defaults_to_no_geometric_tta() -> None:
    args = parse_args(["--checkpoint", "single.pth"])

    assert args.geometric_tta == "none"


def test_parse_args_rejects_unknown_geometric_tta() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--checkpoint", "single.pth", "--geometric-tta", "diagonal-flip"])


def test_geometric_tta_none_matches_base_predictor_exactly() -> None:
    wrapper = SegformerInferenceWrapper(IdentityLogitsModel())
    image = torch.randn(1, 2, 3, 4)

    ordinary = wrapper.predict(image)
    tta = predict_with_geometric_tta(
        wrapper,
        pixel_values=image,
        geometric_tta="none",
        inference_mode="none",
    )

    assert torch.allclose(tta.logits, ordinary.logits)
    assert torch.allclose(tta.probabilities, ordinary.probabilities)
    assert torch.equal(tta.masks, ordinary.masks)
    assert torch.allclose(tta.entropy, ordinary.entropy)


def test_hflip_probability_tta_matches_hand_computed_average() -> None:
    wrapper = SegformerInferenceWrapper(LocalColumnLogitsModel())
    image = torch.zeros(1, 1, 2, 4)
    hflip_view = geometric_tta_views("hflip")[1]

    ordinary_probabilities = wrapper.predict(image).probabilities
    flipped_image = apply_geometric_tta_view(image, hflip_view)
    flipped_probabilities = wrapper.predict(flipped_image).probabilities
    restored_flipped_probabilities = invert_geometric_tta_view(flipped_probabilities, hflip_view)
    expected_probabilities = average_probability_maps(
        [ordinary_probabilities, restored_flipped_probabilities]
    )

    output = predict_with_geometric_tta(
        wrapper,
        pixel_values=image,
        geometric_tta="hflip",
        inference_mode="none",
        aggregation="prob-mean",
    )

    assert torch.allclose(output.probabilities, expected_probabilities, atol=1e-6)
    assert torch.equal(output.masks, torch.argmax(expected_probabilities, dim=1))
    assert torch.allclose(output.entropy, semantic_entropy(expected_probabilities))


def test_rot90_tta_restores_rectangular_output_shape() -> None:
    wrapper = SegformerInferenceWrapper(LocalColumnLogitsModel())
    image = torch.zeros(2, 1, 5, 7)

    output = predict_with_geometric_tta(
        wrapper,
        pixel_values=image,
        geometric_tta="rot90",
        inference_mode="none",
    )

    assert output.logits.shape == (2, 2, 5, 7)
    assert output.probabilities.shape == (2, 2, 5, 7)
    assert output.masks.shape == (2, 5, 7)
    assert output.entropy.shape == (2, 5, 7)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(2, 5, 7), atol=1e-6)


def test_geometric_logit_tta_averages_logits_before_softmax() -> None:
    wrapper = SegformerInferenceWrapper(LocalColumnLogitsModel())
    image = torch.zeros(1, 1, 1, 4)
    hflip_view = geometric_tta_views("hflip")[1]

    ordinary_logits = wrapper.predict(image).logits
    flipped_logits = wrapper.predict(apply_geometric_tta_view(image, hflip_view)).logits
    restored_flipped_logits = invert_geometric_tta_view(flipped_logits, hflip_view)
    expected_logits = average_logit_maps([ordinary_logits, restored_flipped_logits])

    logit_tta = predict_with_geometric_tta(
        wrapper,
        pixel_values=image,
        geometric_tta="hflip",
        inference_mode="none",
        aggregation="logit-mean",
    )
    probability_tta = predict_with_geometric_tta(
        wrapper,
        pixel_values=image,
        geometric_tta="hflip",
        inference_mode="none",
        aggregation="prob-mean",
    )

    assert torch.allclose(logit_tta.logits, expected_logits)
    assert torch.allclose(logit_tta.probabilities, torch.softmax(expected_logits, dim=1))
    assert not torch.allclose(logit_tta.probabilities, probability_tta.probabilities)


def test_geometric_tta_passes_sliding_options_through_to_predictor() -> None:
    predictor = RecordingSlidingPredictor()
    image = torch.randn(1, 1, 3, 4)

    output = predict_with_geometric_tta(
        predictor,
        pixel_values=image,
        geometric_tta="hflip",
        inference_mode="sliding",
        window_size=3,
        stride=2,
        scales=[1.0, 1.25],
        aggregation="logit-mean",
    )

    assert len(predictor.calls) == 2
    assert output.probabilities.shape == (1, 2, 3, 4)
    for call in predictor.calls:
        assert call["window_size"] == 3
        assert call["stride"] == 2
        assert call["scales"] == (1.0, 1.25)
        assert call["aggregation"] == "logit-mean"
        assert call["shape"] == (1, 1, 3, 4)


def test_geometric_tta_preserves_weighted_ensemble_behavior() -> None:
    probabilities_a = torch.tensor([[[[0.9, 0.1]], [[0.1, 0.9]]]], dtype=torch.float32)
    probabilities_b = torch.tensor([[[[0.3, 0.7]], [[0.7, 0.3]]]], dtype=torch.float32)
    ensemble = SegformerEnsembleInferenceWrapper(
        [
            FixedProbabilityPredictor(probabilities_a),
            FixedProbabilityPredictor(probabilities_b),
        ],
        weights=[3.0, 1.0],
    )

    weighted_probabilities = average_probability_maps(
        [probabilities_a, probabilities_b],
        weights=[3.0, 1.0],
    )
    hflip_view = geometric_tta_views("hflip")[1]
    expected_probabilities = average_probability_maps(
        [
            weighted_probabilities,
            invert_geometric_tta_view(weighted_probabilities, hflip_view),
        ]
    )

    output = predict_with_geometric_tta(
        ensemble,
        pixel_values=torch.zeros(1, 1, 1, 2),
        geometric_tta="hflip",
        inference_mode="none",
    )

    assert torch.allclose(output.probabilities, expected_probabilities)


def test_geometric_tta_preserves_per_class_weighted_ensemble_behavior() -> None:
    probabilities_a = torch.tensor([[[[0.7, 0.2]], [[0.2, 0.7]], [[0.1, 0.1]]]], dtype=torch.float32)
    probabilities_b = torch.tensor([[[[0.2, 0.7]], [[0.7, 0.2]], [[0.1, 0.1]]]], dtype=torch.float32)
    per_class_weights = torch.tensor(
        [
            [3.0, 1.0],
            [1.0, 3.0],
            [1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    ensemble = SegformerEnsembleInferenceWrapper(
        [
            FixedProbabilityPredictor(probabilities_a),
            FixedProbabilityPredictor(probabilities_b),
        ],
        per_class_weights=per_class_weights,
    )

    weighted_probabilities = average_probability_maps(
        [probabilities_a, probabilities_b],
        per_class_weights=per_class_weights,
    )
    hflip_view = geometric_tta_views("hflip")[1]
    expected_probabilities = average_probability_maps(
        [
            weighted_probabilities,
            invert_geometric_tta_view(weighted_probabilities, hflip_view),
        ]
    )

    output = predict_with_geometric_tta(
        ensemble,
        pixel_values=torch.zeros(1, 1, 1, 2),
        geometric_tta="hflip",
        inference_mode="none",
    )

    assert torch.allclose(output.probabilities, expected_probabilities)
