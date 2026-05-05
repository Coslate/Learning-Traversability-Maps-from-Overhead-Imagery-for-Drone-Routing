from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from loveda_project.modeling import SegformerBuildConfig, build_segformer_model


@dataclass
class SegformerInferenceOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    masks: torch.Tensor
    entropy: torch.Tensor


def semantic_entropy(probabilities: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute per-pixel entropy for class probabilities.

    Args:
        probabilities: Tensor shaped [B, C, H, W].
        eps: Clamp floor used only inside log to avoid log(0).
    """
    if probabilities.ndim != 4:
        raise ValueError("probabilities must have shape [B, C, H, W]")
    return -(probabilities * probabilities.clamp_min(eps).log()).sum(dim=1)


def sliding_window_start_positions(length: int, window_size: int, stride: int) -> list[int]:
    """Return deterministic crop starts that always cover the full axis."""
    if length <= 0:
        raise ValueError("length must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")

    if length <= window_size:
        return [0]

    last_start = length - window_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def build_gaussian_weight_mask(
    height: int,
    width: int,
    sigma: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a center-weighted [1, 1, H, W] overlap mask with max value 1."""
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    weights = torch.exp(-((yy - cy).square() + (xx - cx).square()) / (2.0 * sigma * sigma))
    weights = weights / weights.max().clamp_min(1e-12)
    return weights.unsqueeze(0).unsqueeze(0)


def _normalize_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-12)


def output_from_probabilities(probabilities: torch.Tensor) -> SegformerInferenceOutput:
    probabilities = _normalize_probabilities(probabilities)
    masks = torch.argmax(probabilities, dim=1)
    entropy = semantic_entropy(probabilities)
    log_probabilities = probabilities.clamp_min(1e-12).log()
    return SegformerInferenceOutput(
        logits=log_probabilities,
        probabilities=probabilities,
        masks=masks,
        entropy=entropy,
    )


def _validate_probability_maps(probability_maps: Sequence[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Size]:
    if not probability_maps:
        raise ValueError("probability_maps must contain at least one tensor")

    maps = list(probability_maps)
    reference = maps[0]
    if reference.ndim != 4:
        raise ValueError("probability maps must have shape [B, C, H, W]")

    reference_shape = reference.shape
    reference_num_classes = reference_shape[1]
    for probabilities in maps:
        if probabilities.ndim != 4:
            raise ValueError("probability maps must have shape [B, C, H, W]")
        if probabilities.shape[1] != reference_num_classes:
            raise ValueError("All probability maps must have the same class dimension")
        if probabilities.shape != reference_shape:
            raise ValueError("All probability maps must have the same shape")
    return maps, reference_shape


def _weights_tensor(
    weights: Sequence[float] | torch.Tensor,
    *,
    num_maps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights_tensor = torch.as_tensor(weights, device=device, dtype=dtype)
    if weights_tensor.ndim != 1:
        raise ValueError("ensemble weights must be a 1D sequence")
    if weights_tensor.numel() != num_maps:
        raise ValueError("ensemble weights length must match the number of probability maps")
    if not bool(torch.isfinite(weights_tensor).all()):
        raise ValueError("ensemble weights must be finite")
    if bool((weights_tensor < 0).any()):
        raise ValueError("ensemble weights must be non-negative")
    if float(weights_tensor.sum().item()) <= 0.0:
        raise ValueError("ensemble weights must have a positive sum")
    return weights_tensor


def _per_class_weights_tensor(
    per_class_weights: Sequence[Sequence[float]] | torch.Tensor,
    *,
    num_classes: int,
    num_maps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights_tensor = torch.as_tensor(per_class_weights, device=device, dtype=dtype)
    if weights_tensor.ndim != 2:
        raise ValueError("per-class ensemble weights must have shape [C, N]")
    if tuple(weights_tensor.shape) != (num_classes, num_maps):
        raise ValueError("per-class ensemble weights must have shape [num_classes, num_probability_maps]")
    if not bool(torch.isfinite(weights_tensor).all()):
        raise ValueError("per-class ensemble weights must be finite")
    if bool((weights_tensor < 0).any()):
        raise ValueError("per-class ensemble weights must be non-negative")
    if bool((weights_tensor.sum(dim=1) <= 0).any()):
        raise ValueError("each class in per-class ensemble weights must have a positive weight sum")
    return weights_tensor


def average_probability_maps(
    probability_maps: Sequence[torch.Tensor],
    weights: Sequence[float] | torch.Tensor | None = None,
    per_class_weights: Sequence[Sequence[float]] | torch.Tensor | None = None,
) -> torch.Tensor:
    """Average probability maps with optional model-level or per-class weights."""
    if weights is not None and per_class_weights is not None:
        raise ValueError("Use either ensemble weights or per-class ensemble weights, not both")

    maps, reference_shape = _validate_probability_maps(probability_maps)
    num_maps = len(maps)
    num_classes = reference_shape[1]
    stacked = torch.stack(maps, dim=0)

    if weights is not None:
        weights_tensor = _weights_tensor(
            weights,
            num_maps=num_maps,
            device=stacked.device,
            dtype=stacked.dtype,
        )
        weighted = stacked * weights_tensor.view(num_maps, 1, 1, 1, 1)
        probabilities = weighted.sum(dim=0) / weights_tensor.sum().clamp_min(1e-12)
        return _normalize_probabilities(probabilities)

    if per_class_weights is not None:
        weights_tensor = _per_class_weights_tensor(
            per_class_weights,
            num_classes=num_classes,
            num_maps=num_maps,
            device=stacked.device,
            dtype=stacked.dtype,
        )
        weighted = stacked * weights_tensor.T.view(num_maps, 1, num_classes, 1, 1)
        class_weight_sum = weights_tensor.sum(dim=1).view(1, num_classes, 1, 1).clamp_min(1e-12)
        probabilities = weighted.sum(dim=0) / class_weight_sum
        return _normalize_probabilities(probabilities)

    return _normalize_probabilities(stacked.mean(dim=0))


class SegformerInferenceWrapper:
    """Frozen SegFormer predictor returning upsampled dense outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def _forward_logits(
        self,
        pixel_values: torch.Tensor,
        target_size: Sequence[int] | torch.Size | None = None,
    ) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits
        output_size = tuple(target_size) if target_size is not None else tuple(pixel_values.shape[-2:])
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits

    @staticmethod
    def _output_from_probabilities(probabilities: torch.Tensor) -> SegformerInferenceOutput:
        return output_from_probabilities(probabilities)

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        target_size: Sequence[int] | torch.Size | None = None,
    ) -> SegformerInferenceOutput:
        was_training = self.model.training
        self.model.eval()
        try:
            logits = self._forward_logits(pixel_values=pixel_values, target_size=target_size)
            probabilities = torch.softmax(logits, dim=1)
            masks = torch.argmax(probabilities, dim=1)
            entropy = semantic_entropy(probabilities)
            return SegformerInferenceOutput(
                logits=logits,
                probabilities=probabilities,
                masks=masks,
                entropy=entropy,
            )
        finally:
            self.model.train(was_training)

    def _sliding_window_probabilities(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
    ) -> torch.Tensor:
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [B, C, H, W]")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")

        _, _, height, width = pixel_values.shape
        if height <= window_size and width <= window_size:
            logits = self._forward_logits(pixel_values=pixel_values, target_size=(height, width))
            return torch.softmax(logits, dim=1)

        y_starts = sliding_window_start_positions(height, window_size, stride)
        x_starts = sliding_window_start_positions(width, window_size, stride)
        sigma = window_size / 4.0

        probability_sum: torch.Tensor | None = None
        weight_sum: torch.Tensor | None = None

        for top in y_starts:
            bottom = min(top + window_size, height)
            for left in x_starts:
                right = min(left + window_size, width)
                crop = pixel_values[:, :, top:bottom, left:right]
                crop_h, crop_w = crop.shape[-2:]
                crop_logits = self._forward_logits(pixel_values=crop, target_size=(crop_h, crop_w))
                crop_probabilities = torch.softmax(crop_logits, dim=1)

                if probability_sum is None:
                    batch_size, num_classes = crop_probabilities.shape[:2]
                    probability_sum = torch.zeros(
                        (batch_size, num_classes, height, width),
                        device=crop_probabilities.device,
                        dtype=crop_probabilities.dtype,
                    )
                    weight_sum = torch.zeros(
                        (batch_size, 1, height, width),
                        device=crop_probabilities.device,
                        dtype=crop_probabilities.dtype,
                    )

                weights = build_gaussian_weight_mask(
                    crop_h,
                    crop_w,
                    sigma=sigma,
                    device=crop_probabilities.device,
                    dtype=crop_probabilities.dtype,
                )
                probability_sum[:, :, top:bottom, left:right] += crop_probabilities * weights
                weight_sum[:, :, top:bottom, left:right] += weights

        if probability_sum is None or weight_sum is None:
            raise RuntimeError("sliding-window inference produced no windows")

        return _normalize_probabilities(probability_sum / weight_sum.clamp_min(1e-12))

    @torch.no_grad()
    def predict_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
    ) -> SegformerInferenceOutput:
        was_training = self.model.training
        self.model.eval()
        try:
            probabilities = self._sliding_window_probabilities(
                pixel_values=pixel_values,
                window_size=window_size,
                stride=stride,
            )
            return self._output_from_probabilities(probabilities)
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def predict_multiscale_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
        scales: Sequence[float],
    ) -> SegformerInferenceOutput:
        if not scales:
            raise ValueError("scales must contain at least one value")
        if any(scale <= 0 for scale in scales):
            raise ValueError("all scales must be positive")

        was_training = self.model.training
        self.model.eval()
        try:
            original_size = tuple(pixel_values.shape[-2:])
            scaled_probabilities = []
            for scale in scales:
                if scale == 1.0:
                    scaled_pixels = pixel_values
                else:
                    scaled_size = (
                        max(1, int(round(original_size[0] * scale))),
                        max(1, int(round(original_size[1] * scale))),
                    )
                    scaled_pixels = F.interpolate(
                        pixel_values,
                        size=scaled_size,
                        mode="bilinear",
                        align_corners=False,
                    )

                probabilities = self._sliding_window_probabilities(
                    pixel_values=scaled_pixels,
                    window_size=window_size,
                    stride=stride,
                )
                if probabilities.shape[-2:] != original_size:
                    probabilities = F.interpolate(
                        probabilities,
                        size=original_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                scaled_probabilities.append(_normalize_probabilities(probabilities))

            probabilities = torch.stack(scaled_probabilities, dim=0).mean(dim=0)
            return self._output_from_probabilities(probabilities)
        finally:
            self.model.train(was_training)


class SegformerEnsembleInferenceWrapper:
    """Average probability predictions from multiple frozen SegFormer predictors."""

    def __init__(
        self,
        predictors: Sequence[SegformerInferenceWrapper],
        weights: Sequence[float] | torch.Tensor | None = None,
        per_class_weights: Sequence[Sequence[float]] | torch.Tensor | None = None,
    ) -> None:
        if not predictors:
            raise ValueError("predictors must contain at least one predictor")
        if weights is not None and per_class_weights is not None:
            raise ValueError("Use either ensemble weights or per-class ensemble weights, not both")
        if weights is not None:
            # Validate the number of predictors early; dtype/device validation happens
            # again on the output tensors so CPU/GPU inputs both work.
            _weights_tensor(
                weights,
                num_maps=len(predictors),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
        if per_class_weights is not None:
            per_class_weights_tensor = torch.as_tensor(per_class_weights, dtype=torch.float32)
            if per_class_weights_tensor.ndim != 2:
                raise ValueError("per-class ensemble weights must have shape [C, N]")
            if per_class_weights_tensor.shape[1] != len(predictors):
                raise ValueError("per-class ensemble weights must have one column per predictor")
            if not bool(torch.isfinite(per_class_weights_tensor).all()):
                raise ValueError("per-class ensemble weights must be finite")
            if bool((per_class_weights_tensor < 0).any()):
                raise ValueError("per-class ensemble weights must be non-negative")
            if bool((per_class_weights_tensor.sum(dim=1) <= 0).any()):
                raise ValueError("each class in per-class ensemble weights must have a positive weight sum")
        self.predictors = list(predictors)
        self.weights = weights
        self.per_class_weights = per_class_weights

    def _output_from_members(self, outputs: Sequence[SegformerInferenceOutput]) -> SegformerInferenceOutput:
        probabilities = average_probability_maps(
            [output.probabilities for output in outputs],
            weights=self.weights,
            per_class_weights=self.per_class_weights,
        )
        return output_from_probabilities(probabilities)

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        target_size: Sequence[int] | torch.Size | None = None,
    ) -> SegformerInferenceOutput:
        outputs = [
            predictor.predict(pixel_values=pixel_values, target_size=target_size)
            for predictor in self.predictors
        ]
        return self._output_from_members(outputs)

    @torch.no_grad()
    def predict_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
    ) -> SegformerInferenceOutput:
        outputs = [
            predictor.predict_sliding(
                pixel_values=pixel_values,
                window_size=window_size,
                stride=stride,
            )
            for predictor in self.predictors
        ]
        return self._output_from_members(outputs)

    @torch.no_grad()
    def predict_multiscale_sliding(
        self,
        pixel_values: torch.Tensor,
        window_size: int,
        stride: int,
        scales: Sequence[float],
    ) -> SegformerInferenceOutput:
        outputs = [
            predictor.predict_multiscale_sliding(
                pixel_values=pixel_values,
                window_size=window_size,
                stride=stride,
                scales=scales,
            )
            for predictor in self.predictors
        ]
        return self._output_from_members(outputs)


def predict_segformer(
    model: torch.nn.Module,
    pixel_values: torch.Tensor,
    target_size: Sequence[int] | torch.Size | None = None,
) -> SegformerInferenceOutput:
    return SegformerInferenceWrapper(model).predict(pixel_values=pixel_values, target_size=target_size)


def load_segformer_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    num_labels: int,
    ignore_index: int,
    variant: str | None = None,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_args = checkpoint.get("args", {})
    model_variant = variant or checkpoint_args.get("variant", "segformer-b0")
    model = build_segformer_model(
        SegformerBuildConfig(
            variant=model_variant,
            num_labels=num_labels,
            ignore_index=ignore_index,
            pretrained=False,
        )
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model
