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
