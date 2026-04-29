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


class SegformerInferenceWrapper:
    """Frozen SegFormer predictor returning upsampled dense outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    @torch.no_grad()
    def predict(
        self,
        pixel_values: torch.Tensor,
        target_size: Sequence[int] | torch.Size | None = None,
    ) -> SegformerInferenceOutput:
        was_training = self.model.training
        self.model.eval()
        try:
            outputs = self.model(pixel_values=pixel_values)
            logits = outputs.logits
            output_size = tuple(target_size) if target_size is not None else tuple(pixel_values.shape[-2:])
            if logits.shape[-2:] != output_size:
                logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)

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
