from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


CLASS_WEIGHT_MODES = {"none", "inverse", "effective", "median"}


def compute_class_weights(
    class_counts: torch.Tensor,
    mode: str,
    ignore_index: int = 0,
    beta: float = 0.9999,
) -> torch.Tensor | None:
    """Build per-class loss weights from pixel counts.

    The ignore class is always assigned weight 0. All formulas are computed over
    valid non-ignore classes with positive counts.
    """
    if mode not in CLASS_WEIGHT_MODES:
        raise ValueError(f"Unsupported class weight mode: {mode}")
    if mode == "none":
        return None
    if class_counts.ndim != 1:
        raise ValueError("class_counts must be a 1D tensor")

    counts = class_counts.to(dtype=torch.float64)
    valid = counts > 0
    if 0 <= ignore_index < counts.numel():
        valid[ignore_index] = False
    if not bool(valid.any()):
        raise ValueError("class_counts must include at least one non-ignore class with a positive count")

    weights = torch.zeros_like(counts, dtype=torch.float64)
    valid_counts = counts[valid]

    if mode == "inverse":
        total = valid_counts.sum()
        num_valid = valid_counts.numel()
        weights[valid] = total / (num_valid * valid_counts)
    elif mode == "effective":
        raw = (1.0 - beta) / (1.0 - torch.pow(torch.full_like(valid_counts, beta), valid_counts))
        weights[valid] = raw * (valid_counts.numel() / raw.sum())
    elif mode == "median":
        median_count = valid_counts.median()
        weights[valid] = median_count / valid_counts

    if 0 <= ignore_index < counts.numel():
        weights[ignore_index] = 0.0
    return weights.to(dtype=torch.float32)


def focal_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    weight: torch.Tensor | None = None,
    ignore_index: int = 0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute focal cross entropy for segmentation logits."""
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, C, H, W]")
    if target.ndim != 3:
        raise ValueError("target must have shape [B, H, W]")

    valid = target != ignore_index
    safe_target = target.clone()
    safe_target[~valid] = 0

    log_probs = F.log_softmax(logits, dim=1)
    target_log_probs = log_probs.gather(1, safe_target.unsqueeze(1)).squeeze(1)
    ce_loss = -target_log_probs
    pt = target_log_probs.exp()
    focal = torch.pow(1.0 - pt, gamma) * ce_loss

    if weight is not None:
        class_weights = weight.to(device=logits.device, dtype=logits.dtype)
        focal = focal * class_weights[safe_target]

    focal = focal * valid.to(dtype=focal.dtype)
    if reduction == "none":
        return focal
    if reduction == "sum":
        return focal.sum()
    valid_count = valid.sum().clamp_min(1)
    return focal.sum() / valid_count


class MulticlassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_index: int = 0, smooth: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        probs = torch.softmax(logits, dim=1) #[B, C, H, W]

        valid_mask = target != self.ignore_index  # [B, H, W]
        safe_target = target.clone()
        safe_target[~valid_mask] = 0

        one_hot = F.one_hot(safe_target, num_classes=self.num_classes).permute(0, 3, 1, 2).float() #[B, H, W, C] -> [B, C, H, W]
        valid_mask_f = valid_mask.unsqueeze(1).float() #[B, 1, H, W]

        probs = probs * valid_mask_f
        one_hot = one_hot * valid_mask_f

        dims = (0, 2, 3)
        intersection = torch.sum(probs * one_hot, dim=dims) #[C]
        cardinality = torch.sum(probs + one_hot, dim=dims) #[C]
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth) #[C]

        # Exclude ignore channel if ignore_index falls inside [0, num_classes)
        class_mask = torch.ones(self.num_classes, dtype=torch.bool, device=logits.device) #[C]
        if 0 <= self.ignore_index < self.num_classes:
            class_mask[self.ignore_index] = False

        dice = dice[class_mask] #[C-1]
        return 1.0 - dice.mean()


@dataclass
class CriterionConfig:
    num_classes: int
    ignore_index: int = 0
    loss_name: str = "ce"
    dice_weight: float = 0.25
    class_weights: torch.Tensor | None = None
    focal_gamma: float = 2.0


class SegmentationCriterion(nn.Module):
    def __init__(self, cfg: CriterionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss(weight=cfg.class_weights, ignore_index=cfg.ignore_index)
        self.dice = MulticlassDiceLoss(num_classes=cfg.num_classes, ignore_index=cfg.ignore_index)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, target)
        if self.cfg.loss_name == "ce":
            return ce_loss
        if self.cfg.loss_name == "ce_dice":
            dice_loss = self.dice(logits, target)
            return ce_loss + self.cfg.dice_weight * dice_loss
        if self.cfg.loss_name == "focal":
            return focal_cross_entropy_loss(
                logits=logits,
                target=target,
                gamma=self.cfg.focal_gamma,
                weight=self.ce.weight,
                ignore_index=self.cfg.ignore_index,
            )
        raise ValueError(f"Unsupported loss_name: {self.cfg.loss_name}")
