from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        raise ValueError(f"Unsupported loss_name: {self.cfg.loss_name}")
