from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class MetricSummary:
    mean_iou: float
    per_class_iou: Dict[str, float]
    pixel_accuracy: float
    confusion_matrix: np.ndarray


class SegmentationMeter:
    def __init__(self, num_classes: int, ignore_index: int, class_names: Dict[int, str]) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = class_names
        self.confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        """
        logits: [B, C, H, W]
        target: [B, H, W]
        """
        pred = torch.argmax(logits, dim=1)
        pred = pred.view(-1)
        target = target.view(-1)

        valid = target != self.ignore_index
        pred = pred[valid]
        target = target[valid]
        if pred.numel() == 0:
            return

        indices = target * self.num_classes + pred
        bincount = torch.bincount(indices, minlength=self.num_classes * self.num_classes)
        self.confusion += bincount.view(self.num_classes, self.num_classes).cpu()

    def compute(self) -> MetricSummary:
        confusion = self.confusion.to(torch.float64)
        tp = torch.diag(confusion)
        gt = confusion.sum(dim=1)
        pred = confusion.sum(dim=0)
        union = gt + pred - tp

        iou = torch.zeros(self.num_classes, dtype=torch.float64)
        valid_cls = union > 0
        iou[valid_cls] = tp[valid_cls] / union[valid_cls]

        eval_mask = torch.ones(self.num_classes, dtype=torch.bool)
        if 0 <= self.ignore_index < self.num_classes:
            eval_mask[self.ignore_index] = False
        eval_mask = eval_mask & valid_cls

        mean_iou = float(iou[eval_mask].mean().item()) if eval_mask.any() else 0.0
        pixel_acc = float(tp.sum().item() / max(confusion.sum().item(), 1.0))

        per_class_iou = {
            self.class_names[i]: float(iou[i].item())
            for i in range(self.num_classes)
            if i != self.ignore_index
        }

        return MetricSummary(
            mean_iou=mean_iou,
            per_class_iou=per_class_iou,
            pixel_accuracy=pixel_acc,
            confusion_matrix=confusion.numpy(),
        )


def save_metrics_json(summary: MetricSummary, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "mean_iou": summary.mean_iou,
                "pixel_accuracy": summary.pixel_accuracy,
                "per_class_iou": summary.per_class_iou,
                "confusion_matrix": summary.confusion_matrix.tolist(),
            },
            f,
            indent=2,
        )


def save_confusion_matrix_plot(
    confusion_matrix: np.ndarray,
    class_names: Sequence[str],
    out_path: str | Path,
    normalize: bool = True,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix.astype(np.float64)
    if normalize:
        row_sum = matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        matrix = matrix / row_sum

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(matrix)
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("ground truth")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = f"{value:.2f}" if normalize else str(int(value))
            ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
