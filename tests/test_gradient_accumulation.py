from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from scripts.train_segformer import train_one_epoch


class TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(1, 2, 1, 1))

    def forward(self, pixel_values: torch.Tensor):
        batch_size, _, height, width = pixel_values.shape
        logits = self.logits.expand(batch_size, 2, height, width)
        return SimpleNamespace(logits=logits)


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, lr: float) -> None:
        super().__init__(params, lr=lr)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure=closure)


class CountingScheduler:
    def __init__(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


def _batch() -> dict:
    return {
        "image": torch.ones(1, 3, 2, 2),
        "mask": torch.zeros(1, 2, 2, dtype=torch.long),
    }


def test_gradient_accumulation_steps_optimizer_after_configured_micro_batches() -> None:
    model = TinySegmentationModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    scheduler = CountingScheduler()
    loader = [_batch(), _batch(), _batch(), _batch()]

    train_one_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        scaler=None,
        amp_enabled=False,
        grad_accum_steps=2,
    )

    assert optimizer.step_count == 2
    assert scheduler.step_count == 2


def test_gradient_accumulation_steps_on_final_leftover_batch() -> None:
    model = TinySegmentationModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    loader = [_batch(), _batch(), _batch()]

    train_one_epoch(
        model=model,
        loader=loader,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=None,
        device=torch.device("cpu"),
        scaler=None,
        amp_enabled=False,
        grad_accum_steps=2,
    )

    assert optimizer.step_count == 2
