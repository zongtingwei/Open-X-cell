from __future__ import annotations

from dataclasses import dataclass

import torch

from open_x_cell.config import ModelConfig


PRIOR_SOURCES: tuple[str, ...] = ("esm", "string", "genept", "depmap", "cell_painting")


@dataclass(frozen=True)
class PriorBatch:
    values: dict[str, torch.Tensor]
    missing: torch.Tensor
    source_names: tuple[str, ...] = PRIOR_SOURCES

    def to(self, device: torch.device | str) -> PriorBatch:
        target = torch.device(device)
        return PriorBatch(
            values={name: value.to(target) for name, value in self.values.items()},
            missing=self.missing.to(target),
            source_names=self.source_names,
        )


def make_zero_priors(batch_size: int, config: ModelConfig, device: torch.device | str | None = None) -> PriorBatch:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    target = torch.device(device) if device is not None else None
    values = {
        source: torch.zeros(batch_size, config.prior_dims[source], dtype=torch.float32, device=target)
        for source in PRIOR_SOURCES
    }
    missing = torch.ones(batch_size, len(PRIOR_SOURCES), dtype=torch.bool, device=target)
    return PriorBatch(values=values, missing=missing, source_names=PRIOR_SOURCES)
