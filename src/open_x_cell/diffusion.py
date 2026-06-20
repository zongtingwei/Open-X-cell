from __future__ import annotations

import math

import torch


def _generator(seed: int | None, device: torch.device | None = None) -> torch.Generator:
    gen = torch.Generator(device=device.type if device is not None and device.type != "mps" else "cpu")
    if seed is not None:
        gen.manual_seed(seed)
    return gen


def make_reveal_mask(
    batch_size: int,
    set_size: int,
    num_genes: int,
    fraction: float,
    seed: int | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1")
    if batch_size <= 0 or set_size <= 0 or num_genes <= 0:
        raise ValueError("batch_size, set_size, and num_genes must be positive")

    reveal_count = math.floor(fraction * num_genes)
    mask = torch.zeros(batch_size, num_genes, dtype=torch.bool)
    gen = _generator(seed)
    for batch_idx in range(batch_size):
        if reveal_count:
            indices = torch.randperm(num_genes, generator=gen)[:reveal_count]
            mask[batch_idx, indices] = True
    mask = mask[:, None, :].expand(batch_size, set_size, num_genes).clone()
    return mask.to(device=device) if device is not None else mask


def mix_control_target(control: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if control.shape != target.shape:
        raise ValueError("control and target must have the same shape")
    if mask.shape != control.shape:
        raise ValueError("mask must have the same shape as control and target")
    return torch.where(mask, target, control)


def cumulative_reveal_steps(
    num_genes: int,
    fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    seed: int | None = None,
    device: torch.device | None = None,
) -> list[torch.Tensor]:
    if num_genes <= 0:
        raise ValueError("num_genes must be positive")
    if tuple(sorted(fractions)) != fractions:
        raise ValueError("fractions must be sorted in ascending order")
    if not fractions or fractions[-1] != 1.0:
        raise ValueError("fractions must end at 1.0")

    gen = _generator(seed)
    ranks = torch.randperm(num_genes, generator=gen)
    previous_count = 0
    steps: list[torch.Tensor] = []
    for fraction in fractions:
        if not 0 <= fraction <= 1:
            raise ValueError("fractions must be between 0 and 1")
        current_count = math.floor(fraction * num_genes)
        if fraction == 1.0:
            current_count = num_genes
        new_indices = ranks[previous_count:current_count]
        step = torch.zeros(num_genes, dtype=torch.bool)
        step[new_indices] = True
        steps.append(step.to(device=device) if device is not None else step)
        previous_count = current_count
    return steps
