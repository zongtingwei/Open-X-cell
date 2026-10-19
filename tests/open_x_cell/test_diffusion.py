import torch

from open_x_cell.diffusion import cumulative_reveal_steps, make_reveal_mask, mix_control_target


def test_make_reveal_mask_reveals_exact_fraction_per_sample():
    mask = make_reveal_mask(batch_size=2, set_size=3, num_genes=8, fraction=0.25, seed=7)

    assert mask.shape == (2, 3, 8)
    assert mask.dtype == torch.bool
    assert mask[0, 0].sum().item() == 2
    assert torch.equal(mask[0, 0], mask[0, 1])


def test_make_reveal_mask_accepts_independent_fraction_per_sample():
    mask = make_reveal_mask(
        batch_size=2,
        set_size=3,
        num_genes=20,
        fraction=(0.25, 0.75),
        seed=9,
    )

    assert mask[0, 0].sum().item() == 5
    assert mask[1, 0].sum().item() == 15
    assert torch.equal(mask[:, 0], mask[:, 2])


def test_mix_control_target_uses_target_where_mask_is_true():
    ctrl = torch.zeros(1, 2, 4)
    target = torch.ones(1, 2, 4)
    mask = torch.tensor([[[True, False, True, False], [False, True, False, True]]])

    mixed = mix_control_target(ctrl, target, mask)

    assert mixed.tolist() == [[[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]]


def test_cumulative_reveal_steps_partition_genes():
    steps = cumulative_reveal_steps(num_genes=8, fractions=(0.25, 0.5, 0.75, 1.0), seed=3)

    assert len(steps) == 4
    assert [step.sum().item() for step in steps] == [2, 2, 2, 2]
    total = torch.stack(steps).sum(dim=0)
    assert torch.all(total == 1)
