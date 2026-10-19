import torch

from open_x_cell.config import ModelConfig, mini_config, tiny_config
from open_x_cell.model import CrossAttentionBlock, OpenXCellModel, RMSNorm
from open_x_cell.priors import make_zero_priors


def test_tiny_model_forward_returns_expression_and_embeddings():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    x = torch.rand(2, 4, cfg.context_length)
    mask = torch.zeros(2, 4, cfg.context_length, dtype=torch.bool)
    gene_ids = torch.arange(cfg.context_length).repeat(2, 1)
    perturbation_ids = torch.tensor([1, 2])
    priors = make_zero_priors(batch_size=2, config=cfg)

    out = model(x, mask, gene_ids, perturbation_ids, priors)

    assert out.expression.shape == (2, 4, cfg.context_length)
    assert out.cell_embedding.shape[:2] == (2, 4)
    assert out.prior_reconstructions["esm"].shape == (2, 5120)
    assert torch.isfinite(out.expression).all()


def test_tiny_model_supports_backward_pass():
    cfg = tiny_config()
    model = OpenXCellModel(cfg)
    x = torch.rand(1, 2, cfg.context_length)
    mask = torch.zeros(1, 2, cfg.context_length, dtype=torch.bool)
    gene_ids = torch.arange(cfg.context_length).repeat(1, 1)
    perturbation_ids = torch.tensor([1])

    out = model(x, mask, gene_ids, perturbation_ids, make_zero_priors(1, cfg))
    loss = out.expression.mean()
    loss.backward()

    assert any(param.grad is not None for param in model.parameters())


def test_post_norm_cross_attention_normalizes_both_residuals():
    config = ModelConfig(
        num_genes=4,
        context_length=4,
        num_layers=1,
        hidden_dim=8,
        num_heads=2,
        ffn_dim=8,
        cross_attention_layers=(0,),
        pre_norm=False,
    )
    block = CrossAttentionBlock(config)
    calls = {"attention": 0, "ffn": 0}
    attention_hook = block.query_norm.register_forward_hook(
        lambda *_: calls.__setitem__("attention", calls["attention"] + 1)
    )
    ffn_hook = block.ffn_norm.register_forward_hook(
        lambda *_: calls.__setitem__("ffn", calls["ffn"] + 1)
    )

    block(torch.randn(2, 3, 8), torch.randn(2, 5, 8), None)

    attention_hook.remove()
    ffn_hook.remove()
    assert calls == {"attention": 1, "ffn": 1}


def test_mini_cross_attention_uses_paper_specific_modern_ffn():
    config = mini_config()
    block = CrossAttentionBlock(config)

    assert block.pre_norm is True
    assert isinstance(block.query_norm, RMSNorm)
    assert block.ffn.activation == "swiglu"
    assert block.ffn.gate.out_features == 4 * config.hidden_dim


def test_mini_parameter_count_matches_table_a3_60m_configuration():
    model = OpenXCellModel(mini_config())

    assert sum(parameter.numel() for parameter in model.parameters()) == 60_001_618
