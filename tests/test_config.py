from lingshu_cell import (
    ModelConfig,
    TrainingConfig,
    paper_model_config,
    tiny_model_config,
)


def test_tiny_profile_is_small_and_internally_consistent():
    cfg = tiny_model_config()

    assert cfg.gene_length == 16
    assert cfg.expression_vocab_size == 281
    assert cfg.mask_token_id == 281
    assert cfg.total_vocab_size > cfg.mask_token_id
    assert cfg.embedding_dim % cfg.num_heads == 0
    assert cfg.head_dim == cfg.embedding_dim // cfg.num_heads
    assert cfg.compressed_gene_length == 4


def test_paper_profile_matches_reported_hyperparameters():
    cfg = paper_model_config()

    assert cfg.gene_length == 18_080
    assert cfg.embedding_dim == 640
    assert cfg.num_layers == 13
    assert cfg.num_heads == 10
    assert cfg.head_dim == 64
    assert cfg.ffn_dim == 2_560
    assert cfg.compression_group_size == 8
    assert cfg.compressed_gene_length == 2_260


def test_genetic_paper_profile_uses_group_size_32():
    cfg = paper_model_config(task="genetic")

    assert cfg.compression_group_size == 32
    assert cfg.compressed_gene_length == 565


def test_config_validation_rejects_incompatible_attention_width():
    try:
        ModelConfig(embedding_dim=10, num_heads=3, gene_length=8)
    except ValueError as exc:
        assert "embedding_dim must be divisible" in str(exc)
    else:
        raise AssertionError("ModelConfig accepted incompatible head dimensions")


def test_training_config_defaults_follow_paper_values():
    cfg = TrainingConfig()

    assert cfg.beta1 == 0.9
    assert cfg.beta2 == 0.95
    assert cfg.weight_decay == 0.1
    assert cfg.grad_clip_norm == 1.0
    assert cfg.min_lr == 2e-5

