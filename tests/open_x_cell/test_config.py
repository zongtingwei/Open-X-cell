from open_x_cell.config import ModelConfig, TrainingConfig, mini_config, tiny_config, ultra_config


def test_tiny_config_is_small_and_internally_consistent():
    cfg = tiny_config()

    assert cfg.num_genes == 32
    assert cfg.context_length == 16
    assert cfg.hidden_dim == 32
    assert cfg.hidden_dim % cfg.num_heads == 0
    assert cfg.cross_attention_layers == (1,)
    assert cfg.prior_dims["esm"] == 5120


def test_mini_config_matches_paper_shape():
    cfg = mini_config()

    assert cfg.num_layers == 12
    assert cfg.hidden_dim == 512
    assert cfg.num_heads == 8
    assert cfg.cross_attention_layers == (2, 5, 8, 11)
    assert cfg.use_qk_norm is False


def test_ultra_config_matches_paper_shape():
    cfg = ultra_config()

    assert cfg.num_layers == 34
    assert cfg.hidden_dim == 2560
    assert cfg.num_heads == 40
    assert cfg.cross_attention_layers[-1] == 33
    assert cfg.use_qk_norm is True
    assert cfg.tied_output is True


def test_model_config_rejects_invalid_attention_width():
    try:
        ModelConfig(hidden_dim=30, num_heads=8)
    except ValueError as exc:
        assert "hidden_dim must be divisible by num_heads" in str(exc)
    else:
        raise AssertionError("invalid attention width was accepted")


def test_training_config_defaults_are_local_smoke_safe():
    cfg = TrainingConfig()

    assert cfg.batch_size == 2
    assert cfg.set_size == 8
    assert cfg.learning_rate == 1e-3
    assert cfg.max_steps == 10
