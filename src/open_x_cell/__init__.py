from open_x_cell.config import LossConfig, ModelConfig, TrainingConfig, mini_config, model_config, tiny_config, ultra_config

__all__ = [
    "OpenXCell",
    "LossConfig",
    "ModelConfig",
    "TrainingConfig",
    "mini_config",
    "model_config",
    "tiny_config",
    "ultra_config",
]


def __getattr__(name):
    if name == "OpenXCell":
        from open_x_cell.api import OpenXCell
        return OpenXCell
    raise AttributeError(name)
