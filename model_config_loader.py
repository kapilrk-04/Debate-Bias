"""Load per-model vLLM configuration from model_configs.yaml."""

import os
import yaml


def load_model_config(model_path, config_file="model_configs.yaml"):
    """Load config for a specific model, falling back to defaults."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    model_overrides = config.get("models", {}).get(model_path, {})

    merged = {**defaults, **model_overrides}
    return merged
