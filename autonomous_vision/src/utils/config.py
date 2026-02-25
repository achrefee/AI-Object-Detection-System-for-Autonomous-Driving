"""
config.py — YAML configuration loader with dot-notation access.
"""

import yaml
from pathlib import Path


class Config:
    """
    Load YAML config files and access values with dot notation.

    Usage:
        cfg = Config("configs/model_config.yaml")
        print(cfg.model.weights)
        print(cfg.detection.confidence)
    """

    def __init__(self, path: str | Path | None = None, data: dict | None = None):
        if path is not None:
            path = Path(path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        if data is None:
            data = {}

        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, Config(data=value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default=None):
        """Get a value by key with a default fallback."""
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        """Convert back to a plain dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    def __repr__(self):
        return f"Config({self.to_dict()})"

    def __contains__(self, key):
        return hasattr(self, key)


def load_config(config_dir: str | Path = "configs") -> Config:
    """
    Load all config files from a directory and merge them.

    Returns a Config object with sections:
        - model (from model_config.yaml)
        - camera (from camera_params.yaml)
        - decision (from decision_thresholds.yaml)
    """
    config_dir = Path(config_dir)
    merged = {}

    config_files = {
        "model_config.yaml": None,
        "camera_params.yaml": None,
        "decision_thresholds.yaml": None,
    }

    for filename in config_files:
        filepath = config_dir / filename
        if filepath.exists():
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                merged.update(data)

    return Config(data=merged)
