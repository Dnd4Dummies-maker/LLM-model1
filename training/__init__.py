import json
from pathlib import Path

from .config import ModelConfig


def load_config(path: str | Path) -> ModelConfig:
    with open(path) as f:
        return ModelConfig(**json.load(f))
