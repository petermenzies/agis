from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "qwen2.5:3b"


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    return Path(base) / "agis" / "config.toml"


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    host: str | None = None
    from_file: bool = False


def load() -> Config:
    """Settings from the config file, overridden by AGIS_MODEL and OLLAMA_HOST."""
    config = Config()
    path = config_path()
    if path.exists():
        data = tomllib.loads(path.read_text())
        config.model = data.get("model", config.model)
        config.host = data.get("host") or None
        config.from_file = True
    config.model = os.environ.get("AGIS_MODEL", config.model)
    config.host = os.environ.get("OLLAMA_HOST", config.host)
    return config


def save(model: str, host: str | None = None) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f'model = "{model}"']
    if host:
        lines.append(f'host = "{host}"')
    path.write_text("\n".join(lines) + "\n")
    return path
