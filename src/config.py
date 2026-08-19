"""
src/config.py
=============
Centralised configuration loader.

Reads config.yaml from the project root and exposes a single
``Config`` object.  All source files import from here — no module
should read YAML directly or contain hard-coded settings.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Locate project root (directory that contains config.yaml)
# ------------------------------------------------------------------ #
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent  # src/ -> project root


def _find_config_file() -> Path:
    """Walk upward from this file to find config.yaml."""
    candidate = PROJECT_ROOT / "config.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"config.yaml not found. Expected at: {candidate}\n"
        "Please run commands from the project root directory."
    )


# ------------------------------------------------------------------ #
# Config class
# ------------------------------------------------------------------ #
class Config:
    """
    Thin wrapper around the YAML config.

    Attributes are accessed via dotted notation or as a plain dict.
    Example:
        cfg = Config()
        cfg.dataset.id          # "ds007169"
        cfg.preprocessing.low_frequency  # 1.0
    """

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = _find_config_file()

        self._path = Path(config_path)
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        with open(self._path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        logger.debug("Config loaded from %s", self._path)

    # ---------------------------------------------------------------- #
    # Attribute access — returns nested Config or plain value
    # ---------------------------------------------------------------- #
    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._data:
            val = self._data[name]
            if isinstance(val, dict):
                return _DictView(val)
            return val
        raise AttributeError(
            f"Config has no attribute '{name}'. "
            f"Check config.yaml for available keys."
        )

    def get(self, key: str, default: Any = None) -> Any:
        """Safe getter with default."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def reload(self) -> None:
        """Reload from disk (useful during development)."""
        self._load()

    def __repr__(self) -> str:
        return f"Config(path={self._path})"


class _DictView:
    """Wraps a nested dict so it supports attribute access."""

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if name in data:
            val = data[name]
            if isinstance(val, dict):
                return _DictView(val)
            return val
        raise AttributeError(f"Config section has no key '{name}'")

    def __contains__(self, key: str) -> bool:
        return key in object.__getattribute__(self, "_data")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def as_dict(self) -> Dict[str, Any]:
        return dict(object.__getattribute__(self, "_data"))

    def __repr__(self) -> str:
        return f"_DictView({object.__getattribute__(self, '_data')})"


# ------------------------------------------------------------------ #
# Convenience helpers
# ------------------------------------------------------------------ #

def get_config(config_path: Optional[Path] = None) -> Config:
    """Return a Config instance. Call from any module."""
    return Config(config_path)


def setup_logging(config: Optional[Config] = None) -> None:
    """
    Configure root logger using settings from config.yaml.
    Call once at application startup (main.py, streamlit_app.py).
    """
    if config is None:
        config = get_config()

    log_cfg = config.logging
    level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
    fmt = log_cfg.get(
        "format",
        "[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    date_fmt = log_cfg.get("date_format", "%Y-%m-%d %H:%M:%S")

    handlers = [logging.StreamHandler()]

    log_file = log_cfg.get("log_file")
    if log_file:
        log_path = PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=date_fmt,
        handlers=handlers,
        force=True,
    )
    logger.info("Logging initialised (level=%s)", logging.getLevelName(level))


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


def resolve_path(relative_path: str, config: Optional[Config] = None) -> Path:
    """
    Resolve a relative path from config.yaml against the project root.

    Example:
        resolve_path("data/features/features.csv")
        # → /abs/path/to/project/data/features/features.csv
    """
    return PROJECT_ROOT / relative_path
