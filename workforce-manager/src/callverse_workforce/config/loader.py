"""Configuration loading and override helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from callverse_workforce.config.models import SimulationConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "normal_day.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> SimulationConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return SimulationConfig.model_validate(raw)


def apply_overrides(config: SimulationConfig, overrides: Mapping[str, Any]) -> SimulationConfig:
    """Return a re-validated copy of ``config`` with dotted-path overrides applied.

    Example: ``apply_overrides(cfg, {"humans.n_advisors": 1, "ai.available": False})``.
    """
    data = config.model_dump(mode="python")
    for dotted, value in overrides.items():
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[_resolve_key(node, key)]
        node[_resolve_key(node, leaf, allow_new=True)] = value
    return SimulationConfig.model_validate(data)


def _resolve_key(node: dict, key: str, allow_new: bool = False) -> Any:
    """Match a string key against dict keys that may be enums."""
    for existing in node:
        if existing == key or getattr(existing, "value", None) == key:
            return existing
    if allow_new:
        return key
    raise KeyError(key)
