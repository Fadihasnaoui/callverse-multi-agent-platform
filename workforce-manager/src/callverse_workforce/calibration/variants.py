"""Twin model variants for paired before/after comparisons.

B1 and B1.5 differ ONLY in ``patience.model`` and ``callback.model``; every
other parameter (including both models' own parameter blocks) is shared.
"""

from __future__ import annotations

from callverse_workforce.config import SimulationConfig, apply_overrides

B1_OVERRIDES = {"patience.model": "exponential", "callback.model": "b1"}
B1_5_OVERRIDES = {"patience.model": "weibull", "callback.model": "b1_5"}


def b1_config(config: SimulationConfig) -> SimulationConfig:
    return apply_overrides(config, B1_OVERRIDES)


def b1_5_config(config: SimulationConfig) -> SimulationConfig:
    return apply_overrides(config, B1_5_OVERRIDES)
