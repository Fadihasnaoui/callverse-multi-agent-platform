"""Supervision configuration (selected on Phase F DEV, frozen before TEST). Deterministic parameters, not RL."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from typing import Any

MANAGER_PROFILES = ("primary", "slow", "conservative")
CONFIDENCE_RULES = ("t80", "share60")
WORLD_MODELS = ("persistence", "mean_reversion")


@dataclass(frozen=True)
class SupervisionConfig:
    # emergency detection (None = trigger disabled)
    t_ai: float | None = 0.45
    t_ab: float | None = None
    t_q: float | None = None
    debounce: int = 1
    cooldown_minutes: float = 30.0
    n_min_ai: int = 10
    n_min_arr: int = 10
    q_min: int = 2
    kpi_window_minutes: float = 30.0
    cadence_minutes: float = 15.0
    # what-if / recommendation
    g_min: float = 5.0
    d_minutes: float = 30.0
    k_futures: int = 16
    confidence: str = "t80"
    horizon_minutes: float = 120.0
    world_model: str = "persistence"
    mean_reversion_minutes: float = 60.0
    f_ref: float = 0.15  # reference observable AI failure rate (DEV NORMAL, PPO_ALONE); set by the DEV protocol
    m_lo: float = 0.3
    # manager
    manager: str = "primary"
    approval_delay_minutes: float = 5.0
    # master switch: False => PPO_ALONE (supervision never runs)
    enabled: bool = True

    def __post_init__(self) -> None:
        """Strict validation, before any simulation starts: an invalid value raises ValueError naming the field (a zero
        cadence, for instance, would make the supervision scheduler loop forever)."""
        if self.manager not in MANAGER_PROFILES or self.confidence not in CONFIDENCE_RULES or self.world_model not in WORLD_MODELS:
            raise ValueError("invalid supervision configuration: unknown manager profile, confidence rule or world model")
        errors = []

        def number(name: str, ok, rule: str, optional: bool = False) -> None:
            v = getattr(self, name)
            if v is None and optional:
                return
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not ok(v):
                errors.append(f"{name}={v!r} ({rule})")

        for name in ("cadence_minutes", "kpi_window_minutes", "horizon_minutes", "d_minutes", "mean_reversion_minutes"):
            number(name, lambda v: v > 0, "durée strictement positive")
        for name in ("cooldown_minutes", "approval_delay_minutes"):
            number(name, lambda v: v >= 0, "durée positive ou nulle")
        for name in ("t_ai", "t_ab"):
            number(name, lambda v: 0 <= v <= 1, "taux entre 0 et 1", optional=True)
        number("t_q", lambda v: v >= 0, "croissance de file positive ou nulle", optional=True)
        number("f_ref", lambda v: 0 <= v < 1, "taux de référence dans [0, 1[")
        number("m_lo", lambda v: 0 < v <= 1, "multiplicateur dans ]0, 1]")
        number("g_min", lambda v: True, "valeur finie")
        for name, low in (("debounce", 1), ("k_futures", 2), ("n_min_ai", 0), ("n_min_arr", 0), ("q_min", 0)):
            number(name, lambda v, low=low: isinstance(v, int) and v >= low, f"entier >= {low}")
        if not errors:  # mutual compatibility of the durations
            if self.kpi_window_minutes < self.cadence_minutes:
                errors.append("kpi_window_minutes < cadence_minutes (des événements ne seraient vus par aucun contrôle)")
            if self.approval_delay_minutes + self.d_minutes > self.horizon_minutes:
                errors.append("approval_delay_minutes + d_minutes > horizon_minutes (le what-if ne couvrirait pas la prudence)")
        if errors:
            raise ValueError("invalid supervision configuration: " + "; ".join(errors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def config_id(self) -> str:
        """Readable, lexicographically sortable identifier of the tunable part."""
        f = lambda v: "off" if v is None else f"{v:g}"
        return (f"ai{f(self.t_ai)}_ab{f(self.t_ab)}_q{f(self.t_q)}_db{self.debounce}_D{self.d_minutes:g}"
                f"_G{self.g_min:g}_K{self.k_futures}_{self.confidence}")

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def with_(self, **kw: Any) -> "SupervisionConfig":
        return replace(self, **kw)


PPO_ALONE = SupervisionConfig(enabled=False)
