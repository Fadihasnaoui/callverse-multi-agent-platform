"""EmergencyDetector: deterministic, observable-only triggers with debounce (thresholds chosen on DEV, frozen)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from callverse_workforce.agent.config import SupervisionConfig

TRIGGER_ORDER = ("AI_FAILURE_RATE", "ABANDONMENT", "QUEUE_GROWTH")


@dataclass(frozen=True)
class Emergency:
    emergency_id: str
    t: float
    trigger: str
    observed: float
    threshold: float
    all_triggers: tuple[tuple[str, float, float], ...]
    debounce_count: int


def evaluate_triggers(kpis: dict[str, Any], cfg: SupervisionConfig) -> list[tuple[str, float, float]]:
    """Pure function of observable KPIs. Never reads a scenario label, a hidden window or the reward."""
    hits = []
    if cfg.t_ai is not None and kpis["n_ai_30"] >= cfg.n_min_ai and kpis["ai_failure_rate_30"] > cfg.t_ai:
        hits.append(("AI_FAILURE_RATE", kpis["ai_failure_rate_30"], cfg.t_ai))
    if cfg.t_ab is not None and kpis["n_arrivals_30"] >= cfg.n_min_arr and kpis["abandonment_rate_30"] > cfg.t_ab:
        hits.append(("ABANDONMENT", kpis["abandonment_rate_30"], cfg.t_ab))
    if cfg.t_q is not None and kpis["human_queue_size"] >= cfg.q_min and kpis["human_queue_growth"] > cfg.t_q:
        hits.append(("QUEUE_GROWTH", kpis["human_queue_growth"], cfg.t_q))
    return hits


class EmergencyDetector:
    """Debounce per trigger (docs/phase_f_spec.md §7: "a trigger must hold on k_debounce consecutive supervision
    cycles"): each indicator has its own streak, reset as soon as its own condition disappears; two different
    indicators never add up. When several indicators reach the debounce on the same cycle, TRIGGER_ORDER decides."""

    def __init__(self, cfg: SupervisionConfig) -> None:
        self.cfg = cfg
        self.streaks = dict.fromkeys(TRIGGER_ORDER, 0)
        self.n = 0

    @property
    def streak(self) -> int:  # longest current streak (read-only view, kept for the existing records / displays)
        return max(self.streaks.values())

    def _reset(self) -> None:
        self.streaks = dict.fromkeys(TRIGGER_ORDER, 0)

    def detect(self, kpis: dict[str, Any], now: float, eligible: bool) -> tuple[Emergency | None, list]:
        hits = evaluate_triggers(kpis, self.cfg)
        if not eligible:  # cooldown / analysis in progress: no sequence is carried over
            self._reset()
            return None, hits
        held = {h[0] for h in hits}
        for name in TRIGGER_ORDER:
            self.streaks[name] = self.streaks[name] + 1 if name in held else 0
        due = [h for h in hits if self.streaks[h[0]] >= self.cfg.debounce]
        if not due:
            return None, hits
        trig = min(due, key=lambda h: TRIGGER_ORDER.index(h[0]))
        count = self.streaks[trig[0]]
        self.n += 1
        self._reset()
        return Emergency(f"E{self.n:03d}", now, trig[0], trig[1], trig[2], tuple(hits), count), hits
