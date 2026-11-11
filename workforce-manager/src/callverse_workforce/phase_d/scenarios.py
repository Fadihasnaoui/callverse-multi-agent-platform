"""Predefined V1 scenarios 5 / 7 / 8 (PREDEFINED_V1_SCENARIO; V1.1 mechanisms OFF).

Parameters are fixed in docs/twin_v1_1_rationale.md before any run. Event start times are drawn per day
from hidden dedicated streams; nothing about the scenario is part of the observation.
"""

from __future__ import annotations

from callverse_workforce.config import SimulationConfig, apply_overrides

PURPOSE = "PREDEFINED_V1_SCENARIO"

AI_DEGRADATION = {"start_min": 120.0, "start_max": 420.0, "duration_minutes": 180.0,
                  "correct_multiplier": 0.5, "service_time_multiplier": 1.5}
DEMAND_SHOCK = {"request_type": "CHANGE_CANCEL", "intensity_increase": 1.5, "start_min": 120.0, "start_max": 480.0,
                "duration_minutes": 120.0, "urgency_probs": {"LOW": 0.1, "MEDIUM": 0.3, "HIGH": 0.6}}
WORST_DAY = {"extra_load_layers": [0.3], "n_advisors": 7, "patience_scale": 0.8, "ai_degradation": AI_DEGRADATION}

SCENARIOS = ("SCENARIO_5", "SCENARIO_7", "SCENARIO_8")


def scenario_config(base: SimulationConfig, name: str) -> SimulationConfig:
    if not base.v1_1.is_v1:
        raise ValueError("predefined scenarios run on the V1 twin (V1.1 mechanisms OFF)")
    common = {"scenario.purpose": PURPOSE, "scenario.name": name.lower()}
    if name == "SCENARIO_5":
        return apply_overrides(base, common | {"events.ai_degradation": AI_DEGRADATION})
    if name == "SCENARIO_7":
        return apply_overrides(base, common | {"events.demand_shock": DEMAND_SHOCK})
    if name == "SCENARIO_8":
        return apply_overrides(base, common | {
            "scenario.extra_load_layers": WORST_DAY["extra_load_layers"],
            "humans.n_advisors": WORST_DAY["n_advisors"],
            "events.patience_scale": WORST_DAY["patience_scale"],
            "events.ai_degradation": WORST_DAY["ai_degradation"],
        })
    raise KeyError(name)


def event_windows(config: SimulationConfig, seed: int) -> dict[str, tuple[float, float] | None]:
    """Hidden event windows of a scenario day (for ANALYSIS only, never for routing)."""
    from callverse_workforce.twin.generator import ContactGenerator, scenario_window
    from callverse_workforce.utils.rng import RngManager, Stream

    rng = RngManager(seed)
    deg = config.events.ai_degradation
    return {
        "ai_degradation": scenario_window(rng, Stream.AI_DEGRADATION_TIMING, deg.start_min, deg.start_max,
                                          deg.duration_minutes) if deg else None,
        "demand_shock": ContactGenerator(config, rng).shock_window(),
    }


def period_of(t: float, window: tuple[float, float] | None) -> str:
    if window is None:
        return "none"
    return "before" if t < window[0] else ("during" if t < window[1] else "after")
