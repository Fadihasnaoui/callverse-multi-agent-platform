"""Phase E environment: frozen V1.1 4-cell mixture x 8 scenario families (+ one training-only load family).

Everything here is frozen BEFORE any PPO training (hashes persisted in artifacts/ppo_v1_1/). Scenario
parameters are PREDEFINED_V1_SCENARIO assumptions (synthetic). No scenario label ever enters the
observation; calendar context (summer / Ramadan / Omra-holidays) is reflected only through the
historical forecast reference profile of that known day type (and day_type=HOLIDAY for Omra/holidays).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from callverse_workforce.config import SimulationConfig, apply_overrides
from callverse_workforce.phase_d.grid import MIXTURE_CELLS, Cell, cell_config
from callverse_workforce.phase_d.scenarios import AI_DEGRADATION, DEMAND_SHOCK, WORST_DAY

PURPOSE = "PPO_V1_1_ENVIRONMENT"
NORMAL_PROFILE = [40, 60, 80, 90, 80, 65, 65, 80, 85, 70, 55, 40]

# --- the four new families (scenarios 2, 3, 4, 6) -----------------------------------------------------
SUMMER_PROFILE = [round(r * 1.35, 2) for r in NORMAL_PROFILE]  # +35% high season, same shape
SUMMER_MIX = {"FAQ": 0.35, "BOOKING": 0.35, "CHANGE_CANCEL": 0.20, "COMPLAINT_INCIDENT": 0.10}
RAMADAN_PROFILE = [30, 40, 55, 60, 55, 45, 45, 60, 80, 95, 90, 75]  # quiet mornings, evening peak
OMRA_PROFILE = [55, 70, 85, 90, 85, 75, 75, 85, 85, 75, 60, 50]  # school holidays / Omra: flatter, higher
OMRA_MIX = {"FAQ": 0.28, "BOOKING": 0.38, "CHANGE_CANCEL": 0.24, "COMPLAINT_INCIDENT": 0.10}
HUMAN_SHORTAGE = {"start_min": 180.0, "start_max": 420.0, "duration_minutes": 240.0, "absent_advisors": 3}

FAMILIES: tuple[str, ...] = ("NORMAL", "SUMMER", "RAMADAN", "OMRA_HOLIDAYS", "AI_DEGRADATION",
                             "HUMAN_SHORTAGE", "DEMAND_SHOCK", "WORST_DAY")
TRAINING_ONLY = ("HIGH_LOAD",)  # +30% nested demand (reduced-resource / high-load bucket); never evaluated

# Frozen training distribution: 60% normal/seasonal, 25% high-load/reduced-resource, 15% rare incidents.
TRAINING_DISTRIBUTION: dict[str, float] = {
    "NORMAL": 0.30, "SUMMER": 0.10, "RAMADAN": 0.10, "OMRA_HOLIDAYS": 0.10,
    "HIGH_LOAD": 0.125, "HUMAN_SHORTAGE": 0.125,
    "AI_DEGRADATION": 0.06, "DEMAND_SHOCK": 0.06, "WORST_DAY": 0.03,
}
MIXTURE_PROBABILITIES: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)
assert abs(sum(TRAINING_DISTRIBUTION.values()) - 1.0) < 1e-12 and abs(sum(MIXTURE_PROBABILITIES) - 1.0) < 1e-12
assert len(MIXTURE_CELLS) == len(MIXTURE_PROBABILITIES) == 4

TRAINING_SEED_MIN = 100_000
TRAINING_SEED_MAX = 2**31 - 1


def family_overrides(family: str) -> dict[str, Any]:
    o: dict[str, Any] = {}
    if family == "NORMAL":
        return o
    if family == "SUMMER":
        o |= {"scenario.hourly_arrival_rates": SUMMER_PROFILE, "forecasting.reference_hourly_arrival_rates": SUMMER_PROFILE}
        o |= {f"request_types.{k}.share": v for k, v in SUMMER_MIX.items()}
    elif family == "RAMADAN":
        o |= {"scenario.hourly_arrival_rates": RAMADAN_PROFILE, "forecasting.reference_hourly_arrival_rates": RAMADAN_PROFILE}
    elif family == "OMRA_HOLIDAYS":
        o |= {"scenario.hourly_arrival_rates": OMRA_PROFILE, "forecasting.reference_hourly_arrival_rates": OMRA_PROFILE,
              "scenario.day_type": "HOLIDAY"}
        o |= {f"request_types.{k}.share": v for k, v in OMRA_MIX.items()}
    elif family == "AI_DEGRADATION":
        o |= {"events.ai_degradation": AI_DEGRADATION}
    elif family == "HUMAN_SHORTAGE":
        o |= {"events.human_shortage": HUMAN_SHORTAGE}
    elif family == "DEMAND_SHOCK":
        o |= {"events.demand_shock": DEMAND_SHOCK}
    elif family == "WORST_DAY":
        o |= {"scenario.extra_load_layers": WORST_DAY["extra_load_layers"], "humans.n_advisors": WORST_DAY["n_advisors"],
              "events.patience_scale": WORST_DAY["patience_scale"], "events.ai_degradation": WORST_DAY["ai_degradation"]}
    elif family == "HIGH_LOAD":
        o |= {"scenario.extra_load_layers": [0.3]}
    else:
        raise KeyError(family)
    return o


def day_config(base: SimulationConfig, family: str, cell: Cell) -> SimulationConfig:
    """V1.1 mixture cell + scenario family. The forecast reference changes only for calendar families."""
    if cell not in MIXTURE_CELLS:
        raise ValueError(f"{cell.cell_id} is not one of the frozen Phase D mixture cells")
    cfg = cell_config(base, cell)
    return apply_overrides(cfg, {"scenario.purpose": PURPOSE,
                                 "scenario.name": f"ppo_{family.lower()}_{cell.cell_id}"} | family_overrides(family))


@dataclass(frozen=True)
class DaySpec:
    family: str
    cell: Cell
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "cell": self.cell.cell_id, "seed": self.seed}


class TrainingDaySampler:
    """Procedural training days (seeds >= 100000) drawn from the frozen distribution, reproducible per run."""

    def __init__(self, run_seed: int) -> None:
        self.rng = np.random.default_rng(np.random.SeedSequence(run_seed, spawn_key=(0x7E7E,)))
        self.families = list(TRAINING_DISTRIBUTION)
        self.p = np.array([TRAINING_DISTRIBUTION[f] for f in self.families])

    def __call__(self) -> DaySpec:
        fam = self.families[int(self.rng.choice(len(self.families), p=self.p))]
        cell = MIXTURE_CELLS[int(self.rng.choice(4, p=np.array(MIXTURE_PROBABILITIES)))]
        seed = int(self.rng.integers(TRAINING_SEED_MIN, TRAINING_SEED_MAX))
        return DaySpec(fam, cell, seed)


def fixed_days(seeds: range | tuple[int, ...]) -> list[DaySpec]:
    """8 families x 10 days, families in blocks of 10 consecutive seeds; cells rotate through the mixture."""
    seeds = list(seeds)
    if len(seeds) != 80:
        raise ValueError("expected 80 seeds")
    return [DaySpec(FAMILIES[i // 10], MIXTURE_CELLS[(i % 10) % 4], s) for i, s in enumerate(seeds)]


def config_hash(cfg: SimulationConfig) -> str:
    return hashlib.sha256(json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()
