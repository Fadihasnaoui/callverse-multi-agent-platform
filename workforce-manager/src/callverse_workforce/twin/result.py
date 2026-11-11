"""Simulation result container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from callverse_workforce.domain.records import ContactSummary, RoutingDecision, Transition
from callverse_workforce.reward.breakdown import RewardBreakdown
from callverse_workforce.tracing.events import EventRecord


@dataclass
class SimulationResult:
    seed: int
    scenario_name: str
    horizon_minutes: float
    n_advisors: int
    contacts: list[ContactSummary]
    decisions: list[RoutingDecision]
    transitions: list[Transition]
    events: list[EventRecord]
    episode_reward: RewardBreakdown
    pre_decision_reward: RewardBreakdown  # reward accrued before the first decision (or on an empty day)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_reward(self) -> float:
        return self.episode_reward.total
