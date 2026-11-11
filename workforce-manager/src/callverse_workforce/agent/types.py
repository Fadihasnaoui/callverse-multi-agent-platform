"""Shared runtime types of the Workforce Manager agent (observable-only data model)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.observation import RoutingObservation


class DecisionType(str, Enum):
    INITIAL = "INITIAL"
    POST_AI_FAILURE = "POST_AI_FAILURE"  # reserved for V2; never enabled in Phase F


ENABLED_DECISION_TYPES: tuple[DecisionType, ...] = (DecisionType.INITIAL,)


class AgentMode(str, Enum):
    NORMAL_PPO = "NORMAL_PPO"
    EMERGENCY_ANALYSIS = "EMERGENCY_ANALYSIS"
    WAITING_MANAGER_APPROVAL = "WAITING_MANAGER_APPROVAL"
    TEMPORARY_PRUDENCE = "TEMPORARY_PRUDENCE"
    TECHNICAL_FALLBACK = "TECHNICAL_FALLBACK"


PPO_MODES = (AgentMode.NORMAL_PPO, AgentMode.EMERGENCY_ANALYSIS, AgentMode.WAITING_MANAGER_APPROVAL)


@dataclass(frozen=True)
class Percept:
    """Everything the agent may know at a routing event."""

    decision_id: str
    contact_id: str
    timestamp: float
    decision_type: DecisionType
    observation: RoutingObservation
    action_mask: ActionMask


@dataclass(frozen=True)
class ObservableEvent:
    """One whitelisted, truth-stripped outcome event as the contact centre sees it."""

    timestamp: float
    event_type: str
    contact_id: str | None
    data: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PublicDayContext:
    """Information known before / outside the day: calendar profile, roster, historical request mix.

    The reference profile is the historical profile of the known day type (never the hidden truth);
    the roster is the visible staffing plan; the request-type prior is the historical (base) mix.
    """

    day_type: str
    start_hour: float
    reference_hourly_rates: tuple[float, ...]
    rostered_advisors: int
    request_type_prior: tuple[tuple[str, float], ...]

    def to_dict(self) -> dict[str, Any]:
        return {"day_type": self.day_type, "start_hour": self.start_hour,
                "reference_hourly_rates": list(self.reference_hourly_rates),
                "rostered_advisors": self.rostered_advisors, "request_type_prior": dict(self.request_type_prior)}
