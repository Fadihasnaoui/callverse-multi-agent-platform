"""Canonical routing observation.

``build_routing_observation`` is the single function that produces the state
seen by every routing policy (development policy now; baselines and PPO later).
It takes only observable inputs — it has no access to ``GroundTruthContact``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any

from callverse_workforce.domain.contact import ObservedContactFeatures
from callverse_workforce.domain.enums import (
    Channel,
    DayType,
    LanguageDialect,
    RequestType,
    Sensitivity,
    Urgency,
)


@dataclass(frozen=True)
class CenterState:
    """Observable snapshot of the contact center at a decision instant."""

    humans_available: int
    human_utilization: float
    human_queue_size: int
    oldest_human_wait: float
    ai_load: float
    recent_ai_failure_rate: float
    wait_queue_size: int
    callbacks_pending: int
    oldest_callback_age: float
    recent_arrival_rate: float  # contacts per hour over the recent window
    forecast_30min: float  # expected arrivals over the next 30 minutes
    recent_abandonment_rate: float
    recent_service_level: float
    day_type: DayType
    clock_hour: float  # wall-clock hour of day, e.g. 9.5 for 09:30


@dataclass(frozen=True)
class RoutingObservation:
    """The 28 semantic features consumed by every routing policy."""

    # Contact / request
    request_type: RequestType
    urgency_pred: Urgency
    sensitivity_pred: Sensitivity
    channel: Channel
    language_dialect: LanguageDialect
    request_value: float
    waiting_time_so_far: float
    deadline_remaining_pred: float
    callback_allowed: bool
    human_requested: bool
    ai_resolution_score: float
    routing_attempt_count: int
    # Human resources
    humans_available: int
    human_utilization: float
    human_queue_size: int
    oldest_human_wait: float
    # AI
    ai_load: float
    recent_ai_failure_rate: float
    # Waiting / callback
    wait_queue_size: int
    callbacks_pending: int
    oldest_callback_age: float
    # Center / context
    recent_arrival_rate: float
    forecast_30min: float
    recent_abandonment_rate: float
    recent_service_level: float
    day_type: DayType
    hour_sin: float
    hour_cos: float

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.value if hasattr(v, "value") else v) for k, v in asdict(self).items()}


OBSERVATION_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(RoutingObservation))


def build_routing_observation(
    features: ObservedContactFeatures,
    waiting_time_so_far: float,
    routing_attempt_count: int,
    center: CenterState,
    now: float,
) -> RoutingObservation:
    """Assemble the canonical observation from observable inputs only."""
    angle = 2.0 * math.pi * (center.clock_hour % 24.0) / 24.0
    return RoutingObservation(
        request_type=features.request_type,
        urgency_pred=features.urgency_pred,
        sensitivity_pred=features.sensitivity_pred,
        channel=features.channel,
        language_dialect=features.language_dialect,
        request_value=features.request_value,
        waiting_time_so_far=waiting_time_so_far,
        deadline_remaining_pred=features.predicted_deadline_at - now,
        callback_allowed=features.callback_allowed,
        human_requested=features.human_requested,
        ai_resolution_score=features.ai_resolution_score,
        routing_attempt_count=routing_attempt_count,
        humans_available=center.humans_available,
        human_utilization=center.human_utilization,
        human_queue_size=center.human_queue_size,
        oldest_human_wait=center.oldest_human_wait,
        ai_load=center.ai_load,
        recent_ai_failure_rate=center.recent_ai_failure_rate,
        wait_queue_size=center.wait_queue_size,
        callbacks_pending=center.callbacks_pending,
        oldest_callback_age=center.oldest_callback_age,
        recent_arrival_rate=center.recent_arrival_rate,
        forecast_30min=center.forecast_30min,
        recent_abandonment_rate=center.recent_abandonment_rate,
        recent_service_level=center.recent_service_level,
        day_type=center.day_type,
        hour_sin=math.sin(angle),
        hour_cos=math.cos(angle),
    )
