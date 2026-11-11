"""Shared test utilities: deterministic configs, scripted contacts and scripted routing."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Sequence

from callverse_workforce.config import SimulationConfig, apply_overrides, load_config
from callverse_workforce.domain.contact import GroundTruthContact
from callverse_workforce.domain.enums import (
    Channel,
    DayType,
    LanguageDialect,
    RequestType,
    RoutingAction,
    Sensitivity,
    Urgency,
)
from callverse_workforce.domain.observation import RoutingObservation
from callverse_workforce.domain.records import DecisionPoint
from callverse_workforce.tracing.summary import summarize
from callverse_workforce.twin.environment import RoutingEnvironment
from callverse_workforce.twin.result import SimulationResult

NOISELESS = {
    "intake_noise.request_type_error_prob": 0.0,
    "intake_noise.urgency_error_prob": 0.0,
    "intake_noise.sensitivity_error_prob": 0.0,
    "intake_noise.deadline_log_sigma": 0.0,
    "intake_noise.ai_score_noise_std": 0.0,
}


def make_config(**overrides: Any) -> SimulationConfig:
    """Default config with noiseless intake; keyword keys use '__' for '.'."""
    dotted = {k.replace("__", "."): v for k, v in overrides.items()}
    return apply_overrides(load_config(), {**NOISELESS, **dotted})


def make_contact(index: int, arrival_time: float, **overrides: Any) -> GroundTruthContact:
    """A fully specified ground-truth contact with benign defaults."""
    fields = dict(
        contact_id=f"C{index:05d}",
        index=index,
        arrival_time=arrival_time,
        request_type=RequestType.FAQ,
        urgency=Urgency.LOW,
        sensitivity=Sensitivity.LOW,
        channel=Channel.PHONE,
        language_dialect=LanguageDialect.FR,
        request_value=0.5,
        deadline_offset_minutes=10_000.0,
        patience_minutes=1_000.0,
        human_service_minutes=10.0,
        ai_service_minutes=2.0,
        ai_correct_prob=1.0,
        ai_false_resolution_prob=0.0,
        false_discovery_delay_minutes=30.0,
        callback_allowed=True,
        human_requested=False,
    )
    fields.update(overrides)
    return GroundTruthContact(**fields)


Chooser = Callable[[DecisionPoint], RoutingAction]


def fixed(action: RoutingAction) -> Chooser:
    return lambda dp: action


def by_contact(plan: dict[str, Sequence[RoutingAction]], default: RoutingAction = RoutingAction.HUMAN) -> Chooser:
    """Per-contact scripted actions, consumed in order; ``default`` once exhausted."""
    remaining = {cid: list(actions) for cid, actions in plan.items()}

    def choose(dp: DecisionPoint) -> RoutingAction:
        queue = remaining.get(dp.contact_id)
        return queue.pop(0) if queue else default

    return choose


def run_scripted(
    config: SimulationConfig,
    contacts: Sequence[GroundTruthContact] | None,
    chooser: Chooser,
    seed: int | None = None,
) -> tuple[RoutingEnvironment, SimulationResult, list[DecisionPoint]]:
    env = RoutingEnvironment(config, contacts=contacts)
    seen: list[DecisionPoint] = []
    dp = env.reset(seed)
    while dp is not None:
        seen.append(dp)
        dp = env.step(chooser(dp)).next_decision
    result = env.result()
    result.metrics = summarize(result)
    return env, result, seen


def events_of(result: SimulationResult, event_type, contact_id: str | None = None) -> list:
    return [
        e for e in result.events if e.event_type is event_type and (contact_id is None or e.contact_id == contact_id)
    ]


def contact(result: SimulationResult, contact_id: str):
    return next(c for c in result.contacts if c.contact_id == contact_id)


def make_observation(**overrides: Any) -> RoutingObservation:
    base = RoutingObservation(
        request_type=RequestType.BOOKING,
        urgency_pred=Urgency.MEDIUM,
        sensitivity_pred=Sensitivity.LOW,
        channel=Channel.PHONE,
        language_dialect=LanguageDialect.FR,
        request_value=0.5,
        waiting_time_so_far=0.0,
        deadline_remaining_pred=600.0,
        callback_allowed=True,
        human_requested=False,
        ai_resolution_score=0.7,
        routing_attempt_count=0,
        humans_available=2,
        human_utilization=0.75,
        human_queue_size=0,
        oldest_human_wait=0.0,
        ai_load=0.2,
        recent_ai_failure_rate=0.1,
        wait_queue_size=0,
        callbacks_pending=0,
        oldest_callback_age=0.0,
        recent_arrival_rate=60.0,
        forecast_30min=30.0,
        recent_abandonment_rate=0.0,
        recent_service_level=1.0,
        day_type=DayType.WEEKDAY,
        hour_sin=0.0,
        hour_cos=1.0,
    )
    return dataclasses.replace(base, **overrides)
