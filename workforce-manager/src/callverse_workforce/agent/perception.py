"""PerceptionAdapter: the agent's only window on the world (observable, truth-stripped).

The simulation adapter reads the live RoutingEnvironment and forwards (1) routing percepts and (2) whitelisted
outcome events with the data the contact centre can actually see. In particular:
* AI_RESOLVED and AI_FALSE_RESOLUTION both appear as AI_APPARENT_RESOLUTION (a false resolution looks resolved);
* CONTACT_RESOLVED of AI contacts is dropped (it would reveal which AI resolutions were correct);
* ARRIVAL truth payloads, the hidden ``degraded`` flag, callback ``dropout_at``, recontact scheduling, rewards and
  every scenario window are never forwarded.
"""

from __future__ import annotations

from typing import Protocol

from callverse_workforce.agent.types import DecisionType, ObservableEvent, Percept, PublicDayContext
from callverse_workforce.domain.enums import EventType, Outcome
from callverse_workforce.domain.records import DecisionPoint

PASS_THROUGH = {
    EventType.ARRIVAL, EventType.AI_ATTEMPT_STARTED, EventType.AI_VISIBLE_FAILURE, EventType.AI_ESCALATION,
    EventType.FALSE_AI_DISCOVERED, EventType.HUMAN_QUEUE_ENTERED, EventType.HUMAN_SERVICE_STARTED,
    EventType.HUMAN_SERVICE_COMPLETED, EventType.WAIT_STARTED, EventType.WAIT_REEVALUATION, EventType.CALLBACK_SCHEDULED,
    EventType.CALLBACK_ATTEMPT_STARTED, EventType.CALLBACK_ATTEMPT_FAILED, EventType.CALLBACK_REACHED,
    EventType.CALLBACK_FAILED, EventType.CALLBACK_WINDOW_EXPIRED, EventType.CALLBACK_DROPOUT, EventType.ABANDONMENT,
    EventType.DEADLINE_MISSED,
}
APPARENT_AI = {EventType.AI_RESOLVED, EventType.AI_FALSE_RESOLUTION}
SAFE_DATA_KEYS = frozenset({"online_wait", "attempt", "escalated", "wait_number"})
AI_APPARENT_RESOLUTION = "AI_APPARENT_RESOLUTION"
VISIBLE_RESOLUTION = {Outcome.RESOLVED_HUMAN: "HUMAN_RESOLVED", Outcome.RESOLVED_CALLBACK: "CALLBACK_RESOLVED"}


def to_observable(record) -> ObservableEvent | None:
    et = record.event_type
    data = {k: v for k, v in record.data.items() if k in SAFE_DATA_KEYS}
    if et in APPARENT_AI:
        return ObservableEvent(record.time, AI_APPARENT_RESOLUTION, record.contact_id, {})
    if et is EventType.CONTACT_RESOLVED:
        name = VISIBLE_RESOLUTION.get(record.data.get("outcome"))
        return ObservableEvent(record.time, name, record.contact_id, {}) if name else None
    if et in PASS_THROUGH:
        if et is EventType.ARRIVAL:
            data = {}
        return ObservableEvent(record.time, et.value, record.contact_id, data)
    return None


class PerceptionAdapter(Protocol):
    def percept(self, dp: DecisionPoint) -> Percept: ...

    def new_events(self) -> list[ObservableEvent]: ...

    def public_context(self) -> PublicDayContext: ...


class SimulationPerception:
    """Digital-Twin substrate: wraps the live RoutingEnvironment (the agent never sees the twin)."""

    def __init__(self, env, public: PublicDayContext) -> None:
        self._env = env
        self._public = public
        self._cursor = 0

    def percept(self, dp: DecisionPoint) -> Percept:
        return Percept(dp.decision_id, dp.contact_id, dp.timestamp, DecisionType.INITIAL, dp.observation, dp.action_mask)

    def new_events(self) -> list[ObservableEvent]:
        records = self._env.twin.events.records
        out = []
        for r in records[self._cursor:]:
            ev = to_observable(r)
            if ev is not None:
                out.append(ev)
        self._cursor = len(records)
        return out

    def public_context(self) -> PublicDayContext:
        return self._public


def public_context_from(config, base_config) -> PublicDayContext:
    """Public, pre-day information: calendar reference profile, visible roster, historical (base) request mix."""
    return PublicDayContext(
        day_type=config.scenario.day_type.value if hasattr(config.scenario.day_type, "value") else str(config.scenario.day_type),
        start_hour=float(config.scenario.start_hour),
        reference_hourly_rates=tuple(float(r) for r in config.forecasting.reference_hourly_arrival_rates),
        rostered_advisors=int(config.humans.n_advisors),
        request_type_prior=tuple(sorted((k.value if hasattr(k, "value") else str(k), float(v.share))
                                        for k, v in base_config.request_types.items())),
    )
