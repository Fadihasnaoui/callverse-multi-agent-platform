"""MemoryStore: bounded working memory (context / KPIs) + append-only audit log.

Memory is fed only by percepts, the agent's own decisions and observable events. It is never an input of the
frozen PPO observation (the StateBuilder reads the Percept only).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from callverse_workforce.agent.types import ObservableEvent, Percept

MAX_AGE_MINUTES = 120.0
SIZES = {"decisions": 5000, "events": 20000, "queue_samples": 5000, "kpi_snapshots": 500}
ONLINE_WAIT_STATES = ("AI_QUEUE", "HUMAN_QUEUE", "WAITING")


@dataclass
class OpenContact:
    """A contact the centre currently handles, described by OBSERVED attributes only."""

    contact_id: str
    observed: dict[str, Any]  # intake prediction at decision time (request_type, urgency_pred, ...)
    decided_at: float
    waited_at_decision: float
    state: str
    since: float
    escalated: bool = False
    waited_accumulated: float = 0.0

    def waited(self, now: float) -> float:
        extra = now - self.since if self.state in ONLINE_WAIT_STATES else 0.0
        return self.waited_at_decision + self.waited_accumulated + extra


ACTION_STATE = {"AI": "AI_QUEUE", "HUMAN": "HUMAN_QUEUE", "WAIT": "WAITING", "CALLBACK": "CALLBACK_QUEUE"}
EVENT_STATE = {
    "AI_ATTEMPT_STARTED": "AI_SERVICE", "HUMAN_QUEUE_ENTERED": "HUMAN_QUEUE", "HUMAN_SERVICE_STARTED": "HUMAN_SERVICE",
    "WAIT_STARTED": "WAITING", "CALLBACK_SCHEDULED": "CALLBACK_QUEUE", "CALLBACK_ATTEMPT_STARTED": "CALLBACK_ATTEMPT",
    "CALLBACK_ATTEMPT_FAILED": "CALLBACK_QUEUE", "CALLBACK_REACHED": "CALLBACK_SERVICE",
}
CLOSING = {"AI_APPARENT_RESOLUTION", "HUMAN_RESOLVED", "CALLBACK_RESOLVED", "HUMAN_SERVICE_COMPLETED", "ABANDONMENT",
           "DEADLINE_MISSED", "CALLBACK_FAILED", "CALLBACK_WINDOW_EXPIRED", "CALLBACK_DROPOUT"}
OBSERVED_FIELDS = ("request_type", "urgency_pred", "sensitivity_pred", "channel", "language_dialect", "request_value",
                   "deadline_remaining_pred", "callback_allowed", "human_requested", "ai_resolution_score")


def _value(v: Any) -> Any:
    return v.value if hasattr(v, "value") else v


class MemoryStore:
    def __init__(self, max_age: float = MAX_AGE_MINUTES) -> None:
        self.max_age = max_age
        self.decisions: deque[dict[str, Any]] = deque(maxlen=SIZES["decisions"])
        self.events: deque[ObservableEvent] = deque(maxlen=SIZES["events"])
        self.queue_samples: deque[tuple[float, int]] = deque(maxlen=SIZES["queue_samples"])
        self.kpi_snapshots: deque[dict[str, Any]] = deque(maxlen=SIZES["kpi_snapshots"])
        self.arrival_times: list[float] = []  # all observed arrival times of the day (forecast correction input)
        self.open: dict[str, OpenContact] = {}
        self.audit: list[dict[str, Any]] = []
        self.last_percept: Percept | None = None
        self.now = 0.0

    # --- writes ---------------------------------------------------------------------------------------
    def record(self, kind: str, t: float, **fields: Any) -> dict[str, Any]:
        rec = {"kind": kind, "t": t, "seq": len(self.audit)} | fields
        self.audit.append(rec)
        return rec

    def add_events(self, events: list[ObservableEvent]) -> None:
        for ev in events:
            self.now = max(self.now, ev.timestamp)
            self.events.append(ev)
            if ev.event_type == "ARRIVAL":
                self.arrival_times.append(ev.timestamp)
            c = self.open.get(ev.contact_id) if ev.contact_id else None
            if c is None:
                continue
            if ev.event_type in CLOSING:
                self.open.pop(ev.contact_id, None)
                continue
            if ev.event_type == "AI_ESCALATION":
                c.escalated = True
            new = EVENT_STATE.get(ev.event_type)
            if new is not None and new != c.state:
                if c.state in ONLINE_WAIT_STATES:
                    c.waited_accumulated += ev.timestamp - c.since
                c.state, c.since = new, ev.timestamp
        self._evict()

    def observe_percept(self, percept: Percept) -> None:
        """Current observation of the centre, recorded at the START of the routing event (before supervision), so the
        supervision tools read the state of this instant, never the previous decision's (timestamped by the percept)."""
        if self.last_percept is not None and self.last_percept.decision_id == percept.decision_id:
            return  # same decision point observed twice: no duplicate queue sample
        self.last_percept = percept
        self.now = max(self.now, percept.timestamp)
        self.queue_samples.append((percept.timestamp, percept.observation.human_queue_size))

    def add_decision(self, percept: Percept, action_id: str, mode: str, policy_used: str) -> None:
        obs = percept.observation
        self.observe_percept(percept)  # no-op when route() already recorded this observation
        self.decisions.append({"t": percept.timestamp, "contact_id": percept.contact_id, "action": action_id,
                               "mode": mode, "policy": policy_used, "request_type": _value(obs.request_type)})
        observed = {f: _value(getattr(obs, f)) for f in OBSERVED_FIELDS}
        prev = self.open.get(percept.contact_id)
        self.open[percept.contact_id] = OpenContact(
            percept.contact_id, observed, percept.timestamp, obs.waiting_time_so_far, ACTION_STATE[action_id], percept.timestamp,
            escalated=prev.escalated if prev else False)

    def _evict(self) -> None:
        horizon = self.now - self.max_age
        for dq in (self.decisions,):
            while dq and dq[0]["t"] < horizon:
                dq.popleft()
        while self.events and self.events[0].timestamp < horizon:
            self.events.popleft()
        while self.queue_samples and self.queue_samples[0][0] < horizon:
            self.queue_samples.popleft()

    # --- reads ----------------------------------------------------------------------------------------
    def count(self, event_types: set[str] | tuple[str, ...], since: float) -> int:
        return sum(1 for e in self.events if e.timestamp >= since and e.event_type in event_types)

    def events_since(self, since: float) -> list[ObservableEvent]:
        return [e for e in self.events if e.timestamp >= since]

    def records(self, kind: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        out = [r for r in self.audit if kind is None or r["kind"] == kind]
        return out[-limit:] if limit else out
