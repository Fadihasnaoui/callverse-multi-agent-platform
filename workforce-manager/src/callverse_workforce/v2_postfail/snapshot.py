"""Observable V2 snapshot of a live V2 twin at a POST_AI_FAILURE decision (spec §7).

WHITELIST ONLY: intake predictions (``rt.observed``), the contact's observable lifecycle state, elapsed times, its
escalated flag, WAIT count, accumulated human-queue time, queue class and routing-attempt count, the observable
centre state, truth-stripped observable events (Phase F perception filter) and public pre-day context.
Never read: ``rt.truth`` (patience, AI correctness, true urgency, service draws, true deadline), scenario label,
hidden windows, future arrivals, ``config.events``. Enforced by an AST test and the leakage gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from callverse_workforce.agent.perception import to_observable
from callverse_workforce.agent.tools import CENTER_FIELDS
from callverse_workforce.agent.types import DecisionType, PublicDayContext
from callverse_workforce.domain.enums import ContactState

OBSERVED_FIELDS = ("request_type", "urgency_pred", "sensitivity_pred", "channel", "language_dialect", "request_value",
                   "predicted_deadline_at", "callback_allowed", "human_requested", "ai_resolution_score", "observed_at")
BUSY = {ContactState.HUMAN_SERVICE: "HUMAN_SERVICE", ContactState.CALLBACK_ATTEMPT: "CALLBACK_ATTEMPT",
        ContactState.CALLBACK_SERVICE: "CALLBACK_SERVICE"}
BUSY_STATES = frozenset(BUSY.values())  # advisor occupied by a visible contact
DIRECT = {ContactState.HUMAN_QUEUE: "HUMAN_QUEUE", ContactState.AI_QUEUE: "AI_QUEUE", ContactState.AI_SERVICE: "AI_SERVICE",
          ContactState.WAITING: "WAITING", ContactState.CALLBACK_QUEUE: "CALLBACK_QUEUE"}


@dataclass(frozen=True)
class V2InSystem:
    contact_id: str
    state: str
    observed: tuple[tuple[str, Any], ...]
    escalated: bool
    wait_count: int
    human_queue_minutes: float
    queue_class: str | None
    routing_attempt_count: int
    online_wait: float
    elapsed_in_state: float
    entered_at: float = 0.0  # live time the current state was entered (queue position / timers)
    hq_prev: float = 0.0  # human-queue minutes of EARLIER spells
    handoff_applied: bool = False
    decision_type: str = "INITIAL"
    callback_started_at: float | None = None
    callback_attempts: int = 0
    callback_attempt_times: tuple[float, ...] = ()


@dataclass(frozen=True)
class V2Snapshot:
    now: float
    target_contact_id: str
    public: PublicDayContext
    center: tuple[tuple[str, Any], ...]
    in_system: tuple[V2InSystem, ...]
    ai_failure_rate_30: float
    n_ai_30: int
    request_mix_60: tuple[tuple[str, int], ...]
    history_60: tuple[tuple[float, str, float], ...]
    arrival_times: tuple[float, ...]

    def center_value(self, key: str) -> Any:
        return dict(self.center)[key]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["public"] = self.public.to_dict()
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()


def _v(x: Any) -> Any:
    return x.value if hasattr(x, "value") else x


def take_snapshot(twin, target_contact_id: str, public: PublicDayContext) -> V2Snapshot:
    now = twin.now
    pending = {rt.contact_id for rt, _ in twin.pending}
    rows = []
    for cid in sorted(twin._active):
        rt = twin._active[cid]
        st = twin.v2.get(cid)
        if rt.state is ContactState.PENDING_DECISION or cid in pending:
            state = "PENDING_POST" if st is not None and st.decision_type is DecisionType.POST_AI_FAILURE else "PENDING_INITIAL"
        elif rt.state in BUSY:
            state = BUSY[rt.state]
        elif rt.state in DIRECT:
            state = DIRECT[rt.state]
        else:  # AWAITING_FALSE_DISCOVERY looks resolved to the centre -> not in the observable system
            continue
        elapsed = now - rt.state_entered_at
        hq = rt.human_queue_minutes + (elapsed if rt.state is ContactState.HUMAN_QUEUE else 0.0)
        online = rt.online_wait_accumulated + (elapsed if state in ("HUMAN_QUEUE", "AI_QUEUE", "WAITING") else 0.0)
        rows.append(V2InSystem(cid, state, tuple((f, _v(getattr(rt.observed, f))) for f in OBSERVED_FIELDS), rt.escalated,
                               rt.wait_count, hq, st.queue_class if st else None, rt.routing_attempt_count, online, elapsed,
                               entered_at=rt.state_entered_at, hq_prev=rt.human_queue_minutes,
                               handoff_applied=bool(st and st.handoff_applied),
                               decision_type=(st.decision_type.value if st else "INITIAL"),
                               callback_started_at=rt.callback_started_at, callback_attempts=rt.callback_attempts,
                               callback_attempt_times=tuple(rt.callback_attempt_times)))
    events = [e for e in (to_observable(r) for r in twin.events.records) if e is not None]
    done = sum(1 for e in events if e.timestamp >= now - 30 and e.event_type in ("AI_APPARENT_RESOLUTION", "AI_VISIBLE_FAILURE"))
    fail = sum(1 for e in events if e.timestamp >= now - 30 and e.event_type in ("AI_VISIBLE_FAILURE", "FALSE_AI_DISCOVERED"))
    mix: dict[str, int] = {}
    for cid, rt in twin.contacts.items():
        if rt.observed.observed_at >= now - 60:
            k = _v(rt.observed.request_type)
            mix[k] = mix.get(k, 0) + 1
    obs_center = twin.center_state()
    center = []
    for f in CENTER_FIELDS:
        if f in ("hour_sin", "hour_cos"):
            continue
        center.append((f, float(getattr(obs_center, f))))
    return V2Snapshot(now, target_contact_id, public, tuple(center), tuple(rows), min(1.0, fail / done) if done else 0.0, done,
                      tuple(sorted(mix.items())),
                      tuple((e.timestamp, e.event_type, float(e.data.get("online_wait", 0.0))) for e in events if e.timestamp >= now - 60),
                      tuple(e.timestamp for e in events if e.event_type == "ARRIVAL"))
