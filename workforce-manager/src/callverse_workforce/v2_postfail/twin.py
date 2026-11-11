"""V2 Digital Twin: subclass of the frozen V1.1 DigitalTwin (twin/digital_twin.py is not modified).

Differences, all specified in docs/v2_headroom_spec.md:
* after a non-hangup VISIBLE AI failure, the contact gets a POST_AI_FAILURE decision (AI masked) instead of the
  hard-coded escalation; WAIT re-evaluations stay POST_AI_FAILURE (spec §2, §3.3);
* V2 actions: INITIAL {AI, HUMAN}; POST {HUMAN_NORMAL, HUMAN_HIGH, WAIT, CALLBACK} (V2Mask, U1 wait_count);
* one shared human pool with HIGH class and NORMAL aging at A_age (spec §4, queue.AgingHumanPool);
* handoff overhead added exactly once, at an actual human connection (online service start or answered callback),
  for escalated contacts (spec §3.5, U2).
Everything else (arrivals, intake, AI service, patience, abandonment, deadlines, callbacks, recontacts, reward,
end-of-day settlement, RNG streams) is inherited unchanged.

Hypothetical worlds only: ``seed_contact`` restores an in-system contact from an observable snapshot at its exact
pending point (spec §7.0a); the live (non-restored) paths are unaffected by that machinery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from callverse_workforce.agent.types import DecisionType
from callverse_workforce.domain.contact import ContactRuntimeState, GroundTruthContact, ObservedContactFeatures
from callverse_workforce.domain.enums import CallbackReason, ContactState, EventType, Outcome, WakeCause
from callverse_workforce.domain.observation import RoutingObservation, build_routing_observation
from callverse_workforce.twin.digital_twin import DigitalTwin, InvalidActionError
from callverse_workforce.utils.rng import Stream
from callverse_workforce.v2_postfail import actions as A
from callverse_workforce.v2_postfail.queue import CALLBACK, HIGH, HOLD, NORMAL, AgingHumanPool

A_AGE_PRIMARY = 8.387374  # artifacts/v2_headroom/u3_aging_threshold.json (spec §4.1)
A_AGE_SENSITIVITY = 3.0


@dataclass
class V2ContactState:
    decision_type: DecisionType = DecisionType.INITIAL
    handoff_applied: bool = False
    queue_class: str | None = None
    post_decisions: int = 0


@dataclass(frozen=True)
class V2Observation:
    """V2 schema V2_INITIAL_POSTFAIL: the 28 frozen semantic features + decision_type, wait_count, escalated."""

    base: RoutingObservation
    decision_type: DecisionType
    wait_count: int
    escalated: bool


@dataclass(frozen=True)
class V2DecisionPoint:
    decision_id: str
    contact_id: str
    timestamp: float
    decision_type: DecisionType
    observation: V2Observation
    mask: A.V2Mask


@dataclass(frozen=True)
class RestoreSpec:
    """Observable lifecycle state of an in-system contact (hypothetical-world restoration, spec §7.0a)."""

    state: str
    entered_at: float  # live time the contact entered its current state (queue position / timers)
    elapsed_in_state: float
    escalated: bool
    wait_count: int
    hq_prev: float  # human-queue minutes accumulated in EARLIER spells
    queue_class: str | None
    routing_attempt_count: int
    online_wait: float  # total online wait so far (including the current spell)
    handoff_applied: bool
    decision_type: DecisionType
    callback_started_at: float | None = None
    callback_attempts: int = 0
    callback_attempt_times: tuple[float, ...] = field(default_factory=tuple)


class V2Twin(DigitalTwin):
    def __init__(self, config, seed: int, contacts=None, forecaster=None, analyzer=None, a_age: float = A_AGE_PRIMARY) -> None:
        if config.wait.max_waits != A.MAX_WAITS or config.wait.reevaluation_minutes != 2.0:
            raise ValueError("frozen WAIT semantics expected (2.0 min, 3 waits)")
        super().__init__(config, seed, contacts, forecaster, analyzer)
        # the base shortage process reads self.humans lazily at run time, so replacing the pool here is safe
        self.humans = AgingHumanPool(self.env, capacity=config.humans.n_advisors, a_age=a_age)
        self.a_age = a_age
        self.v2: dict[str, V2ContactState] = {}
        self.handoff_events = 0
        self.class_waits: dict[str, list[float]] = {NORMAL: [], HIGH: []}  # cumulative human-queue wait at service start
        self.restored: dict[str, RestoreSpec] = {}

    def st(self, rt: ContactRuntimeState) -> V2ContactState:
        return self.v2.setdefault(rt.contact_id, V2ContactState())

    # ------------------------------------------------------------------------------------------- controller API
    def observe_v2(self, rt: ContactRuntimeState) -> tuple[V2Observation, A.V2Mask]:
        center = self.center_state()
        obs = build_routing_observation(rt.observed, waiting_time_so_far=self._waiting_time_so_far(rt),
                                        routing_attempt_count=rt.routing_attempt_count, center=center, now=self.now)
        st = self.st(rt)
        mask = A.v2_mask(st.decision_type, obs, rt.wait_count, self.service_availability(), self.config.mask, self.config.wait)
        return V2Observation(obs, st.decision_type, rt.wait_count, rt.escalated), mask

    def observe(self, rt):  # the V1.1 observation API is not available on the V2 twin
        raise RuntimeError("use observe_v2 on the V2 twin")

    def apply_decision(self, decision_id: str, action: str, mask: A.V2Mask) -> ContactRuntimeState:  # type: ignore[override]
        if not self.pending:
            raise RuntimeError("no routing decision is pending")
        if not mask.is_valid(action):
            raise InvalidActionError(f"action {action} is masked: {mask.reasons}")
        rt, event = self.pending.popleft()
        st = self.st(rt)
        if mask.decision_type is not st.decision_type:
            raise RuntimeError("mask decision type does not match the contact context")
        self.events.log(self.now, EventType.ROUTING_DECISION, rt.contact_id, decision_id, action=action,
                        decision_type=st.decision_type.value, mask=dict(zip(A.V2_ACTIONS, mask.valid)),
                        routing_attempt_count=rt.routing_attempt_count, wait_count=rt.wait_count)
        rt.actions.append(action)
        rt.decision_ids.append(decision_id)
        rt.routing_attempt_count += 1
        if action == A.WAIT:  # U1: incremented ONLY when WAIT executes
            rt.wait_count += 1
        if st.decision_type is DecisionType.POST_AI_FAILURE:
            st.post_decisions += 1
        event.succeed(action)
        return rt

    # ------------------------------------------------------------------------------------------- processes
    def _decide(self, rt: ContactRuntimeState):
        if rt.state is not ContactState.PENDING_DECISION:
            self._set_state(rt, ContactState.PENDING_DECISION)
        decision = self.env.event()
        self.pending.append((rt, decision))
        action = yield decision
        return action

    def _contact_process(self, truth: GroundTruthContact):
        rt = self._register_arrival(truth)
        self.st(rt)
        yield from self._run_initial(rt)

    def _run_initial(self, rt: ContactRuntimeState):
        action = yield from self._decide(rt)
        if action == A.AI:
            yield from self._ai_path(rt)
        elif action == A.HUMAN:
            yield from self._human_path_v2(rt, NORMAL)
        else:  # pragma: no cover - the INITIAL mask forbids it
            raise InvalidActionError(action)

    def _post_failure(self, rt: ContactRuntimeState, resume_wait_minutes: float | None = None):
        """POST_AI_FAILURE context: every (re-)decision stays POST_AI_FAILURE with AI masked."""
        st = self.st(rt)
        st.decision_type = DecisionType.POST_AI_FAILURE
        if resume_wait_minutes is not None:  # restored WAITING contact: remaining re-evaluation time only
            again = yield from self._wait_v2(rt, resume_wait_minutes, resumed=True)
            if not again:
                return
        while True:
            action = yield from self._decide(rt)
            if action == A.WAIT:
                again = yield from self._wait_v2(rt, self.config.wait.reevaluation_minutes)
                if again:
                    continue
                return
            if action == A.HUMAN_NORMAL:
                yield from self._human_path_v2(rt, NORMAL)
            elif action == A.HUMAN_HIGH:
                yield from self._human_path_v2(rt, HIGH)
            elif action == A.CALLBACK:
                yield from self._callback_path_v2(rt)
            else:  # pragma: no cover - the POST mask forbids AI
                raise InvalidActionError(action)
            return

    def _wait_v2(self, rt: ContactRuntimeState, timer_minutes: float, resumed: bool = False):
        """Frozen V1.1 WAIT semantics (twin._wait) with an explicit timer (2.0, or the remainder when restored)."""
        if not resumed:
            self._set_state(rt, ContactState.WAITING)
            self.events.log(self.now, EventType.WAIT_STARTED, rt.contact_id, wait_number=rt.wait_count)
        else:
            self._set_state(rt, ContactState.WAITING)
        wake = self.env.event()
        self._wait_registry[rt.contact_id] = wake
        timer = self.env.timeout(timer_minutes)
        patience = self._patience_timer(rt)
        deadline = self._deadline_timer(rt)
        fired = yield self.env.any_of([wake, timer, patience, deadline])
        self._wait_registry.pop(rt.contact_id, None)
        if patience in fired or deadline in fired:
            self._end_wait_without_service(rt, fired, patience, deadline)
            return False
        cause = WakeCause.HUMAN_AVAILABLE if wake in fired else WakeCause.TIMER
        self.events.log(self.now, EventType.WAIT_REEVALUATION, rt.contact_id, cause=cause, wait_number=rt.wait_count)
        return True

    def _wait(self, rt):  # the base WAIT is never used by the V2 twin (all V2 waits go through _wait_v2)
        raise RuntimeError("V1.1 wait path is not used in V2")

    def _human_path(self, rt: ContactRuntimeState):  # V1.1 entry point is never used by the V2 twin
        raise RuntimeError("V1.1 human path is not used in V2")

    def _apply_handoff_once(self, rt: ContactRuntimeState) -> float:
        st = self.st(rt)
        h = self.config.v1_1.handoff_overhead_minutes
        if rt.escalated and h > 0 and not st.handoff_applied:
            st.handoff_applied = True
            rt.handoff_minutes = h
            self.handoff_events += 1
            return h
        return 0.0

    def _human_path_v2(self, rt: ContactRuntimeState, cls: str, t_entry: float | None = None, acc_before: float | None = None):
        self.st(rt).queue_class = cls
        self._set_state(rt, ContactState.HUMAN_QUEUE)
        self.events.log(self.now, EventType.HUMAN_QUEUE_ENTERED, rt.contact_id, escalated=rt.escalated,
                        humans_idle=self.humans_idle(), queue_ahead=len(self.humans.queue), queue_class=cls,
                        **({"restored": True} if t_entry is not None else {}))
        request = self.humans.request(cls=cls, acc_before=rt.human_queue_minutes if acc_before is None else acc_before,
                                      t_entry=t_entry, owner=rt.contact_id)
        patience, deadline = self._patience_timer(rt), self._deadline_timer(rt)
        fired = yield self.env.any_of([request, patience, deadline])
        if not request.triggered:
            request.cancel()
            self._end_wait_without_service(rt, fired, patience, deadline)
            return
        self._set_state(rt, ContactState.HUMAN_SERVICE)
        rt.human_touched = True
        self.class_waits[cls].append(rt.human_queue_minutes)
        self.indicators.record_answered(self.now, rt.online_wait_accumulated)
        extra_t = self._apply_handoff_once(rt)  # U2: actual connection
        extra = {"handoff_minutes": extra_t} if extra_t else {}
        self.events.log(self.now, EventType.HUMAN_SERVICE_STARTED, rt.contact_id, online_wait=rt.online_wait_accumulated, **extra)
        yield self.env.timeout(rt.truth.human_service_minutes + extra_t)
        self.events.log(self.now, EventType.HUMAN_SERVICE_COMPLETED, rt.contact_id)
        self._resolve(rt, Outcome.RESOLVED_HUMAN)
        self._release_human(request)

    def _ai_path(self, rt: ContactRuntimeState, resume_residual: float | None = None):
        """Frozen V1.1 AI path; only the consequence of a non-hangup visible failure differs (POST decision).
        ``resume_residual`` (restoration only): an attempt ALREADY in progress continues for its residual time; it is
        not a new attempt (no new start event) and its degradation status is that of its start (before the snapshot)."""
        if resume_residual is None:
            self._set_state(rt, ContactState.AI_QUEUE)
            self.events.log(self.now, EventType.AI_QUEUE_ENTERED, rt.contact_id, ai_sessions_busy=self.ai.count)
            request = self.ai.request()
            patience, deadline = self._patience_timer(rt), self._deadline_timer(rt)
            fired = yield self.env.any_of([request, patience, deadline])
            if not request.triggered:
                request.cancel()
                self._end_wait_without_service(rt, fired, patience, deadline)
                return
            self._set_state(rt, ContactState.AI_SERVICE)
            self.indicators.record_answered(self.now, rt.online_wait_accumulated)
            truth = rt.truth
            degraded = self._ai_degraded(self.now)
            deg = self.config.events.ai_degradation
            extra = {"degraded": True} if degraded else {}
            self.events.log(self.now, EventType.AI_ATTEMPT_STARTED, rt.contact_id, online_wait=rt.online_wait_accumulated, **extra)
            service = truth.ai_service_minutes * deg.service_time_multiplier if degraded else truth.ai_service_minutes
        else:
            request = self.ai.request()
            yield request
            self._set_state(rt, ContactState.AI_SERVICE)
            truth = rt.truth
            degraded = False  # the attempt started before the snapshot, i.e. before any hypothetical degradation window
            deg = self.config.events.ai_degradation
            service = resume_residual
        yield self.env.timeout(service)
        self.indicators.record_ai_completion(self.now)
        p_correct = truth.ai_correct_prob * deg.correct_multiplier if degraded else truth.ai_correct_prob
        correct = self._stream(Stream.AI_SUCCESS, truth).random() < p_correct
        p_not_correct = 1.0 - p_correct
        p_false_given_not = truth.ai_false_resolution_prob / p_not_correct if p_not_correct > 0 else 0.0
        false_resolution = not correct and self._stream(Stream.FALSE_AI_RESOLUTION, truth).random() < p_false_given_not
        if correct:
            self.events.log(self.now, EventType.AI_RESOLVED, rt.contact_id)
            self._resolve(rt, Outcome.RESOLVED_AI)
            self.ai.release(request)
        elif false_resolution:
            amount = self.reward.resolution(truth.value_weight)
            self.events.log(self.now, EventType.AI_FALSE_RESOLUTION, rt.contact_id, provisional_reward=amount,
                            discovery_at=self.now + truth.false_discovery_delay_minutes)
            self._set_state(rt, ContactState.AWAITING_FALSE_DISCOVERY)
            self.ai.release(request)
            self.env.process(self._false_discovery_process(rt))
        else:
            self.indicators.record_ai_failure_observed(self.now)
            self.events.log(self.now, EventType.AI_VISIBLE_FAILURE, rt.contact_id)
            v11 = self.config.v1_1
            hangup_p = v11.immediate_hangup_prob_after_ai_failure
            if hangup_p > 0 and self._stream(Stream.AI_FAILURE_HANGUP, truth).random() < hangup_p:
                self.ai.release(request)
                self._abandon(rt, cause="post_ai_failure_hangup")
                return
            rt.escalated = True
            extra = {}
            m = v11.frustration_patience_multiplier
            if m != 1.0:
                waited = self._waiting_time_so_far(rt)
                before = rt.truth.patience_minutes if rt.effective_patience is None else rt.effective_patience
                remaining = max(0.0, before - waited)
                rt.effective_patience = waited + remaining * m
                extra = {"remaining_patience_before": remaining, "remaining_patience_after": remaining * m}
            self.events.log(self.now, EventType.AI_ESCALATION, rt.contact_id, to="POST_AI_FAILURE_DECISION", **extra)
            self.ai.release(request)
            yield from self._post_failure(rt)

    def _callback_path_v2(self, rt: ContactRuntimeState, resume: dict[str, Any] | None = None):
        """Frozen V1.1 callback lifecycle; the handoff (escalated contacts) is added once at the ANSWERED attempt.

        ``resume`` (restoration only) continues a callback created before the snapshot, keeping its creation time
        (window), attempt count and attempt times; phase QUEUE (pending, next dial not before ``next_request_at``),
        ATTEMPT (dial in progress, advisor already held) or SERVICE (connected, advisor already held). The dropout
        clock is re-drawn from now: the frozen dropout hazard is constant, so the remaining time is Exp(hazard)."""
        cb = self.config.callback
        self.st(rt).queue_class = CALLBACK
        hazard = cb.dropout_hazard_per_minute
        answer_rng = self._stream(Stream.CALLBACK, rt.truth)
        if resume is None:
            self._set_state(rt, ContactState.CALLBACK_QUEUE)
            created = self.now
            rt.callback_started_at = created
            window_end = created + cb.max_delay_minutes if cb.max_delay_minutes is not None else None
            dropout_at = None
            if hazard > 0:
                dropout_at = created + float(self._stream(Stream.CALLBACK_DROPOUT, rt.truth).exponential(1.0 / hazard))
            self.events.log(self.now, EventType.CALLBACK_SCHEDULED, rt.contact_id, model=cb.model, window_end=window_end, dropout_at=dropout_at)
            phase, next_request_at, t_entry, remaining = "QUEUE", self.now, None, None
        else:
            created = resume["created"]
            rt.callback_started_at = created
            rt.callback_attempts = resume["attempts"]
            rt.callback_attempt_times = list(resume["attempt_times"])
            window_end = created + cb.max_delay_minutes if cb.max_delay_minutes is not None else None
            dropout_at = self.now + float(self._stream(Stream.CALLBACK_DROPOUT, rt.truth).exponential(1.0 / hazard)) if hazard > 0 else None
            phase, next_request_at, t_entry, remaining = resume["phase"], resume.get("next_request_at", self.now), resume.get("t_entry"), resume.get("remaining")
        request = None
        while True:
            if phase == "QUEUE":
                if rt.state is not ContactState.CALLBACK_QUEUE:
                    self._set_state(rt, ContactState.CALLBACK_QUEUE)
                if next_request_at > self.now + 1e-12:  # retry delay still running (restored contacts only)
                    retry = self.env.timeout(next_request_at - self.now)
                    clocks = self._callback_clocks(rt, dropout_at, window_end)
                    fired = yield self.env.any_of([retry, *clocks.values()])
                    if any(ev in fired for ev in clocks.values()):
                        self._end_callback_pending(rt, fired, clocks)
                        return
                request = self.humans.request(cls=CALLBACK, t_entry=t_entry, owner=rt.contact_id)
                t_entry = None
                clocks = self._callback_clocks(rt, dropout_at, window_end)
                fired = yield self.env.any_of([request, *clocks.values()])
                if not request.triggered:
                    request.cancel()
                    self._end_callback_pending(rt, fired, clocks)
                    return
                self._set_state(rt, ContactState.CALLBACK_ATTEMPT)
                rt.callback_attempts += 1
                rt.callback_attempt_times.append(self.now)
                self.events.log(self.now, EventType.CALLBACK_ATTEMPT_STARTED, rt.contact_id, attempt=rt.callback_attempts)
                remaining = cb.attempt_minutes
                phase = "ATTEMPT"
            if phase in ("ATTEMPT", "SERVICE") and request is None:  # restored: the advisor is already held
                request = self.humans.request(cls=HOLD, owner=rt.contact_id)
                yield request
                self._set_state(rt, ContactState.CALLBACK_ATTEMPT if phase == "ATTEMPT" else ContactState.CALLBACK_SERVICE)
            if phase == "ATTEMPT":
                yield self.env.timeout(remaining)
                if answer_rng.random() < cb.answer_probability:
                    self._set_state(rt, ContactState.CALLBACK_SERVICE)
                    rt.human_touched = True
                    rt.callback_completed_at = self.now
                    extra_t = self._apply_handoff_once(rt)  # U2: actual connection (answered callback)
                    self.events.log(self.now, EventType.CALLBACK_REACHED, rt.contact_id, attempt=rt.callback_attempts,
                                    pending_minutes=rt.callback_pending_minutes, **({"handoff_minutes": extra_t} if extra_t else {}))
                    yield self.env.timeout(rt.truth.human_service_minutes + extra_t)
                    rt.callback_reason = CallbackReason.CALLBACK_SUCCESS
                    self._resolve(rt, Outcome.RESOLVED_CALLBACK)
                    self._release_human(request)
                    return
                self.events.log(self.now, EventType.CALLBACK_ATTEMPT_FAILED, rt.contact_id, attempt=rt.callback_attempts)
                if rt.callback_attempts >= cb.max_attempts:
                    self._callback_terminal(rt, CallbackReason.MAX_ATTEMPTS)
                    self._release_human(request)
                    return
                self._set_state(rt, ContactState.CALLBACK_QUEUE)
                self._release_human(request)
                request = None
                retry = self.env.timeout(cb.retry_delay_minutes)
                clocks = self._callback_clocks(rt, dropout_at, window_end)
                fired = yield self.env.any_of([retry, *clocks.values()])
                if any(ev in fired for ev in clocks.values()):
                    self._end_callback_pending(rt, fired, clocks)
                    return
                phase, next_request_at = "QUEUE", self.now
                continue
            # phase == "SERVICE" (restored): connected callback, residual service, then success
            rt.human_touched = True
            yield self.env.timeout(remaining)
            rt.callback_reason = CallbackReason.CALLBACK_SUCCESS
            self._resolve(rt, Outcome.RESOLVED_CALLBACK)
            self._release_human(request)
            return

    def _callback_path(self, rt):  # V1.1 entry point is never used by the V2 twin
        raise RuntimeError("V1.1 callback path is not used in V2")

    def _resume_human_service(self, rt: ContactRuntimeState, residual: float):
        """Restored HUMAN_SERVICE: the advisor is already held; the service completes after its residual time."""
        request = self.humans.request(cls=HOLD, owner=rt.contact_id)
        yield request
        self._set_state(rt, ContactState.HUMAN_SERVICE)
        rt.human_touched = True
        yield self.env.timeout(residual)
        self.events.log(self.now, EventType.HUMAN_SERVICE_COMPLETED, rt.contact_id)
        self._resolve(rt, Outcome.RESOLVED_HUMAN)
        self._release_human(request)

    # ------------------------------------------------------------------------------------------- restoration
    def seed_contact(self, truth: GroundTruthContact, observed: ObservedContactFeatures, spec: RestoreSpec, start: float) -> None:
        """Hypothetical worlds only: restore an in-system contact at ``start`` from its OBSERVABLE lifecycle state.

        ``truth`` holds hidden attributes re-drawn from the frozen distributions (see world._seed_truth); for busy /
        in-progress contacts its service time is the TOTAL duration, conditioned on exceeding the elapsed time."""
        self.env.process(self._seeded_process(truth, observed, spec, start))

    def _seeded_process(self, truth, observed, sp: RestoreSpec, start: float):
        delay = start - self.now
        if delay > 0:
            yield self.env.timeout(delay)
        self.accrue()
        rt = ContactRuntimeState(truth=truth, observed=observed, state=ContactState.PENDING_DECISION, state_entered_at=self.now)
        rt.escalated, rt.wait_count, rt.routing_attempt_count = sp.escalated, sp.wait_count, sp.routing_attempt_count
        rt.online_wait_accumulated = sp.online_wait
        rt.human_queue_minutes = sp.hq_prev + (sp.elapsed_in_state if sp.state == "HUMAN_QUEUE" else 0.0)
        m = self.config.v1_1.frustration_patience_multiplier
        if sp.escalated and m != 1.0:
            rt.effective_patience = sp.online_wait + max(0.0, truth.patience_minutes - sp.online_wait) * m
        self._state_counts[rt.state] += 1
        self.contacts[truth.contact_id] = rt
        self._active[truth.contact_id] = rt
        self.restored[truth.contact_id] = sp
        st = self.st(rt)
        st.decision_type, st.handoff_applied, st.queue_class = sp.decision_type, sp.handoff_applied, sp.queue_class
        self.events.log(self.now, EventType.ARRIVAL, truth.contact_id, restored=True, state=sp.state)
        cb = self.config.callback
        h = self.config.v1_1.handoff_overhead_minutes if sp.handoff_applied else 0.0
        s = sp.state
        if s == "PENDING_INITIAL":
            yield from self._run_initial(rt)
        elif s == "PENDING_POST":
            yield from self._post_failure(rt)
        elif s == "WAITING":
            yield from self._post_failure(rt, resume_wait_minutes=max(0.0, self.config.wait.reevaluation_minutes - sp.elapsed_in_state))
        elif s == "HUMAN_QUEUE":
            yield from self._human_path_v2(rt, sp.queue_class or NORMAL, t_entry=sp.entered_at, acc_before=sp.hq_prev)
        elif s == "HUMAN_SERVICE":
            yield from self._resume_human_service(rt, max(0.0, truth.human_service_minutes + h - sp.elapsed_in_state))
        elif s == "AI_QUEUE":
            yield from self._ai_path(rt)
        elif s == "AI_SERVICE":
            yield from self._ai_path(rt, resume_residual=max(0.0, truth.ai_service_minutes - sp.elapsed_in_state))
        elif s in ("CALLBACK_QUEUE", "CALLBACK_ATTEMPT", "CALLBACK_SERVICE"):
            phase = {"CALLBACK_QUEUE": "QUEUE", "CALLBACK_ATTEMPT": "ATTEMPT", "CALLBACK_SERVICE": "SERVICE"}[s]
            last = sp.callback_attempt_times[-1] if sp.callback_attempt_times else None
            resume = {"created": sp.callback_started_at, "attempts": sp.callback_attempts, "attempt_times": sp.callback_attempt_times,
                      "phase": phase}
            if phase == "QUEUE":
                # after a failed dial the next request waits the frozen retry delay; a never-dialled callback is queued
                retry_at = (last + cb.attempt_minutes + cb.retry_delay_minutes) if last is not None else None
                resume["next_request_at"] = retry_at if retry_at is not None else start
                # queue position: a never-dialled callback has queued since creation; after a failed dial the request
                # was (re)issued when the retry delay ended; if that is still in the future it is issued then
                resume["t_entry"] = sp.entered_at if retry_at is None else (retry_at if retry_at <= start else None)
            elif phase == "ATTEMPT":
                resume["remaining"] = max(0.0, cb.attempt_minutes - sp.elapsed_in_state)
            else:
                resume["remaining"] = max(0.0, truth.human_service_minutes + h - sp.elapsed_in_state)
            yield from self._callback_path_v2(rt, resume=resume)
        else:  # pragma: no cover
            raise ValueError(s)


class V2Environment:
    """Minimal controller around the V2 twin (decision points, actions, reward observer, end of day)."""

    def __init__(self, twin: V2Twin) -> None:
        self.twin = twin
        self.current: V2DecisionPoint | None = None

    def next_decision(self) -> V2DecisionPoint | None:
        rt = self.twin.advance_to_next_decision()
        if rt is None:
            self.twin.finalize()
            self.current = None
            return None
        obs, mask = self.twin.observe_v2(rt)
        self.current = V2DecisionPoint(self.twin.new_decision_id(), rt.contact_id, self.twin.now, obs.decision_type, obs, mask)
        return self.current

    def step(self, action: str) -> V2DecisionPoint | None:
        dp = self.current
        self.twin.apply_decision(dp.decision_id, action, dp.mask)
        return self.next_decision()

    def class_metrics(self) -> dict[str, Any]:
        """Descriptive fairness metrics by human-queue class (spec §4.3)."""
        tw = self.twin
        out = {}
        for cls in (NORMAL, HIGH):
            ids = [cid for cid, st in tw.v2.items() if st.queue_class == cls]
            rts = [tw.contacts[c] for c in ids if c in tw.contacts]
            w = tw.class_waits[cls]
            out[cls] = {"n": len(rts), "served": len(w), "wait_mean": float(sum(w) / len(w)) if w else 0.0,
                        "wait_max": max(w) if w else 0.0, "wait_p95": float(sorted(w)[int(0.95 * (len(w) - 1))]) if w else 0.0,
                        "abandoned": sum(1 for r in rts if r.outcome is Outcome.ABANDONED),
                        "deadline_missed": sum(1 for r in rts if r.outcome is Outcome.MISSED_DEADLINE),
                        "resolved": sum(1 for r in rts if r.outcome in (Outcome.RESOLVED_HUMAN, Outcome.RESOLVED_CALLBACK))}
        return out

    def summary(self) -> dict[str, Any]:
        tw = self.twin
        rts = list(tw.contacts.values())
        return {"total_reward": tw.reward.episode.total, "reward": tw.reward.episode.to_dict(),
                "outcomes": {rt.contact_id: (rt.outcome.value if rt.outcome else None) for rt in rts},
                "aging_violations": tw.humans.aging_violations, "aging_overrides": tw.humans.aging_overrides,
                "handoff_events": tw.handoff_events, "classes": self.class_metrics()}
