"""Event-driven Digital Twin of the contact center (SimPy).

Each contact is a SimPy process. Whenever a contact needs a routing decision it
appends itself to ``pending`` and blocks on an event; the controller
(``RoutingEnvironment``) steps the SimPy clock until a decision is pending,
asks a policy, and resumes the contact with the chosen action.

Reward accounting
-----------------
Every contact state change goes through ``_set_state``, which first integrates
time-based costs over ALL contacts since the previous change (counts per state),
so waiting/busy/callback-pending minutes are center-level by construction.
Terminal outcomes go through ``_terminate``, which refuses a second terminal
outcome for the same contact — each event-based reward is charged once.
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Generator, Iterable, Sequence

import simpy

from callverse_workforce.config.models import SimulationConfig
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.contact import ContactRuntimeState, GroundTruthContact
from callverse_workforce.domain.enums import (
    AI_BUSY_STATES,
    CallbackReason,
    CALLBACK_PENDING_STATES,
    HUMAN_BUSY_STATES,
    ONLINE_WAITING_STATES,
    ContactState,
    EventType,
    Outcome,
    RoutingAction,
    WakeCause,
)
from callverse_workforce.domain.observation import CenterState, RoutingObservation, build_routing_observation
from callverse_workforce.forecasting.forecaster import DemandForecaster, build_forecaster
from callverse_workforce.intake.analyzer import RequestAnalyzer, SimulatedIntakeAnalyzer
from callverse_workforce.reward.accumulator import RewardAccumulator
from callverse_workforce.routing.mask import ServiceAvailability, compute_action_mask
from callverse_workforce.tracing.events import EventLog
from callverse_workforce.twin.center_metrics import CenterIndicators
from callverse_workforce.twin.generator import ContactGenerator, draw_recontact, scenario_window
from callverse_workforce.utils.rng import RngManager, Stream

ONLINE_PRIORITY = 0  # online human-queue requests are served before callbacks
CALLBACK_PRIORITY = 1

ProcessGen = Generator[simpy.Event, object, object]


class InvalidActionError(ValueError):
    """Raised when a masked action is submitted."""


class DigitalTwin:
    def __init__(
        self,
        config: SimulationConfig,
        seed: int,
        contacts: Sequence[GroundTruthContact] | None = None,
        forecaster: DemandForecaster | None = None,
        analyzer: RequestAnalyzer | None = None,
    ) -> None:
        self.config = config
        self.seed = seed
        self.rng = RngManager(seed)
        self.env = simpy.Environment()
        self.horizon = config.scenario.horizon_minutes

        if contacts is None:
            contacts = ContactGenerator(config, self.rng).generate_day()
        self.truth: list[GroundTruthContact] = _validated_contacts(contacts, self.horizon)

        self.analyzer = analyzer or SimulatedIntakeAnalyzer(config.intake_noise)
        obs = config.observation
        # The forecaster only ever receives the forecasting section (never `scenario`).
        self.forecaster = forecaster or build_forecaster(config.forecasting, obs.forecast_horizon_minutes)
        self.indicators = CenterIndicators(
            obs.arrival_rate_window_minutes,
            obs.ai_failure_window_minutes,
            obs.abandonment_window_minutes,
            obs.service_level_window_minutes,
            obs.service_level_threshold_minutes,
        )

        self.humans = simpy.PriorityResource(self.env, capacity=config.humans.n_advisors)
        self.ai = simpy.Resource(self.env, capacity=config.ai.max_concurrency)

        self.events = EventLog()
        self.reward = RewardAccumulator(config.reward)
        self.contacts: dict[str, ContactRuntimeState] = {}
        self._active: dict[str, ContactRuntimeState] = {}
        self.pending: deque[tuple[ContactRuntimeState, simpy.Event]] = deque()
        self._wait_registry: dict[str, simpy.Event] = {}  # insertion order = WAIT FIFO
        self._state_counts: Counter[ContactState] = Counter()
        self._last_accrual = 0.0
        self._observed_arrivals: list[float] = []
        self._decision_seq = 0
        self.finalized = False
        deg = config.events.ai_degradation
        # Scenario 5 hidden window (never observable; only its operational effects are).
        self.degradation_window = (
            scenario_window(self.rng, Stream.AI_DEGRADATION_TIMING, deg.start_min, deg.start_max, deg.duration_minutes)
            if deg is not None else None
        )
        # Headroom counterfactual branching (inactive in normal runs).
        self.future_variant: int | None = None
        self.branch_time: float | None = None
        self._arrival_queue: deque[GroundTruthContact] = deque(self.truth)
        self._arrival_gen = 0

        self.env.process(self._arrival_process(self._arrival_gen))
        short = config.events.human_shortage
        if short is not None:
            if short.absent_advisors >= config.humans.n_advisors:
                raise ValueError("human shortage must leave at least one advisor")
            self.shortage_window = scenario_window(self.rng, Stream.HUMAN_SHORTAGE_TIMING, short.start_min,
                                                   short.start_max, short.duration_minutes)
            self.env.process(self._shortage_process(short.absent_advisors, *self.shortage_window))
        else:
            self.shortage_window = None

    # ======================================================================
    # Controller API (used by RoutingEnvironment)
    # ======================================================================
    @property
    def now(self) -> float:
        return float(self.env.now)

    def advance_to_next_decision(self) -> ContactRuntimeState | None:
        """Run the simulation until a routing decision is pending, or the horizon is reached."""
        while not self.pending:
            if self.env.peek() >= self.horizon:
                return None
            self.env.step()
        return self.pending[0][0]

    def new_decision_id(self) -> str:
        self._decision_seq += 1
        return f"D{self._decision_seq:06d}"

    def observe(self, rt: ContactRuntimeState) -> tuple[RoutingObservation, ActionMask]:
        center = self.center_state()
        observation = build_routing_observation(
            rt.observed,
            waiting_time_so_far=self._waiting_time_so_far(rt),
            routing_attempt_count=rt.routing_attempt_count,
            center=center,
            now=self.now,
        )
        mask = compute_action_mask(observation, self.service_availability(), self.config.mask, self.config.wait)
        return observation, mask

    def apply_decision(self, decision_id: str, action: RoutingAction, mask: ActionMask) -> ContactRuntimeState:
        if not self.pending:
            raise RuntimeError("no routing decision is pending")
        if not mask.is_valid(action):
            raise InvalidActionError(f"action {action.value} is masked: {mask.reasons}")
        rt, event = self.pending.popleft()
        self.events.log(
            self.now,
            EventType.ROUTING_DECISION,
            rt.contact_id,
            decision_id,
            action=action,
            mask=mask.to_dict()["valid"],
            n_valid_actions=mask.n_valid,
            mask_reasons=mask.to_dict()["reasons"],
            routing_attempt_count=rt.routing_attempt_count,
        )
        rt.actions.append(action)
        rt.decision_ids.append(decision_id)
        rt.routing_attempt_count += 1
        if action is RoutingAction.WAIT:
            rt.wait_count += 1
        event.succeed(action)
        return rt

    def accrue(self) -> None:
        """Integrate time-based costs up to the current simulated time."""
        now = self.now
        dt = now - self._last_accrual
        if dt <= 0:
            return
        counts = self._state_counts
        self.reward.accrue_time(
            dt,
            n_human_busy=sum(counts[s] for s in HUMAN_BUSY_STATES),
            n_ai_busy=sum(counts[s] for s in AI_BUSY_STATES),
            n_online_waiting=sum(counts[s] for s in ONLINE_WAITING_STATES),
            n_callback_pending=sum(counts[s] for s in CALLBACK_PENDING_STATES),
        )
        self._last_accrual = now

    def finalize(self) -> None:
        """Advance to the horizon and settle every non-terminal contact (end of day)."""
        if self.finalized:
            return
        if self.pending:
            raise RuntimeError("cannot finalize while a routing decision is pending")
        if self.now < self.horizon:
            self.env.run(until=self.horizon)
        self.accrue()
        settle = self.config.false_ai_discovery.settle_at_horizon
        for rt in list(self._active.values()):
            v = rt.truth.value_weight
            if rt.state is ContactState.AWAITING_FALSE_DISCOVERY:
                if settle:
                    amount = self.reward.false_ai_detected(v)
                    self._terminate(rt, Outcome.FALSE_AI_DISCOVERED, EventType.FALSE_AI_SETTLED_AT_HORIZON, reward=amount)
                else:
                    self._terminate(rt, Outcome.FALSE_AI_UNDISCOVERED, EventType.FALSE_AI_SETTLED_AT_HORIZON, reward=0.0)
            else:
                state = rt.state
                if rt.callback_started_at is not None:
                    rt.callback_reason = CallbackReason.UNFINISHED  # never WINDOW_EXPIRED at closing
                    rt.callback_completed_at = rt.callback_completed_at or self.horizon
                amount = self.reward.unfinished(v)
                self._terminate(rt, Outcome.UNFINISHED, EventType.UNFINISHED_AT_HORIZON, state=state, reward=amount)
        self.finalized = True

    # ======================================================================
    # Observable center state
    # ======================================================================
    def humans_idle(self) -> int:
        """Advisors free and not already claimed by a queued request."""
        return max(0, self.humans.capacity - self.humans.count - len(self.humans.queue))

    def service_availability(self) -> ServiceAvailability:
        cb = self.config.callback
        return ServiceAvailability(
            ai_available=self.config.ai.available,
            callback_available=cb.available and self._callbacks_pending() < cb.max_pending,
        )

    def center_state(self) -> CenterState:
        now = self.now
        human_queue = [rt for rt in self._active.values() if rt.state is ContactState.HUMAN_QUEUE]
        callbacks = [rt for rt in self._active.values() if rt.state in CALLBACK_PENDING_STATES]
        return CenterState(
            humans_available=self.humans_idle(),
            human_utilization=self.humans.count / self.humans.capacity,
            human_queue_size=len(human_queue),
            oldest_human_wait=max((now - rt.state_entered_at for rt in human_queue), default=0.0),
            ai_load=(self.ai.count + len(self.ai.queue)) / self.ai.capacity,
            recent_ai_failure_rate=self.indicators.ai_failure_rate(now),
            wait_queue_size=self._state_counts[ContactState.WAITING],
            callbacks_pending=len(callbacks),
            oldest_callback_age=max((now - (rt.callback_started_at or now) for rt in callbacks), default=0.0),
            recent_arrival_rate=self.indicators.arrival_rate_per_hour(now),
            forecast_30min=self.forecaster.forecast(now, tuple(self._observed_arrivals)),
            recent_abandonment_rate=self.indicators.abandonment_rate(now),
            recent_service_level=self.indicators.service_level(now),
            day_type=self.config.scenario.day_type,
            clock_hour=self.config.scenario.start_hour + now / 60.0,
        )

    def _callbacks_pending(self) -> int:
        return sum(self._state_counts[s] for s in CALLBACK_PENDING_STATES)

    def _waiting_time_so_far(self, rt: ContactRuntimeState) -> float:
        extra = self.now - rt.state_entered_at if rt.state in ONLINE_WAITING_STATES else 0.0
        return rt.online_wait_accumulated + extra

    # ======================================================================
    # State bookkeeping
    # ======================================================================
    def _set_state(self, rt: ContactRuntimeState, new: ContactState) -> None:
        self.accrue()
        now = self.now
        old = rt.state
        elapsed = now - rt.state_entered_at
        if old in ONLINE_WAITING_STATES:
            rt.online_wait_accumulated += elapsed
        if old in CALLBACK_PENDING_STATES:
            rt.callback_pending_minutes += elapsed
        if old is ContactState.HUMAN_QUEUE:
            rt.human_queue_minutes += elapsed
        if old is ContactState.HUMAN_SERVICE:
            rt.human_direct_minutes += elapsed
        elif old in (ContactState.CALLBACK_ATTEMPT, ContactState.CALLBACK_SERVICE):
            rt.callback_human_minutes += elapsed
        self._state_counts[old] -= 1
        self._state_counts[new] += 1
        rt.state = new
        rt.state_entered_at = now

    def _terminate(self, rt: ContactRuntimeState, outcome: Outcome, event_type: EventType, **data: object) -> None:
        if rt.is_terminal:
            raise RuntimeError(f"{rt.contact_id} already terminal ({rt.outcome}); refusing {outcome}")
        self._set_state(rt, ContactState.TERMINAL)
        rt.outcome = outcome
        rt.terminal_time = self.now
        self._active.pop(rt.contact_id, None)
        self._wait_registry.pop(rt.contact_id, None)
        self.events.log(self.now, event_type, rt.contact_id, outcome=outcome, **data)

    def _resolve(self, rt: ContactRuntimeState, outcome: Outcome) -> None:
        amount = self.reward.resolution(rt.truth.value_weight)
        self._terminate(rt, outcome, EventType.CONTACT_RESOLVED, reward=amount)

    def _abandon(self, rt: ContactRuntimeState, cause: str | None = None) -> None:
        self.indicators.record_abandonment(self.now)
        amount = self.reward.abandonment(rt.truth.value_weight)
        extra = {"cause": cause} if cause else {}
        self._terminate(rt, Outcome.ABANDONED, EventType.ABANDONMENT, from_state=rt.state, reward=amount, **extra)

    def _miss_deadline(self, rt: ContactRuntimeState) -> None:
        amount = self.reward.missed_deadline(rt.truth.value_weight)
        self._terminate(rt, Outcome.MISSED_DEADLINE, EventType.DEADLINE_MISSED, from_state=rt.state, reward=amount)

    def _patience_timer(self, rt: ContactRuntimeState) -> simpy.Timeout:
        """Fires when cumulative online waiting reaches patience (call on entering an online state)."""
        patience = rt.truth.patience_minutes if rt.effective_patience is None else rt.effective_patience
        return self.env.timeout(max(0.0, patience - self._waiting_time_so_far(rt)))

    def _ai_degraded(self, t: float) -> bool:
        w = self.degradation_window
        return w is not None and w[0] <= t < w[1]

    def _deadline_timer(self, rt: ContactRuntimeState) -> simpy.Timeout:
        return self.env.timeout(max(0.0, rt.truth.deadline_at - self.now))

    def _end_wait_without_service(
        self, rt: ContactRuntimeState, fired: Iterable[simpy.Event], patience: simpy.Event, deadline: simpy.Event
    ) -> None:
        fired = list(fired)
        if patience in fired:
            self._abandon(rt)
        elif deadline in fired:
            self._miss_deadline(rt)
        else:  # pragma: no cover - defensive
            raise RuntimeError("wait ended without service, abandonment or deadline")

    # ======================================================================
    # SimPy processes
    # ======================================================================
    def _arrival_process(self, generation: int) -> ProcessGen:
        while self._arrival_queue:
            truth = self._arrival_queue[0]
            delay = truth.arrival_time - self.now
            if delay > 0:
                yield self.env.timeout(delay)
                if generation != self._arrival_gen:  # future arrivals were replaced at a branch point
                    return
            self._arrival_queue.popleft()
            self.env.process(self._contact_process(truth))

    def _shortage_process(self, absent: int, start: float, end: float) -> ProcessGen:
        """Scenario 6: ``absent`` advisors leave at ``start`` (after finishing their current call) until ``end``.

        Modelled as top-priority, non-preemptive resource holds; they are not contacts, so they never count as
        human busy minutes. Observable only through humans_available / utilization / queue effects.
        """
        yield self.env.timeout(start)
        holds = [self.humans.request(priority=-1) for _ in range(absent)]
        yield self.env.timeout(max(0.0, end - self.now))
        for h in holds:
            if h.triggered:
                self._release_human(h)
            else:
                h.cancel()

    def activate_future_variant(self, variant: int, future_contacts: Sequence[GroundTruthContact]) -> None:
        """Headroom branch point (called at the target decision, before the forced action is applied).

        From now on: (1) arrivals not yet realised are replaced by ``future_contacts`` (variant-k future,
        strictly after now); (2) every event-time random draw (intake noise, AI outcome, false-resolution
        split, callback answer/dropout, recontact) uses a stream keyed by ``(..., FUTURE_TAG, variant)``.
        Everything realised before now (history, hidden attributes of contacts already arrived) is kept.
        The draws depend on the variant and the contact, never on the forced action, so all actions of a
        state share future k (common random numbers).
        """
        if self.future_variant is not None:
            raise RuntimeError("future variant already active")
        now = self.now
        future = sorted(future_contacts, key=lambda c: (c.arrival_time, c.index))
        if any(c.arrival_time <= now for c in future):
            raise ValueError("future contacts must arrive strictly after the branch time")
        known = {c.rng_key for c in self.truth}
        if any(c.rng_key in known for c in future):
            raise ValueError("future contacts must have fresh RNG keys")
        self.future_variant = variant
        self.branch_time = now
        self._arrival_queue = deque(future)
        self._arrival_gen += 1
        self.env.process(self._arrival_process(self._arrival_gen))

    def _stream(self, name: str, truth: GroundTruthContact):
        """Per-contact event-time stream; variant-keyed after a headroom branch point."""
        return self.rng.contact_stream(name, truth.index, truth.recontact_depth, self.future_variant)

    def _register_arrival(self, truth: GroundTruthContact) -> ContactRuntimeState:
        now = self.now
        observed = self.analyzer.analyze_request(truth, self._stream(Stream.INTAKE_NOISE, truth), now)
        self.accrue()
        rt = ContactRuntimeState(truth=truth, observed=observed, state=ContactState.PENDING_DECISION, state_entered_at=now)
        self._state_counts[rt.state] += 1
        self.contacts[truth.contact_id] = rt
        self._active[truth.contact_id] = rt
        self._observed_arrivals.append(now)
        self.indicators.record_arrival(now)
        self.events.log(
            now,
            EventType.ARRIVAL,
            truth.contact_id,
            root_contact_id=truth.root_contact_id,
            parent_contact_id=truth.parent_contact_id,
            recontact_depth=truth.recontact_depth,
            truth={
                "request_type": truth.request_type,
                "urgency": truth.urgency,
                "sensitivity": truth.sensitivity,
                "channel": truth.channel,
                "request_value": truth.request_value,
                "deadline_at": truth.deadline_at,
                "patience_minutes": truth.patience_minutes,
                "ai_correct_prob": truth.ai_correct_prob,
                "ai_false_resolution_prob": truth.ai_false_resolution_prob,
            },
        )
        self.events.log(
            now,
            EventType.INTAKE_PREDICTION,
            truth.contact_id,
            request_type=observed.request_type,
            urgency_pred=observed.urgency_pred,
            sensitivity_pred=observed.sensitivity_pred,
            predicted_deadline_at=observed.predicted_deadline_at,
            ai_resolution_score=observed.ai_resolution_score,
            callback_allowed=observed.callback_allowed,
            human_requested=observed.human_requested,
        )
        return rt

    def _contact_process(self, truth: GroundTruthContact) -> ProcessGen:
        rt = self._register_arrival(truth)
        while True:
            if rt.state is not ContactState.PENDING_DECISION:
                self._set_state(rt, ContactState.PENDING_DECISION)
            decision = self.env.event()
            self.pending.append((rt, decision))
            action = yield decision
            if action is RoutingAction.WAIT:
                reevaluate = yield from self._wait(rt)
                if reevaluate:
                    continue
            elif action is RoutingAction.AI:
                yield from self._ai_path(rt)
            elif action is RoutingAction.HUMAN:
                yield from self._human_path(rt)
            else:
                yield from self._callback_path(rt)
            return

    # --- WAIT ----------------------------------------------------------------
    def _wait(self, rt: ContactRuntimeState) -> ProcessGen:
        """Defer the contact; return True if it must be re-routed, False if it ended."""
        self._set_state(rt, ContactState.WAITING)
        self.events.log(self.now, EventType.WAIT_STARTED, rt.contact_id, wait_number=rt.wait_count)
        wake = self.env.event()
        self._wait_registry[rt.contact_id] = wake
        timer = self.env.timeout(self.config.wait.reevaluation_minutes)
        patience = self._patience_timer(rt)
        deadline = self._deadline_timer(rt)
        fired = yield self.env.any_of([wake, timer, patience, deadline])
        self._wait_registry.pop(rt.contact_id, None)
        if patience in fired or deadline in fired:
            self._end_wait_without_service(rt, fired, patience, deadline)
            return False
        cause = WakeCause.HUMAN_AVAILABLE if wake in fired else WakeCause.TIMER
        self.events.log(
            self.now, EventType.WAIT_REEVALUATION, rt.contact_id, cause=cause, wait_number=rt.wait_count
        )
        return True

    def _wake_waiting_contacts(self, _event: simpy.Event | None = None) -> None:
        """Bring WAIT contacts back early, oldest first, one per truly idle advisor."""
        idle = self.humans_idle()
        for contact_id in list(self._wait_registry):
            if idle <= 0:
                break
            wake = self._wait_registry.pop(contact_id)
            if not wake.triggered:
                wake.succeed(WakeCause.HUMAN_AVAILABLE)
                idle -= 1

    def _release_human(self, request: simpy.Request) -> None:
        release = self.humans.release(request)
        # Queued requests are granted when the release event is processed;
        # only then do we know whether an advisor is genuinely idle.
        release.callbacks.append(self._wake_waiting_contacts)

    # --- HUMAN ---------------------------------------------------------------
    def _human_path(self, rt: ContactRuntimeState) -> ProcessGen:
        self._set_state(rt, ContactState.HUMAN_QUEUE)
        self.events.log(
            self.now,
            EventType.HUMAN_QUEUE_ENTERED,
            rt.contact_id,
            escalated=rt.escalated,
            humans_idle=self.humans_idle(),
            queue_ahead=len(self.humans.queue),
        )
        request = self.humans.request(priority=ONLINE_PRIORITY)
        patience, deadline = self._patience_timer(rt), self._deadline_timer(rt)
        fired = yield self.env.any_of([request, patience, deadline])
        if not request.triggered:
            request.cancel()
            self._end_wait_without_service(rt, fired, patience, deadline)
            return
        self._set_state(rt, ContactState.HUMAN_SERVICE)
        rt.human_touched = True
        self.indicators.record_answered(self.now, rt.online_wait_accumulated)
        service = rt.truth.human_service_minutes
        extra = {}
        handoff = self.config.v1_1.handoff_overhead_minutes
        if rt.escalated and handoff > 0:  # V1.1 A: only after visible AI failure -> human escalation
            rt.handoff_minutes = handoff
            service += handoff
            extra["handoff_minutes"] = handoff
        self.events.log(
            self.now, EventType.HUMAN_SERVICE_STARTED, rt.contact_id, online_wait=rt.online_wait_accumulated, **extra
        )
        yield self.env.timeout(service)
        self.events.log(self.now, EventType.HUMAN_SERVICE_COMPLETED, rt.contact_id)
        self._resolve(rt, Outcome.RESOLVED_HUMAN)
        self._release_human(request)

    # --- AI ------------------------------------------------------------------
    def _ai_path(self, rt: ContactRuntimeState) -> ProcessGen:
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
        degraded = self._ai_degraded(self.now)  # scenario 5: attempts STARTED inside the hidden window
        deg = self.config.events.ai_degradation
        extra = {"degraded": True} if degraded else {}
        self.events.log(self.now, EventType.AI_ATTEMPT_STARTED, rt.contact_id, online_wait=rt.online_wait_accumulated,
                        **extra)
        service = truth.ai_service_minutes * deg.service_time_multiplier if degraded else truth.ai_service_minutes
        yield self.env.timeout(service)

        self.indicators.record_ai_completion(self.now)
        p_correct = truth.ai_correct_prob * deg.correct_multiplier if degraded else truth.ai_correct_prob
        correct = self._stream(Stream.AI_SUCCESS, truth).random() < p_correct
        p_not_correct = 1.0 - p_correct
        p_false_given_not = truth.ai_false_resolution_prob / p_not_correct if p_not_correct > 0 else 0.0
        false_resolution = (
            not correct
            and self._stream(Stream.FALSE_AI_RESOLUTION, truth).random() < p_false_given_not
        )

        if correct:
            self.events.log(self.now, EventType.AI_RESOLVED, rt.contact_id)
            self._resolve(rt, Outcome.RESOLVED_AI)
            self.ai.release(request)
        elif false_resolution:
            # Looks resolved to the center; the error surfaces later.
            amount = self.reward.resolution(truth.value_weight)
            self.events.log(
                self.now,
                EventType.AI_FALSE_RESOLUTION,
                rt.contact_id,
                provisional_reward=amount,
                discovery_at=self.now + truth.false_discovery_delay_minutes,
            )
            self._set_state(rt, ContactState.AWAITING_FALSE_DISCOVERY)
            self.ai.release(request)
            self.env.process(self._false_discovery_process(rt))
        else:
            self.indicators.record_ai_failure_observed(self.now)
            self.events.log(self.now, EventType.AI_VISIBLE_FAILURE, rt.contact_id)
            v11 = self.config.v1_1
            hangup_p = v11.immediate_hangup_prob_after_ai_failure
            if hangup_p > 0 and self._stream(Stream.AI_FAILURE_HANGUP, truth).random() < hangup_p:
                # V1.1 C: customer loss right after the failed bot, before escalation (abandonment, once).
                self.ai.release(request)
                self._abandon(rt, cause="post_ai_failure_hangup")
                return
            rt.escalated = True
            extra = {}
            m = v11.frustration_patience_multiplier
            if m != 1.0:  # V1.1 B: scale only the REMAINING patience; time already waited is kept
                waited = self._waiting_time_so_far(rt)
                before = rt.truth.patience_minutes if rt.effective_patience is None else rt.effective_patience
                remaining = max(0.0, before - waited)
                rt.effective_patience = waited + remaining * m
                extra = {"remaining_patience_before": remaining, "remaining_patience_after": remaining * m}
            self.events.log(self.now, EventType.AI_ESCALATION, rt.contact_id, to="HUMAN_QUEUE", **extra)
            self.ai.release(request)
            yield from self._human_path(rt)

    def _false_discovery_process(self, rt: ContactRuntimeState) -> ProcessGen:
        yield self.env.timeout(rt.truth.false_discovery_delay_minutes)
        if rt.is_terminal:
            return
        self.indicators.record_ai_failure_observed(self.now)
        amount = self.reward.false_ai_detected(rt.truth.value_weight)
        self._terminate(rt, Outcome.FALSE_AI_DISCOVERED, EventType.FALSE_AI_DISCOVERED, reward=amount)
        self._maybe_schedule_recontact(rt)

    def _maybe_schedule_recontact(self, parent: ContactRuntimeState) -> None:
        """After an in-day discovery, the customer may come back as a new contact instance (depth <= max)."""
        rc = self.config.recontact
        truth = parent.truth
        if not rc.enabled or truth.recontact_depth >= rc.max_depth:
            return
        rng = self._stream(Stream.RECONTACT, truth)
        happens, recontact = draw_recontact(truth, self.now, rng, self.config)
        if not happens or recontact.arrival_time >= self.horizon:
            reason = "customer_does_not_return" if not happens else "after_horizon"
            self.events.log(self.now, EventType.RECONTACT_NOT_CREATED, truth.contact_id, reason=reason)
            return
        self.events.log(
            self.now,
            EventType.RECONTACT_SCHEDULED,
            truth.contact_id,
            recontact_id=recontact.contact_id,
            arrival_time=recontact.arrival_time,
        )
        self.env.process(self._recontact_arrival_process(recontact))

    def _recontact_arrival_process(self, truth: GroundTruthContact) -> ProcessGen:
        yield self.env.timeout(truth.arrival_time - self.now)
        yield from self._contact_process(truth)

    # --- CALLBACK ------------------------------------------------------------
    def _callback_path(self, rt: ContactRuntimeState) -> ProcessGen:
        """Callback lifecycle; B1 and B1.5 differ only through ``CallbackConfig`` parameters.

        * dial attempts: one advisor attempt of ``attempt_minutes``; answered with ``answer_probability``;
        * B1.5 completion window: if not reached within ``max_delay_minutes`` -> WINDOW_EXPIRED (-1);
        * B1.5 dropout: constant hazard lambda; the dropout clock is ONE draw E ~ Exp(lambda) from the
          contact's ``callback_dropout`` stream at creation, so dropout happens after exactly E minutes of
          pending time. For any interval dt, P(drop in dt | still pending) = 1 - exp(-lambda dt), which is
          exact for irregular simulated time (memoryless). Dropout -> abandonment (-3V) only.
        A clock that passes during a dial attempt takes effect when the attempt ends unanswered; an
        answered attempt is a success. Simultaneous clocks resolve as dropout > deadline > window.
        Timers at or after the horizon never fire: those callbacks become UNFINISHED (-2V) at closing.
        """
        cb = self.config.callback
        self._set_state(rt, ContactState.CALLBACK_QUEUE)
        created = self.now
        rt.callback_started_at = created
        window_end = created + cb.max_delay_minutes if cb.max_delay_minutes is not None else None
        hazard = cb.dropout_hazard_per_minute
        dropout_at = None
        if hazard > 0:
            dropout_rng = self._stream(Stream.CALLBACK_DROPOUT, rt.truth)
            dropout_at = created + float(dropout_rng.exponential(1.0 / hazard))
        self.events.log(
            self.now, EventType.CALLBACK_SCHEDULED, rt.contact_id,
            model=cb.model, window_end=window_end, dropout_at=dropout_at,  # dropout_at: simulator truth
        )
        answer_rng = self._stream(Stream.CALLBACK, rt.truth)
        while True:
            request = self.humans.request(priority=CALLBACK_PRIORITY)
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
            yield self.env.timeout(cb.attempt_minutes)

            if answer_rng.random() < cb.answer_probability:
                self._set_state(rt, ContactState.CALLBACK_SERVICE)
                rt.human_touched = True
                rt.callback_completed_at = self.now
                self.events.log(
                    self.now,
                    EventType.CALLBACK_REACHED,
                    rt.contact_id,
                    attempt=rt.callback_attempts,
                    pending_minutes=rt.callback_pending_minutes,
                )
                yield self.env.timeout(rt.truth.human_service_minutes)
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
            retry = self.env.timeout(cb.retry_delay_minutes)
            clocks = self._callback_clocks(rt, dropout_at, window_end)  # clocks already due fire at once
            fired = yield self.env.any_of([retry, *clocks.values()])
            if any(ev in fired for ev in clocks.values()):
                self._end_callback_pending(rt, fired, clocks)
                return

    def _callback_clocks(
        self, rt: ContactRuntimeState, dropout_at: float | None, window_end: float | None
    ) -> dict[CallbackReason, simpy.Timeout]:
        """Pending-callback terminal clocks, in precedence order."""
        clocks = {}
        if dropout_at is not None:
            clocks[CallbackReason.CUSTOMER_DROPOUT] = self.env.timeout(max(0.0, dropout_at - self.now))
        clocks[CallbackReason.MISSED_DEADLINE] = self._deadline_timer(rt)
        if window_end is not None:
            clocks[CallbackReason.WINDOW_EXPIRED] = self.env.timeout(max(0.0, window_end - self.now))
        return clocks

    def _end_callback_pending(
        self, rt: ContactRuntimeState, fired: Iterable[simpy.Event], clocks: dict[CallbackReason, simpy.Timeout]
    ) -> None:
        fired = list(fired)
        reason = next(r for r, ev in clocks.items() if ev in fired)
        self._callback_terminal(rt, reason)

    def _callback_terminal(self, rt: ContactRuntimeState, reason: CallbackReason) -> None:
        """Frozen callback reward mapping; each branch charges exactly one event term."""
        rt.callback_reason = reason
        rt.callback_completed_at = self.now
        v = rt.truth.value_weight
        if reason is CallbackReason.CUSTOMER_DROPOUT:
            amount = self.reward.abandonment(v)  # -3V only, never also callback failure
            self._terminate(rt, Outcome.CALLBACK_DROPOUT, EventType.CALLBACK_DROPOUT, reward=amount)
        elif reason is CallbackReason.MISSED_DEADLINE:
            self._miss_deadline(rt)
        elif reason is CallbackReason.WINDOW_EXPIRED:
            amount = self.reward.callback_failure()  # -1
            self._terminate(rt, Outcome.CALLBACK_WINDOW_EXPIRED, EventType.CALLBACK_WINDOW_EXPIRED, reward=amount)
        elif reason is CallbackReason.MAX_ATTEMPTS:
            amount = self.reward.callback_failure()  # -1
            self._terminate(rt, Outcome.CALLBACK_FAILED, EventType.CALLBACK_FAILED, reason=reason, reward=amount)
        else:  # pragma: no cover - success and end-of-day are handled elsewhere
            raise ValueError(reason)


def _validated_contacts(contacts: Sequence[GroundTruthContact], horizon: float) -> list[GroundTruthContact]:
    ordered = sorted(contacts, key=lambda c: (c.arrival_time, c.index))
    ids = [c.contact_id for c in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("contact_ids must be unique")
    if len({c.rng_key for c in ordered}) != len(ordered):
        raise ValueError("contact (index, recontact_depth) keys must be unique")
    for c in ordered:
        if not 0 <= c.arrival_time < horizon:
            raise ValueError(f"{c.contact_id} arrives outside [0, horizon)")
    return ordered
