"""Decision-level environment API over the Digital Twin (SMDP view).

``reset`` returns the first ``DecisionPoint``; ``step(action)`` applies the
action, runs the twin to the next decision (or the end of day) and returns the
``Transition`` whose reward covers ``[t_k, t_{k+1})`` for the whole center.
This is the interface the future MaskablePPO wrapper will build on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from callverse_workforce.config.models import SimulationConfig
from callverse_workforce.domain.contact import GroundTruthContact
from callverse_workforce.domain.enums import RoutingAction
from callverse_workforce.domain.records import ContactSummary, DecisionPoint, RoutingDecision, Transition
from callverse_workforce.forecasting.forecaster import DemandForecaster
from callverse_workforce.intake.analyzer import RequestAnalyzer
from callverse_workforce.reward.breakdown import RewardBreakdown
from callverse_workforce.twin.digital_twin import DigitalTwin
from callverse_workforce.twin.result import SimulationResult


@dataclass(frozen=True)
class StepResult:
    transition: Transition
    next_decision: DecisionPoint | None
    done: bool


class RoutingEnvironment:
    def __init__(
        self,
        config: SimulationConfig,
        contacts: Sequence[GroundTruthContact] | None = None,
        forecaster: DemandForecaster | None = None,
        analyzer: RequestAnalyzer | None = None,
    ) -> None:
        self.config = config
        self._contacts = contacts
        self._forecaster = forecaster
        self._analyzer = analyzer
        self.twin: DigitalTwin | None = None
        self.current: DecisionPoint | None = None
        self.decisions: list[RoutingDecision] = []
        self.transitions: list[Transition] = []
        self.pre_decision_reward = RewardBreakdown()
        self.done = True
        self.seed = config.seed

    def reset(self, seed: int | None = None) -> DecisionPoint | None:
        self.seed = self.config.seed if seed is None else seed
        self.twin = DigitalTwin(self.config, self.seed, self._contacts, self._forecaster, self._analyzer)
        self.decisions, self.transitions = [], []
        self.current = self._next_decision_point()
        if self.current is None:
            self.twin.finalize()
        self.twin.accrue()
        self.pre_decision_reward = self.twin.reward.drain()
        self.done = self.current is None
        return self.current

    def step(self, action: RoutingAction) -> StepResult:
        if self.done or self.current is None or self.twin is None:
            raise RuntimeError("episode is finished; call reset()")
        dp = self.current
        self.twin.apply_decision(dp.decision_id, action, dp.action_mask)
        self.decisions.append(
            RoutingDecision(
                decision_id=dp.decision_id,
                contact_id=dp.contact_id,
                timestamp=dp.timestamp,
                action=action,
                action_mask=dp.action_mask,
                n_valid_actions=dp.action_mask.n_valid,
                routing_attempt_count=dp.observation.routing_attempt_count,
            )
        )

        nxt = self._next_decision_point()
        if nxt is None:
            self.twin.finalize()
        self.twin.accrue()
        breakdown = self.twin.reward.drain()
        t_next = nxt.timestamp if nxt is not None else self.twin.horizon
        transition = Transition(
            decision_id=dp.decision_id,
            contact_id=dp.contact_id,
            timestamp=dp.timestamp,
            state=dp.observation,
            action=action,
            action_mask=dp.action_mask,
            reward=breakdown.total,
            reward_breakdown=breakdown,
            next_state=nxt.observation if nxt else None,
            next_action_mask=nxt.action_mask if nxt else None,
            delta_t_minutes=t_next - dp.timestamp,
            done=nxt is None,
        )
        self.transitions.append(transition)
        self.current = nxt
        self.done = nxt is None
        return StepResult(transition, nxt, self.done)

    def transition_discount(self, delta_t_minutes: float) -> float:
        """``gamma_per_minute ** delta_t`` (exposed for the future SMDP learner)."""
        return self.config.smdp.gamma_per_minute**delta_t_minutes

    def _next_decision_point(self) -> DecisionPoint | None:
        assert self.twin is not None
        rt = self.twin.advance_to_next_decision()
        if rt is None:
            return None
        observation, mask = self.twin.observe(rt)
        return DecisionPoint(
            decision_id=self.twin.new_decision_id(),
            contact_id=rt.contact_id,
            timestamp=self.twin.now,
            observation=observation,
            action_mask=mask,
        )

    def result(self) -> SimulationResult:
        if not self.done or self.twin is None:
            raise RuntimeError("episode not finished")
        twin = self.twin
        ambiguous = [rt.contact_id for rt in twin.contacts.values() if not rt.is_terminal]
        if ambiguous:
            raise RuntimeError(f"contacts without terminal outcome: {ambiguous[:5]}")
        contacts = [
            ContactSummary(
                contact_id=rt.contact_id,
                root_contact_id=rt.truth.root_contact_id,  # type: ignore[arg-type]  # filled in __post_init__
                parent_contact_id=rt.truth.parent_contact_id,
                recontact_depth=rt.truth.recontact_depth,
                request_type=rt.truth.request_type,
                channel=rt.truth.channel,
                language_dialect=rt.truth.language_dialect,
                arrival_time=rt.truth.arrival_time,
                value_weight=rt.truth.value_weight,
                outcome=rt.outcome,  # type: ignore[arg-type]  # all terminal after finalize
                terminal_time=rt.terminal_time,  # type: ignore[arg-type]
                online_wait_minutes=rt.online_wait_accumulated,
                callback_pending_minutes=rt.callback_pending_minutes,
                n_decisions=len(rt.decision_ids),
                n_waits=rt.wait_count,
                actions=tuple(rt.actions),
                escalated=rt.escalated,
                callback_attempts=rt.callback_attempts,
                callback_created_at=rt.callback_started_at,
                callback_attempt_times=tuple(rt.callback_attempt_times),
                callback_completed_at=rt.callback_completed_at,
                callback_reason=rt.callback_reason,
                human_touched=rt.human_touched,
                human_queue_minutes=rt.human_queue_minutes,
                human_direct_minutes=rt.human_direct_minutes,
                callback_human_minutes=rt.callback_human_minutes,
            )
            for rt in twin.contacts.values()
        ]
        return SimulationResult(
            seed=self.seed,
            scenario_name=self.config.scenario.name,
            horizon_minutes=twin.horizon,
            n_advisors=self.config.humans.n_advisors,
            contacts=contacts,
            decisions=list(self.decisions),
            transitions=list(self.transitions),
            events=twin.events.records,
            episode_reward=twin.reward.episode.copy(),
            pre_decision_reward=self.pre_decision_reward,
        )
