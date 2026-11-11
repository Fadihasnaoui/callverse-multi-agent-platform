"""Structured records produced by a simulation run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import CallbackReason, Channel, LanguageDialect, Outcome, RequestType, RoutingAction
from callverse_workforce.domain.observation import RoutingObservation
from callverse_workforce.reward.breakdown import RewardBreakdown


@dataclass(frozen=True)
class DecisionPoint:
    """A routing decision the environment is waiting for."""

    decision_id: str
    contact_id: str
    timestamp: float
    observation: RoutingObservation
    action_mask: ActionMask


@dataclass(frozen=True)
class RoutingDecision:
    """A routing decision that was taken."""

    decision_id: str
    contact_id: str
    timestamp: float
    action: RoutingAction
    action_mask: ActionMask
    n_valid_actions: int
    routing_attempt_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "contact_id": self.contact_id,
            "timestamp": self.timestamp,
            "action": self.action.value,
            "action_mask": self.action_mask.to_dict(),
            "n_valid_actions": self.n_valid_actions,
            "routing_attempt_count": self.routing_attempt_count,
        }


@dataclass(frozen=True)
class Transition:
    """One SMDP transition between two consecutive routing decisions.

    ``reward`` covers everything that happened in the center during
    ``[timestamp, timestamp + delta_t_minutes)``. The future learner discounts it
    with ``gamma_per_minute ** delta_t_minutes``.
    """

    decision_id: str
    contact_id: str
    timestamp: float
    state: RoutingObservation
    action: RoutingAction
    action_mask: ActionMask
    reward: float
    reward_breakdown: RewardBreakdown
    next_state: RoutingObservation | None
    next_action_mask: ActionMask | None
    delta_t_minutes: float
    done: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "contact_id": self.contact_id,
            "timestamp": self.timestamp,
            "state": self.state.to_dict(),
            "action": self.action.value,
            "action_mask": self.action_mask.to_dict(),
            "reward": self.reward,
            "reward_breakdown": self.reward_breakdown.to_dict(),
            "next_state": self.next_state.to_dict() if self.next_state else None,
            "next_action_mask": self.next_action_mask.to_dict() if self.next_action_mask else None,
            "delta_t_minutes": self.delta_t_minutes,
            "done": self.done,
        }


@dataclass(frozen=True)
class ContactSummary:
    """Final per-contact record (simulator view, includes truth)."""

    contact_id: str
    root_contact_id: str
    parent_contact_id: str | None
    recontact_depth: int
    request_type: RequestType
    channel: Channel
    language_dialect: LanguageDialect
    arrival_time: float
    value_weight: float
    outcome: Outcome
    terminal_time: float
    online_wait_minutes: float
    callback_pending_minutes: float
    n_decisions: int
    n_waits: int
    actions: tuple[RoutingAction, ...]
    escalated: bool
    callback_attempts: int
    callback_created_at: float | None
    callback_attempt_times: tuple[float, ...]
    callback_completed_at: float | None
    callback_reason: CallbackReason | None
    human_touched: bool
    human_queue_minutes: float
    human_direct_minutes: float
    callback_human_minutes: float
