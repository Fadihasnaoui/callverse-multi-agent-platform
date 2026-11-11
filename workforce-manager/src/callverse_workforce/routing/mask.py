"""Minimal action mask.

The mask removes ONLY physically impossible or explicitly business-forbidden
actions. Context such as queue length, AI load, a low AI score or a high
forecast is deliberately NOT masked: those are preferences to be learned.

HUMAN is never masked in V1 (the human queue always accepts), which guarantees
at least one valid action.
"""

from __future__ import annotations

from dataclasses import dataclass

from callverse_workforce.config.models import MaskConfig, WaitConfig
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import ACTION_ORDER, RoutingAction, Urgency
from callverse_workforce.domain.observation import RoutingObservation


@dataclass(frozen=True)
class ServiceAvailability:
    """Physical availability of services at the decision instant (observable)."""

    ai_available: bool
    callback_available: bool


def compute_action_mask(
    observation: RoutingObservation,
    availability: ServiceAvailability,
    mask_config: MaskConfig,
    wait_config: WaitConfig,
) -> ActionMask:
    reasons: dict[RoutingAction, list[str]] = {a: [] for a in ACTION_ORDER}

    if not availability.ai_available:
        reasons[RoutingAction.AI].append("ai_unavailable")
    if observation.human_requested:
        reasons[RoutingAction.AI].append("human_requested")

    if not observation.callback_allowed:
        reasons[RoutingAction.CALLBACK].append("callback_not_allowed")
    if not availability.callback_available:
        reasons[RoutingAction.CALLBACK].append("callback_unavailable")

    # SYNTHETIC_ASSUMPTION_V1: urgent contacts with < 3h predicted deadline cannot be deferred.
    if (
        observation.urgency_pred is Urgency.HIGH
        and observation.deadline_remaining_pred < mask_config.urgent_deadline_threshold_minutes
    ):
        reasons[RoutingAction.WAIT].append("urgent_short_deadline")
        reasons[RoutingAction.CALLBACK].append("urgent_short_deadline")

    # In V1 only WAIT sends a contact back to routing, so routing_attempt_count == WAITs used.
    if observation.routing_attempt_count >= wait_config.max_waits:
        reasons[RoutingAction.WAIT].append("max_waits_reached")

    valid = tuple(not reasons[a] for a in ACTION_ORDER)
    return ActionMask(
        valid=valid,  # type: ignore[arg-type]
        reasons={a.value: tuple(r) for a, r in reasons.items() if r},
    )
