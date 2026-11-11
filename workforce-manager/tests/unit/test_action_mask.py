"""Action-mask rules: only impossible / business-forbidden actions are removed."""

from __future__ import annotations

import itertools

import pytest

from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import RoutingAction, Urgency
from callverse_workforce.routing.mask import ServiceAvailability, compute_action_mask
from helpers import make_config, make_observation

CFG = make_config()
ALL_UP = ServiceAvailability(ai_available=True, callback_available=True)
A = RoutingAction


def mask_for(availability: ServiceAvailability = ALL_UP, **obs) -> ActionMask:
    return compute_action_mask(make_observation(**obs), availability, CFG.mask, CFG.wait)


def test_default_context_leaves_all_actions_valid():
    assert mask_for().valid_actions == (A.AI, A.HUMAN, A.WAIT, A.CALLBACK)


def test_callback_not_allowed_masks_callback():  # smoke 11
    mask = mask_for(callback_allowed=False)
    assert not mask.is_valid(A.CALLBACK)
    assert mask.reasons["CALLBACK"] == ("callback_not_allowed",)
    assert mask.is_valid(A.AI) and mask.is_valid(A.HUMAN) and mask.is_valid(A.WAIT)


def test_callback_service_unavailable_masks_callback():
    mask = mask_for(ServiceAvailability(ai_available=True, callback_available=False))
    assert not mask.is_valid(A.CALLBACK)
    assert "callback_unavailable" in mask.reasons["CALLBACK"]


def test_human_requested_masks_ai():  # smoke 12
    mask = mask_for(human_requested=True)
    assert not mask.is_valid(A.AI)
    assert mask.reasons["AI"] == ("human_requested",)
    assert mask.is_valid(A.HUMAN)


def test_ai_unavailable_masks_ai():
    mask = mask_for(ServiceAvailability(ai_available=False, callback_available=True))
    assert not mask.is_valid(A.AI)
    assert mask.reasons["AI"] == ("ai_unavailable",)


def test_urgent_short_deadline_masks_wait_and_callback():  # smoke 13
    mask = mask_for(urgency_pred=Urgency.HIGH, deadline_remaining_pred=179.0)
    assert not mask.is_valid(A.WAIT)
    assert not mask.is_valid(A.CALLBACK)
    assert mask.is_valid(A.AI) and mask.is_valid(A.HUMAN)
    assert "urgent_short_deadline" in mask.reasons["WAIT"]


@pytest.mark.parametrize(
    "urgency, remaining",
    [(Urgency.HIGH, 180.0), (Urgency.HIGH, 500.0), (Urgency.MEDIUM, 10.0), (Urgency.LOW, 1.0)],
)
def test_urgent_rule_needs_both_high_urgency_and_short_deadline(urgency, remaining):
    mask = mask_for(urgency_pred=urgency, deadline_remaining_pred=remaining)
    assert mask.is_valid(A.WAIT) and mask.is_valid(A.CALLBACK)


def test_wait_masked_after_max_waits():  # smoke 10 (mask side)
    assert mask_for(routing_attempt_count=2).is_valid(A.WAIT)
    mask = mask_for(routing_attempt_count=3)
    assert not mask.is_valid(A.WAIT)
    assert mask.reasons["WAIT"] == ("max_waits_reached",)


def test_high_ai_load_does_not_mask_ai():  # smoke 14
    mask = mask_for(ai_load=25.0, recent_ai_failure_rate=1.0, ai_resolution_score=0.0)
    assert mask.is_valid(A.AI)


def test_large_human_queue_does_not_mask_human():  # smoke 15
    mask = mask_for(human_queue_size=500, oldest_human_wait=300.0, humans_available=0, human_utilization=1.0)
    assert mask.is_valid(A.HUMAN)


def test_context_preferences_are_never_masked():
    mask = mask_for(waiting_time_so_far=1_000.0, forecast_30min=10_000.0, recent_abandonment_rate=1.0,
                    recent_service_level=0.0, wait_queue_size=300, callbacks_pending=150)
    assert mask.n_valid == 4


def test_mask_always_leaves_a_valid_action():  # smoke 24 (exhaustive over rule inputs)
    combos = itertools.product(
        [True, False],  # human_requested
        [True, False],  # callback_allowed
        list(Urgency),
        [0.0, 179.0, 10_000.0],  # deadline remaining
        [0, 3, 7],  # routing attempts
        [True, False],  # ai available
        [True, False],  # callback available
    )
    for hr, cb_allowed, urg, rem, attempts, ai_up, cb_up in combos:
        mask = mask_for(
            ServiceAvailability(ai_available=ai_up, callback_available=cb_up),
            human_requested=hr,
            callback_allowed=cb_allowed,
            urgency_pred=urg,
            deadline_remaining_pred=rem,
            routing_attempt_count=attempts,
        )
        assert mask.n_valid >= 1
        assert mask.is_valid(A.HUMAN)


def test_action_mask_rejects_all_false():
    with pytest.raises(ValueError):
        ActionMask(valid=(False, False, False, False))
