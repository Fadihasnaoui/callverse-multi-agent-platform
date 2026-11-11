"""Observation schema and the ground-truth / observed separation."""

from __future__ import annotations

import inspect
import math
import typing

from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.contact import GroundTruthContact
from callverse_workforce.domain.enums import RoutingAction as A
from callverse_workforce.domain.observation import (
    OBSERVATION_FEATURE_NAMES,
    RoutingObservation,
    build_routing_observation,
)
from callverse_workforce.routing.policy import DevelopmentPolicy, RoutingPolicy
from callverse_workforce.twin.environment import RoutingEnvironment
from helpers import make_config, make_contact, make_observation

EXPECTED_FEATURES = (
    "request_type", "urgency_pred", "sensitivity_pred", "channel", "language_dialect", "request_value",
    "waiting_time_so_far", "deadline_remaining_pred", "callback_allowed", "human_requested",
    "ai_resolution_score", "routing_attempt_count",
    "humans_available", "human_utilization", "human_queue_size", "oldest_human_wait",
    "ai_load", "recent_ai_failure_rate",
    "wait_queue_size", "callbacks_pending", "oldest_callback_age",
    "recent_arrival_rate", "forecast_30min", "recent_abandonment_rate", "recent_service_level",
    "day_type", "hour_sin", "hour_cos",
)


def test_observation_has_exactly_the_28_canonical_features():
    assert OBSERVATION_FEATURE_NAMES == EXPECTED_FEATURES
    assert len(OBSERVATION_FEATURE_NAMES) == 28


def test_observation_field_names_contain_no_hidden_truth():  # smoke 23 (schema)
    forbidden = ("true", "truth", "patience", "future", "scenario", "incident", "ai_correct", "false_",
                 "service_minutes", "deadline_at", "discovery")
    for name in OBSERVATION_FEATURE_NAMES:
        assert not any(token in name for token in forbidden), name


def test_builder_and_policy_never_accept_ground_truth():  # smoke 23 (interfaces)
    hints = typing.get_type_hints(build_routing_observation)
    assert GroundTruthContact not in hints.values()
    assert "truth" not in inspect.signature(build_routing_observation).parameters
    predict_hints = typing.get_type_hints(RoutingPolicy.predict)
    assert predict_hints["observation"] is RoutingObservation


def _first_observation(contact):
    env = RoutingEnvironment(make_config(), contacts=[contact])
    return env.reset().observation


def test_hidden_truth_does_not_change_the_observation():  # smoke 23 (behaviour)
    base = make_contact(0, 5.0)
    altered = make_contact(
        0,
        5.0,
        patience_minutes=0.5,
        human_service_minutes=99.0,
        ai_service_minutes=42.0,
        ai_false_resolution_prob=0.0,  # hidden outcome mix (score depends only on P(correct))
        false_discovery_delay_minutes=1.0,
    )
    assert _first_observation(base) == _first_observation(altered)


def test_noisy_intake_predictions_differ_from_truth():
    env = RoutingEnvironment(make_config(**{
        "intake_noise__deadline_log_sigma": 0.3,
        "intake_noise__ai_score_noise_std": 0.2,
    }))
    dp = env.reset(seed=3)
    truth = env.twin.contacts[dp.contact_id].truth
    true_remaining = truth.deadline_at - dp.timestamp
    assert dp.observation.deadline_remaining_pred != true_remaining
    assert dp.observation.ai_resolution_score != truth.ai_correct_prob


def test_hour_encoding():
    obs = _first_observation(make_contact(0, 0.0))  # 08:00
    assert math.isclose(obs.hour_sin, math.sin(2 * math.pi * 8 / 24))
    assert math.isclose(obs.hour_cos, math.cos(2 * math.pi * 8 / 24))


def test_development_policy_respects_mask_and_is_deterministic():
    obs = make_observation()
    only_human = ActionMask(valid=(False, True, False, False))
    p = DevelopmentPolicy()
    assert [p.predict(obs, only_human) for _ in range(4)] == [A.HUMAN] * 4
    full = ActionMask(valid=(True, True, True, True))
    p.reset()
    first = [p.predict(obs, full) for _ in range(4)]
    p.reset()
    assert first == [p.predict(obs, full) for _ in range(4)] == [A.AI, A.HUMAN, A.WAIT, A.CALLBACK]
