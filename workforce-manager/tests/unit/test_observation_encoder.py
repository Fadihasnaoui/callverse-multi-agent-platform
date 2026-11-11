"""Canonical NumPy observation encoder (obs_encoder_v1)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from callverse_workforce.domain.enums import Channel, DayType, LanguageDialect, RequestType, Sensitivity, Urgency
from callverse_workforce.encoding import FEATURE_NAMES, OBSERVATION_DIM, ObservationEncoder
from callverse_workforce.twin.environment import RoutingEnvironment
from helpers import make_config, make_contact, make_observation

ENC = ObservationEncoder()

FROZEN_FEATURE_NAMES = (
    "request_type=FAQ", "request_type=BOOKING", "request_type=CHANGE_CANCEL", "request_type=COMPLAINT_INCIDENT",
    "urgency_pred=LOW", "urgency_pred=MEDIUM", "urgency_pred=HIGH",
    "sensitivity_pred=LOW", "sensitivity_pred=MEDIUM", "sensitivity_pred=HIGH",
    "channel=PHONE", "channel=CHAT", "channel=WHATSAPP", "channel=EMAIL",
    "language_dialect=FR", "language_dialect=EN", "language_dialect=AR_MSA", "language_dialect=AR_DIALECT",
    "request_value", "waiting_time_so_far", "deadline_remaining_pred", "callback_allowed", "human_requested",
    "ai_resolution_score", "routing_attempt_count",
    "humans_available", "human_utilization", "human_queue_size", "oldest_human_wait",
    "ai_load", "recent_ai_failure_rate",
    "wait_queue_size", "callbacks_pending", "oldest_callback_age",
    "recent_arrival_rate", "forecast_30min", "recent_abandonment_rate", "recent_service_level",
    "day_type=WEEKDAY", "day_type=WEEKEND", "day_type=HOLIDAY",
    "hour_sin", "hour_cos",
)


def test_fixed_dimension_and_order():  # B1-12
    assert OBSERVATION_DIM == 43 == ENC.dim
    assert FEATURE_NAMES == FROZEN_FEATURE_NAMES
    v = ENC.encode(make_observation())
    assert v.shape == (43,) and v.dtype == np.float32


def test_deterministic_encoding():  # B1-11
    obs = make_observation(waiting_time_so_far=3.3, forecast_30min=41.0)
    a, b = ENC.encode(obs), ObservationEncoder().encode(dataclasses.replace(obs))
    assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize(
    "field, value, name",
    [("request_type", RequestType.COMPLAINT_INCIDENT, "request_type=COMPLAINT_INCIDENT"),
     ("urgency_pred", Urgency.HIGH, "urgency_pred=HIGH"),
     ("sensitivity_pred", Sensitivity.MEDIUM, "sensitivity_pred=MEDIUM"),
     ("channel", Channel.EMAIL, "channel=EMAIL"),
     ("language_dialect", LanguageDialect.AR_DIALECT, "language_dialect=AR_DIALECT"),
     ("day_type", DayType.HOLIDAY, "day_type=HOLIDAY")],
)
def test_categorical_one_hot(field, value, name):
    v = ENC.encode(make_observation(**{field: value}))
    block = [i for i, n in enumerate(FEATURE_NAMES) if n.startswith(field + "=")]
    assert v[block].sum() == 1.0
    assert v[FEATURE_NAMES.index(name)] == 1.0


def test_numeric_scaling_and_saturation_are_finite():
    extreme = make_observation(waiting_time_so_far=1e9, deadline_remaining_pred=-50.0, human_queue_size=10_000,
                               ai_load=99.0, forecast_30min=1e6, routing_attempt_count=7, callbacks_pending=999)
    v = ENC.encode(extreme)
    assert np.isfinite(v).all()
    idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
    assert v[idx["waiting_time_so_far"]] == 1.0 and v[idx["deadline_remaining_pred"]] == 0.0
    assert v[idx["human_queue_size"]] == 1.0 and v[idx["routing_attempt_count"]] == 1.0
    mid = ENC.encode(make_observation(waiting_time_so_far=30.0, ai_load=1.0))
    assert mid[idx["waiting_time_so_far"]] == pytest.approx(0.5) and mid[idx["ai_load"]] == pytest.approx(0.5)


def test_non_finite_input_is_rejected():
    with pytest.raises(ValueError):
        ENC.encode(make_observation(oldest_human_wait=float("nan")))


def test_only_routing_observations_are_accepted():  # B1-13
    with pytest.raises(TypeError):
        ENC.encode(make_contact(0, 0.0))  # a GroundTruthContact
    with pytest.raises(TypeError):
        ENC.encode(make_observation().to_dict())
    forbidden = ("true", "truth", "patience", "scenario", "incident_state", "future", "purpose", "service_minutes")
    assert not any(tok in n for n in FEATURE_NAMES for tok in forbidden)


def _first_vector(contact):
    env = RoutingEnvironment(make_config(), contacts=[contact])
    return ENC.encode(env.reset().observation)


def test_hidden_fields_do_not_change_the_vector():  # B1-13
    base = make_contact(0, 3.0)
    hidden = make_contact(0, 3.0, patience_minutes=0.1, human_service_minutes=77.0, ai_service_minutes=9.0,
                          false_discovery_delay_minutes=1.0, ai_false_resolution_prob=0.0)
    assert _first_vector(base).tobytes() == _first_vector(hidden).tobytes()


def test_full_day_vectors_are_finite_and_bounded():
    from callverse_workforce.config import load_config
    from callverse_workforce.routing import DevelopmentPolicy
    from callverse_workforce.twin import run_simulation

    result = run_simulation(load_config(), DevelopmentPolicy(), seed=8)
    m = ENC.encode_batch([t.state for t in result.transitions])
    assert m.shape == (len(result.transitions), 43) and m.dtype == np.float32
    assert np.isfinite(m).all() and m.min() >= -1.0 and m.max() <= 1.0
    hour = [FEATURE_NAMES.index("hour_sin"), FEATURE_NAMES.index("hour_cos")]
    rest = np.delete(m, hour, axis=1)
    assert rest.min() >= 0.0
