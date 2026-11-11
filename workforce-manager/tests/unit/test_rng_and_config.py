"""RNG stream independence, generation determinism and configuration validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from callverse_workforce.config import apply_overrides, load_config
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.utils.rng import RngManager, Stream


def test_same_seed_same_stream():
    a, b = RngManager(11).stream(Stream.ARRIVALS), RngManager(11).stream(Stream.ARRIVALS)
    assert (a.random(5) == b.random(5)).all()


def test_streams_are_distinct():
    rng = RngManager(11)
    assert rng.stream(Stream.ARRIVALS).random() != rng.stream(Stream.PATIENCE).random()
    assert rng.contact_stream(Stream.AI_SUCCESS, 0).random() != rng.contact_stream(Stream.AI_SUCCESS, 1).random()
    assert (
        rng.contact_stream(Stream.FALSE_AI_RESOLUTION, 0).random()
        != rng.contact_stream(Stream.FALSE_AI_DISCOVERY, 0).random()
    )


def test_contact_stream_is_independent_of_draw_order():
    rng = RngManager(5)
    first = rng.contact_stream(Stream.SERVICE_TIMES, 7).random()
    for i in range(20):  # draws for other contacts happen in between
        rng.contact_stream(Stream.SERVICE_TIMES, i).random(3)
    assert rng.contact_stream(Stream.SERVICE_TIMES, 7).random() == first


def test_generation_is_deterministic_and_seed_sensitive():
    cfg = load_config()
    day1 = ContactGenerator(cfg, RngManager(1)).generate_day()
    again = ContactGenerator(cfg, RngManager(1)).generate_day()
    day2 = ContactGenerator(cfg, RngManager(2)).generate_day()
    assert day1 == again
    assert [c.arrival_time for c in day1] != [c.arrival_time for c in day2]
    assert len(day1) > 500  # ~850 expected for the normal day profile


def test_generated_values_respect_config_bounds():
    cfg = load_config()
    for c in ContactGenerator(cfg, RngManager(3)).generate_day():
        assert 0 <= c.arrival_time < cfg.scenario.arrival_window_minutes
        assert 0.0 <= c.request_value <= 1.0
        rng = cfg.deadlines[c.urgency]
        assert rng.min <= c.deadline_offset_minutes <= rng.max
        assert c.ai_correct_prob + c.ai_false_resolution_prob <= 1.0 + 1e-9
        assert c.patience_minutes > 0 and c.human_service_minutes > 0


def test_default_config_loads():
    cfg = load_config()
    assert cfg.wait.reevaluation_minutes == 2.0
    assert cfg.wait.max_waits == 3
    assert cfg.mask.urgent_deadline_threshold_minutes == 180
    assert cfg.smdp.gamma_per_minute == 0.995
    assert cfg.scenario.horizon_minutes == 780


def test_frozen_reward_coefficients():
    r = load_config().reward
    assert (r.resolution_multiplier, r.human_busy_per_minute, r.ai_busy_per_minute) == (2.0, 0.08, 0.01)
    assert (r.online_waiting_per_minute, r.callback_pending_per_minute) == (0.05, 0.02)
    assert (r.abandonment_multiplier, r.missed_deadline_multiplier, r.false_ai_extra_penalty) == (3.0, 4.0, 4.0)
    assert (r.callback_failure_penalty, r.unfinished_multiplier) == (1.0, 2.0)


def test_invalid_config_is_rejected():
    cfg = load_config()
    with pytest.raises(ValidationError):
        apply_overrides(cfg, {"request_types.FAQ.share": 0.9})  # shares no longer sum to 1
    with pytest.raises(ValidationError):
        apply_overrides(cfg, {"request_types.FAQ.ai_correct_prob": 0.99})  # correct + false > 1
    with pytest.raises(ValidationError):
        apply_overrides(cfg, {"humans.n_advisors": 0})


def test_overrides_produce_new_validated_config():
    cfg = load_config()
    new = apply_overrides(cfg, {"humans.n_advisors": 3, "request_types.FAQ.patience_mean_minutes": 9.0})
    assert new.humans.n_advisors == 3
    assert new.request_types["FAQ"].patience_mean_minutes == 9.0
    assert cfg.humans.n_advisors == 8  # original untouched
