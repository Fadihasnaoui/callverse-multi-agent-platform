"""B1.5 callback realism: attempts, completion window, dropout, reasons and frozen reward mapping."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from callverse_workforce.config import apply_overrides, load_config
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.domain.enums import CallbackReason as R
from callverse_workforce.domain.enums import EventType as E
from callverse_workforce.domain.enums import Outcome, RoutingAction
from callverse_workforce.routing import DevelopmentPolicy
from callverse_workforce.twin import run_simulation
from callverse_workforce.utils.rng import RngManager, Stream
from helpers import by_contact, contact, events_of, fixed, make_config, make_contact, run_scripted

A = RoutingAction
C0, C1 = "C00000", "C00001"
SEED = 42
NO_DROPOUT = {"callback__b1_5__dropout_probability_per_hour": 0.0}


def _busy_advisor(minutes=10_000.0):
    """One advisor occupied by C0 so that callbacks stay pending."""
    return make_contact(0, 0.0, human_service_minutes=minutes, request_value=0.0)


def _dropout_delay(index, prob_per_hour, seed=SEED, depth=0):
    lam = -math.log(1 - prob_per_hour) / 60.0
    return float(RngManager(seed).contact_stream(Stream.CALLBACK_DROPOUT, index, depth).exponential(1 / lam))


def test_b1_and_b1_5_callback_models_selectable():  # B1.5-6, B1.5-7
    cfg = load_config()
    assert cfg.callback.model == "b1_5"
    assert (cfg.callback.answer_probability, cfg.callback.max_attempts, cfg.callback.max_delay_minutes) == (0.65, 2, 120)
    assert cfg.callback.dropout_hazard_per_minute == pytest.approx(-math.log(0.95) / 60)
    b1 = apply_overrides(cfg, {"callback.model": "b1"})
    assert (b1.callback.answer_probability, b1.callback.max_attempts) == (0.85, 3)
    assert b1.callback.max_delay_minutes is None and b1.callback.dropout_hazard_per_minute == 0.0


def test_b1_model_never_uses_window_or_dropout():
    cfg = apply_overrides(load_config(), {"callback.model": "b1"})
    m = run_simulation(cfg, DevelopmentPolicy(), seed=1001).metrics["callback"]
    assert m["reasons"]["WINDOW_EXPIRED"] == 0 and m["reasons"]["CUSTOMER_DROPOUT"] == 0
    assert m["scheduled"] > 100


def test_max_attempts_enforced_and_charged_exactly_minus_one():  # B1.5-8, B1.5-13
    cfg = make_config(callback__b1_5__answer_probability=0.0, callback__retry_delay_minutes=5.0, **NO_DROPOUT)
    _, result, _ = run_scripted(cfg, [make_contact(0, 0.0)], fixed(A.CALLBACK))
    s = contact(result, C0)
    assert s.callback_attempts == 2 and len(s.callback_attempt_times) == 2
    assert s.callback_attempt_times == pytest.approx((0.0, 1.0 + 5.0))
    assert s.callback_reason is R.MAX_ATTEMPTS and s.outcome is Outcome.CALLBACK_FAILED
    ep = result.episode_reward
    assert ep.callback_failure == -1.0 and ep.n_callback_failures == 1
    assert ep.abandonment == 0.0 and ep.unfinished == 0.0 and ep.resolution == 0.0
    assert len(events_of(result, E.CALLBACK_FAILED, C0)) == 1


def test_window_expired_enforced_once_with_minus_one_and_pending_cost():  # B1.5-9, 14, 18
    cfg = make_config(humans__n_advisors=1, **NO_DROPOUT)
    _, result, _ = run_scripted(cfg, [_busy_advisor(), make_contact(1, 10.0, request_value=0.5)],
                                by_contact({C0: [A.HUMAN], C1: [A.CALLBACK]}))
    s = contact(result, C1)
    assert s.callback_reason is R.WINDOW_EXPIRED and s.outcome is Outcome.CALLBACK_WINDOW_EXPIRED
    expired = events_of(result, E.CALLBACK_WINDOW_EXPIRED, C1)
    assert len(expired) == 1 and expired[0].time == pytest.approx(10.0 + 120.0)
    assert expired[0].data["reward"] == -1.0
    assert s.callback_pending_minutes == pytest.approx(120.0) and s.callback_attempts == 0
    ep = result.episode_reward
    assert ep.callback_failure == -1.0 and ep.abandonment == 0.0
    assert ep.callback_pending == pytest.approx(-0.02 * 120.0)  # accrued before the terminal event
    assert not s.human_touched


def test_customer_dropout_minus_three_v_only():  # B1.5-10, 11, 12
    p = 0.9  # high hazard so the dropout happens well inside the window
    cfg = make_config(humans__n_advisors=1, callback__b1_5__dropout_probability_per_hour=p)
    _, result, _ = run_scripted(cfg, [_busy_advisor(), make_contact(1, 10.0, request_value=0.5)],
                                by_contact({C0: [A.HUMAN], C1: [A.CALLBACK]}), seed=SEED)
    delay = _dropout_delay(1, p)
    assert delay < 120.0
    s = contact(result, C1)
    assert s.callback_reason is R.CUSTOMER_DROPOUT and s.outcome is Outcome.CALLBACK_DROPOUT
    drop = events_of(result, E.CALLBACK_DROPOUT, C1)
    assert len(drop) == 1 and drop[0].time == pytest.approx(10.0 + delay)
    ep = result.episode_reward
    assert ep.abandonment == pytest.approx(-3 * 1.5)
    assert ep.callback_failure == 0.0 and ep.n_callback_failures == 0
    assert ep.callback_pending == pytest.approx(-0.02 * delay)
    assert not events_of(result, E.CALLBACK_WINDOW_EXPIRED) and not events_of(result, E.CALLBACK_FAILED)


def test_dropout_respects_elapsed_time_at_irregular_creation_times():  # B1.5-19
    p = 0.9
    cfg = make_config(humans__n_advisors=1, callback__b1_5__dropout_probability_per_hour=p,
                      callback__b1_5__max_delay_minutes=10_000.0)
    for created in (3.7, 41.13, 250.0):
        contacts = [_busy_advisor(), make_contact(5, created)]
        _, result, _ = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], "C00005": [A.CALLBACK]}))
        expected = created + _dropout_delay(5, p)
        drop = events_of(result, E.CALLBACK_DROPOUT, "C00005")
        if expected < result.horizon_minutes:
            assert len(drop) == 1 and drop[0].time == pytest.approx(expected)
            assert contact(result, "C00005").callback_pending_minutes == pytest.approx(expected - created)
        else:
            assert not drop


def test_dropout_hazard_matches_five_percent_per_hour():  # B1.5-19 (statistical)
    n = 600
    cfg = make_config(humans__n_advisors=1, callback__max_pending=10_000, callback__b1_5__max_delay_minutes=10_000.0)
    contacts = [_busy_advisor()] + [make_contact(i, 0.0) for i in range(1, n + 1)]
    _, result, _ = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN]}, default=A.CALLBACK))
    times = np.array([e.time for e in events_of(result, E.CALLBACK_DROPOUT)])
    p60, p780 = np.mean(times <= 60.0) * len(times) / n, len(times) / n
    assert p60 == pytest.approx(0.05, abs=0.025)
    assert p780 == pytest.approx(1 - 0.95 ** 13, abs=0.06)


def test_dropout_rng_reproducible_and_isolated():  # B1.5-20, 21
    cfg = load_config()
    a = run_simulation(cfg, DevelopmentPolicy(), seed=1003)
    b = run_simulation(cfg, DevelopmentPolicy(), seed=1003)
    drops = lambda r: [(e.contact_id, e.time) for e in r.events if e.event_type is E.CALLBACK_DROPOUT]
    assert drops(a) == drops(b)
    b1 = run_simulation(apply_overrides(cfg, {"callback.model": "b1"}), DevelopmentPolicy(), seed=1003)
    exo = lambda r: [(e.contact_id, e.time, json.dumps(e.to_dict()["data"], sort_keys=True)) for e in r.events
                     if e.event_type in (E.ARRIVAL, E.INTAKE_PREDICTION) and "-R" not in e.contact_id]
    assert exo(a) == exo(b1)  # callback model never shifts exogenous arrivals / intake
    kinds = (E.AI_RESOLVED, E.AI_VISIBLE_FAILURE, E.AI_FALSE_RESOLUTION)
    ai = lambda r: {e.contact_id: e.event_type for e in r.events if e.event_type in kinds}
    common = set(ai(a)) & set(ai(b1))
    assert len(common) > 200 and all(ai(a)[c] == ai(b1)[c] for c in common)


def test_pending_callback_at_closing_is_unfinished_not_window_expired():  # B1.5-15, 16, 17
    # Spec example scaled: horizon 60 (=20:00), callback created at 30 (=19:30), window 120 -> would end at 150.
    cfg = make_config(humans__n_advisors=1, scenario__arrival_window_minutes=60, scenario__drain_minutes=0,
                      **NO_DROPOUT)
    _, result, _ = run_scripted(cfg, [_busy_advisor(), make_contact(1, 30.0, request_value=0.5)],
                                by_contact({C0: [A.HUMAN], C1: [A.CALLBACK]}))
    s = contact(result, C1)
    assert s.outcome is Outcome.UNFINISHED and s.callback_reason is R.UNFINISHED
    assert not events_of(result, E.CALLBACK_WINDOW_EXPIRED)
    unfinished = events_of(result, E.UNFINISHED_AT_HORIZON, C1)
    assert len(unfinished) == 1 and unfinished[0].data["reward"] == pytest.approx(-2 * 1.5)
    assert s.callback_pending_minutes == pytest.approx(30.0)
    ep = result.episode_reward
    assert ep.callback_failure == 0.0 and ep.abandonment == 0.0
    assert ep.unfinished == pytest.approx(-2 * 1.5 - 2 * 1.0)  # C1 plus the in-service C0, once each


def test_window_expiring_exactly_at_closing_is_unfinished():
    cfg = make_config(humans__n_advisors=1, scenario__arrival_window_minutes=130, scenario__drain_minutes=0,
                      **NO_DROPOUT)
    _, result, _ = run_scripted(cfg, [_busy_advisor(), make_contact(1, 10.0)],
                                by_contact({C0: [A.HUMAN], C1: [A.CALLBACK]}))
    assert contact(result, C1).callback_reason is R.UNFINISHED  # window end == horizon: not "during operating time"


def test_callback_success_counts_in_hir_but_unanswered_dials_do_not():  # B1.5-26, 27
    ok = make_config(callback__b1_5__answer_probability=1.0, **NO_DROPOUT)
    _, r_ok, _ = run_scripted(ok, [make_contact(0, 0.0)], fixed(A.CALLBACK))
    assert contact(r_ok, C0).callback_reason is R.CALLBACK_SUCCESS and contact(r_ok, C0).human_touched
    assert r_ok.metrics["human_intervention_rate"]["instance_level"] == 1.0
    no = make_config(callback__b1_5__answer_probability=0.0, **NO_DROPOUT)
    _, r_no, _ = run_scripted(no, [make_contact(0, 0.0)], fixed(A.CALLBACK))
    s = contact(r_no, C0)
    assert s.callback_human_minutes > 0 and not s.human_touched  # advisor dialled, nobody was served
    assert r_no.metrics["human_intervention_rate"]["instance_level"] == 0.0


def test_human_routing_rate_is_separate_from_hir():  # B1.5-23
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=100.0), make_contact(1, 0.0, patience_minutes=2.0)]
    _, routed, _ = run_scripted(cfg, contacts, fixed(A.HUMAN))
    assert routed.metrics["human_routing_rate"] == 1.0
    assert routed.metrics["human_intervention_rate"]["instance_level"] == 0.5  # C1 abandoned before service
    esc = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=0.0)
    _, escalated, _ = run_scripted(make_config(), [esc], fixed(A.AI))
    assert escalated.metrics["human_routing_rate"] == 0.0
    assert escalated.metrics["human_intervention_rate"]["instance_level"] == 1.0


@pytest.mark.parametrize("name, advisors, minutes", [("normal_day.yaml", 8, 780), ("load_check.yaml", 4, 720)])
def test_advisor_capacity_calculation(name, advisors, minutes):  # B1.5-22
    cfg = load_config(PROJECT_ROOT / "configs" / name)
    cap = run_simulation(cfg, DevelopmentPolicy(), seed=1001).metrics["capacity"]
    assert cap["n_advisors"] == advisors and cap["operating_minutes"] == minutes
    assert cap["capacity_minutes"] == advisors * minutes
    assert cap["utilization"] == pytest.approx(cap["busy_minutes"] / cap["capacity_minutes"])
    assert 0.0 < cap["utilization"] <= 1.0 + 1e-9


def test_callback_reasons_reconcile_with_outcomes():  # B1.5-30
    for model in ("b1", "b1_5"):
        cfg = apply_overrides(load_config(PROJECT_ROOT / "configs" / "load_check.yaml"), {"callback.model": model})
        r = run_simulation(cfg, DevelopmentPolicy(), seed=1002)
        rec = r.metrics["reconciliation"]
        assert rec["balanced"] and rec["sum_terminal"] == len(r.contacts)
        cb = [c for c in r.contacts if c.callback_created_at is not None]
        assert all(c.callback_reason is not None for c in cb)
        expected = {R.CALLBACK_SUCCESS: Outcome.RESOLVED_CALLBACK, R.CUSTOMER_DROPOUT: Outcome.CALLBACK_DROPOUT,
                    R.MAX_ATTEMPTS: Outcome.CALLBACK_FAILED, R.WINDOW_EXPIRED: Outcome.CALLBACK_WINDOW_EXPIRED,
                    R.MISSED_DEADLINE: Outcome.MISSED_DEADLINE, R.UNFINISHED: Outcome.UNFINISHED}
        assert all(c.outcome is expected[c.callback_reason] for c in cb)
        ep = r.episode_reward
        n_fail = sum(c.callback_reason in (R.MAX_ATTEMPTS, R.WINDOW_EXPIRED) for c in cb)
        assert ep.n_callback_failures == n_fail and ep.callback_failure == pytest.approx(-n_fail)
