"""Frozen Human Intervention Rate semantics and terminal-outcome reconciliation."""

from __future__ import annotations

import dataclasses

import pytest

from callverse_workforce.config import load_config
from callverse_workforce.domain.enums import EventType as E
from callverse_workforce.domain.enums import Outcome, RoutingAction
from callverse_workforce.routing import DevelopmentPolicy
from callverse_workforce.tracing.summary import human_intervention_rates
from callverse_workforce.twin import run_simulation
from callverse_workforce.utils.rng import RngManager, Stream
from helpers import by_contact, contact, events_of, fixed, make_config, make_contact, run_scripted

A = RoutingAction


def test_ai_then_human_escalation_counts_once():  # B1-14
    c = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=0.0)
    _, result, _ = run_scripted(make_config(), [c], fixed(A.AI))
    hir = result.metrics["human_intervention_rate"]
    assert contact(result, "C00000").human_touched
    assert hir["touched_instances"] == 1 and hir["instance_level"] == 1.0 and hir["root_level"] == 1.0


def test_ai_success_is_not_a_human_touch():
    _, result, _ = run_scripted(make_config(), [make_contact(0, 0.0)], fixed(A.AI))
    assert result.metrics["human_intervention_rate"]["instance_level"] == 0.0


def test_human_routing_without_service_is_not_a_touch():
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=100.0), make_contact(1, 0.0, patience_minutes=2.0)]
    _, result, _ = run_scripted(cfg, contacts, fixed(A.HUMAN))
    assert contact(result, "C00001").outcome is Outcome.ABANDONED
    assert not contact(result, "C00001").human_touched
    assert result.metrics["human_intervention_rate"]["instance_level"] == 0.5


def _index_with_first_dial_failing(seed: int, p: float) -> int:
    for i in range(1000):
        draws = RngManager(seed).contact_stream(Stream.CALLBACK, i).random(2)
        if draws[0] >= p and draws[1] < p:
            return i
    raise AssertionError("no suitable index")


def test_human_callback_counts_and_repeated_touches_do_not_double_count():  # B1-16
    p = 0.5
    idx = _index_with_first_dial_failing(42, p)
    cfg = make_config(callback__model="b1", callback__b1__reach_probability=p, callback__retry_delay_minutes=5.0)
    _, result, _ = run_scripted(cfg, [make_contact(idx, 0.0)], fixed(A.CALLBACK), seed=42)
    s = contact(result, f"C{idx:05d}")
    assert s.outcome is Outcome.RESOLVED_CALLBACK and s.callback_attempts == 2  # two advisor dials
    assert s.human_touched
    hir = result.metrics["human_intervention_rate"]
    assert hir["touched_instances"] == 1 and hir["total_instances"] == 1


def test_root_vs_recontact_semantics():  # B1-15
    cfg = make_config(recontact__probability=1.0, recontact__mean_extra_delay_minutes=0.0)
    root = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=1.0, false_discovery_delay_minutes=10)
    _, result, _ = run_scripted(cfg, [root], by_contact({"C00000": [A.AI], "C00000-R1": [A.HUMAN]}))
    hir = result.metrics["human_intervention_rate"]
    assert hir["total_instances"] == 2 and hir["touched_instances"] == 1
    assert hir["instance_level"] == pytest.approx(0.5)  # root instance untouched, recontact touched
    assert hir["total_roots"] == 1 and hir["touched_roots"] == 1 and hir["root_level"] == 1.0


def test_root_level_counts_a_root_once_even_if_several_instances_touched():
    _, result, _ = run_scripted(make_config(), [make_contact(0, 0.0), make_contact(1, 0.0)], fixed(A.HUMAN))
    a, b = result.contacts
    lineage = [a, dataclasses.replace(b, root_contact_id=a.contact_id, parent_contact_id=a.contact_id,
                                      recontact_depth=1)]
    hir = human_intervention_rates(lineage)
    assert hir["instance_level"] == 1.0 and hir["touched_roots"] == 1 and hir["total_roots"] == 1


def test_hir_is_distinct_from_human_busy_minutes():
    _, result, _ = run_scripted(make_config(), [make_contact(0, 0.0, human_service_minutes=7.0)], fixed(A.HUMAN))
    assert result.metrics["human_intervention_rate"]["instance_level"] == 1.0
    assert result.metrics["human_busy_minutes"] == pytest.approx(7.0)


@pytest.mark.parametrize("config_name", ["normal_day.yaml", "load_check.yaml"])
def test_terminal_outcome_reconciliation_is_exact(config_name):  # B1-17
    from callverse_workforce.config.loader import PROJECT_ROOT

    cfg = load_config(PROJECT_ROOT / "configs" / config_name)
    result = run_simulation(cfg, DevelopmentPolicy(), seed=5)
    rec = result.metrics["reconciliation"]
    assert rec["balanced"]
    assert rec["sum_terminal"] == rec["instances_created"] == len(result.contacts) == rec["arrival_events"]
    assert rec["instances_created"] == rec["exogenous_instances"] + rec["recontact_instances"]
    assert rec["recontact_instances"] == result.metrics["recontacts_created"] > 0
    assert set(rec["terminal_by_outcome"]) == {o.value for o in Outcome}
    # per-contact human minutes reconcile with the center-level exposure
    total = sum(c.human_direct_minutes + c.callback_human_minutes for c in result.contacts)
    assert total == pytest.approx(result.episode_reward.human_busy_minutes)
    assert len(events_of(result, E.ARRIVAL)) == len(result.contacts)
