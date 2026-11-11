"""False AI resolution -> delayed discovery -> optional delayed recontact (max depth 1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from callverse_workforce.config import apply_overrides, load_config
from callverse_workforce.domain.enums import EventType as E
from callverse_workforce.domain.enums import Outcome, RoutingAction
from callverse_workforce.twin import run_simulation
from helpers import by_contact, contact, events_of, fixed, make_config, make_contact, run_scripted

A = RoutingAction
ROOT, RECONTACT = "C00000", "C00000-R1"
ALWAYS_BACK = dict(recontact__probability=1.0, recontact__min_delay_minutes=5.0,
                   recontact__mean_extra_delay_minutes=0.0)


def _false_root(**kw):
    return make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=1.0, ai_service_minutes=2.0,
                        false_discovery_delay_minutes=30.0, request_value=0.5, **kw)


def test_recontact_lineage():  # B1-3
    _, result, decisions = run_scripted(make_config(**ALWAYS_BACK), [_false_root()],
                                        by_contact({ROOT: [A.AI], RECONTACT: [A.HUMAN]}))
    rc = contact(result, RECONTACT)
    assert rc.root_contact_id == ROOT and rc.parent_contact_id == ROOT and rc.recontact_depth == 1
    root = contact(result, ROOT)
    assert root.root_contact_id == ROOT and root.parent_contact_id is None and root.recontact_depth == 0
    assert rc.arrival_time == pytest.approx(2.0 + 30.0 + 5.0)  # AI end + discovery delay + recontact delay
    sched = events_of(result, E.RECONTACT_SCHEDULED, ROOT)
    assert len(sched) == 1 and sched[0].data["recontact_id"] == RECONTACT
    arrival = events_of(result, E.ARRIVAL, RECONTACT)[0]
    assert arrival.data["root_contact_id"] == ROOT and arrival.data["recontact_depth"] == 1
    assert [d.contact_id for d in decisions] == [ROOT, RECONTACT]  # new routing workload


def test_correct_recontact_resolution_earns_normal_reward():  # B1-7
    _, result, _ = run_scripted(make_config(**ALWAYS_BACK), [_false_root()],
                                by_contact({ROOT: [A.AI], RECONTACT: [A.HUMAN]}))
    v = 1.5
    ep = result.episode_reward
    assert contact(result, ROOT).outcome is Outcome.FALSE_AI_DISCOVERED
    assert contact(result, RECONTACT).outcome is Outcome.RESOLVED_HUMAN
    assert ep.false_ai == pytest.approx(-(2 * v + 4))  # root: credit reversed + 4
    assert ep.resolution == pytest.approx(2 * v + 2 * v)  # root's provisional credit + recontact's real one
    assert ep.n_resolutions == 2 and ep.n_false_ai == 1
    rc_reward = events_of(result, E.CONTACT_RESOLVED, RECONTACT)[0].data["reward"]
    assert rc_reward == pytest.approx(2 * v)


def test_false_recontact_is_penalized_but_creates_no_second_recontact():  # B1-4, B1-8
    cfg = make_config(**ALWAYS_BACK, recontact__human_requested_prob=0.0)  # keep AI routable for the recontact
    _, result, _ = run_scripted(cfg, [_false_root()], fixed(A.AI))
    assert contact(result, RECONTACT).outcome is Outcome.FALSE_AI_DISCOVERED
    assert result.episode_reward.n_false_ai == 2
    assert result.episode_reward.false_ai == pytest.approx(-2 * (2 * 1.5 + 4))
    assert not events_of(result, E.RECONTACT_SCHEDULED, RECONTACT)
    assert max(c.recontact_depth for c in result.contacts) == 1
    assert len(result.contacts) == 2


def test_max_depth_above_one_is_rejected_by_config():
    with pytest.raises(ValidationError):
        apply_overrides(load_config(), {"recontact.max_depth": 2})


def test_recontact_disabled_or_declined_creates_nothing():
    _, off, _ = run_scripted(make_config(recontact__enabled=False), [_false_root()], fixed(A.AI))
    assert len(off.contacts) == 1 and not events_of(off, E.RECONTACT_NOT_CREATED)
    _, declined, _ = run_scripted(make_config(recontact__probability=0.0), [_false_root()], fixed(A.AI))
    assert len(declined.contacts) == 1
    assert events_of(declined, E.RECONTACT_NOT_CREATED, ROOT)[0].data["reason"] == "customer_does_not_return"


def test_recontact_after_horizon_is_not_created():
    cfg = make_config(scenario__arrival_window_minutes=60, scenario__drain_minutes=0,
                      recontact__probability=1.0, recontact__min_delay_minutes=40.0,
                      recontact__mean_extra_delay_minutes=0.0)
    _, result, _ = run_scripted(cfg, [_false_root()], fixed(A.AI))  # discovered at 32, would return at 72
    assert len(result.contacts) == 1
    assert events_of(result, E.RECONTACT_NOT_CREATED, ROOT)[0].data["reason"] == "after_horizon"


def test_recontact_latent_attributes_are_policy_independent():  # B1-5
    cfg = make_config(**ALWAYS_BACK, intake_noise__ai_score_noise_std=0.2, intake_noise__deadline_log_sigma=0.3,
                      humans__n_advisors=1)
    others = [make_contact(i, 1.0 + i, human_service_minutes=50.0) for i in (1, 2, 3)]
    plan_a = by_contact({ROOT: [A.AI], "C00001": [A.HUMAN], "C00002": [A.HUMAN], "C00003": [A.HUMAN]})
    plan_b = by_contact({ROOT: [A.AI], "C00001": [A.CALLBACK], "C00002": [A.WAIT, A.AI], "C00003": [A.AI]})
    env_a, res_a, _ = run_scripted(cfg, [_false_root()] + others, plan_a)
    env_b, res_b, _ = run_scripted(cfg, [_false_root()] + others, plan_b)
    assert res_a.total_reward != res_b.total_reward  # the policies really differ
    truth_a = env_a.twin.contacts[RECONTACT].truth
    truth_b = env_b.twin.contacts[RECONTACT].truth
    assert truth_a == truth_b  # identical hidden latent attributes
    assert env_a.twin.contacts[RECONTACT].observed == env_b.twin.contacts[RECONTACT].observed  # identical intake
    root = env_a.twin.contacts[ROOT].truth
    assert truth_a.patience_minutes != root.patience_minutes  # fresh draws, not a copy of the root


def test_recontacts_do_not_shift_exogenous_arrivals_or_other_draws():  # B1-6
    base = load_config()
    on = run_simulation(base, _AlwaysAI(), seed=99)
    off = run_simulation(apply_overrides(base, {"recontact.enabled": False}), _AlwaysAI(), seed=99)
    assert on.metrics["recontacts_created"] > 0 and off.metrics["recontacts_created"] == 0

    def exogenous_arrivals(r):
        return [(e.contact_id, e.time, e.to_dict()["data"]) for e in r.events
                if e.event_type in (E.ARRIVAL, E.INTAKE_PREDICTION) and "-R" not in e.contact_id]

    assert exogenous_arrivals(on) == exogenous_arrivals(off)

    def ai_outcomes(r):
        kinds = (E.AI_RESOLVED, E.AI_VISIBLE_FAILURE, E.AI_FALSE_RESOLUTION)
        return {e.contact_id: e.event_type for e in r.events if e.event_type in kinds and "-R" not in e.contact_id}

    a, b = ai_outcomes(on), ai_outcomes(off)
    common = set(a) & set(b)
    assert len(common) > 500
    assert all(a[c] == b[c] for c in common)  # per-contact AI draws are untouched by recontacts


class _AlwaysAI:
    def reset(self) -> None:
        pass

    def predict(self, observation, action_mask):
        return A.AI if action_mask.is_valid(A.AI) else A.HUMAN
