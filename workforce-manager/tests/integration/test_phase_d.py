"""Phase D: Twin V1.1 mechanisms, predefined scenarios, protocol guards and diagnostics."""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pytest

from callverse_workforce.baselines.policies import AIFirstPolicy, make_policy
from callverse_workforce.config import apply_overrides, load_config
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.domain.enums import EventType, Outcome, RoutingAction
from callverse_workforce.domain.observation import OBSERVATION_FEATURE_NAMES
from callverse_workforce.headroom import experiment as H
from callverse_workforce.phase_d import experiment as D
from callverse_workforce.phase_d.grid import (
    GRID,
    HANDOFF,
    HANGUP,
    MAX_CELL,
    MID_CELL,
    MULTIPLIER,
    RATIONALE_PATH,
    V1_CELL,
    Cell,
    cell_config,
    grid_manifest,
    rationale_hash,
)
from callverse_workforce.phase_d.scenarios import SCENARIOS, event_windows, scenario_config
from callverse_workforce.seeds import (
    FINAL_TEST_RESERVED,
    PHASE_D_AUDIT_SEEDS,
    PHASE_D_HEADROOM_SEEDS,
    PHASE_D_VALIDATION_SEEDS,
    SEED_SETS,
    SeedPolicyError,
)
from callverse_workforce.twin import run_simulation
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.utils.rng import RngManager
from helpers import by_contact, contact, events_of, fixed, make_config, make_contact, run_scripted

A = RoutingAction
BASE = load_config()
V1 = H.load_frozen()
OUT = PROJECT_ROOT / "artifacts" / "v1_1"
C0, C1 = "C00000", "C00001"
FAIL = dict(ai_correct_prob=0.0, ai_false_resolution_prob=0.0, ai_service_minutes=2.0)


def _events(r):
    return [e.to_dict() for e in r.events]


# --------------------------------------------------------------------------- V1 preservation
def test_v1_1_mechanisms_off_reproduce_v1_exactly():  # D-1, D-6, D-7
    explicit = apply_overrides(BASE, {"v1_1.handoff_overhead_minutes": 0.0, "v1_1.frustration_patience_multiplier": 1.0,
                                      "v1_1.immediate_hangup_prob_after_ai_failure": 0.0})
    assert cell_config(BASE, V1_CELL) is BASE and explicit.v1_1.is_v1
    for name, params in (("RulePolicy", V1["RulePolicy"]), ("AIFirstPolicy", {}), ("RandomPolicy", {})):
        a = run_simulation(BASE, make_policy(name, params, 1401), seed=1401)
        b = run_simulation(explicit, make_policy(name, params, 1401), seed=1401)
        assert _events(a) == _events(b)
        assert [t.to_dict() for t in a.transitions] == [t.to_dict() for t in b.transitions]
        assert json.dumps(a.metrics, sort_keys=True, default=str) == json.dumps(b.metrics, sort_keys=True, default=str)


def test_frozen_b2_results_still_reproduce():  # D-20 (B2 part)
    path = PROJECT_ROOT / "artifacts" / "baselines" / "eval_normal.json"
    if not path.exists():
        pytest.skip("B2 artifacts missing")
    ev = json.loads(path.read_text(encoding="utf-8"))
    day = ev["policies"]["RulePolicy"]["days"][0]
    r = run_simulation(BASE, make_policy("RulePolicy", V1["RulePolicy"], day["seed"]), seed=day["seed"])
    assert r.total_reward == day["total_reward"]


def test_phase_c_branch_replay_still_reproduces():  # D-20 (headroom part)
    branches = PROJECT_ROOT / "artifacts" / "headroom" / "primary" / "normal" / "branches.jsonl"
    states = PROJECT_ROOT / "artifacts" / "headroom" / "sampled_states" / "normal.json"
    if not branches.exists():
        pytest.skip("Phase C artifacts missing")
    st = json.loads(states.read_text(encoding="utf-8"))["states"][3]
    old = {}
    with open(branches, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["state_id"] == st["state_id"]:
                old[(r["action"], r["variant"])] = r["discounted_return"]
    from callverse_workforce.headroom.branch import run_branch

    jobs = H.make_jobs(BASE, [st], V1["RulePolicy"], ("RulePolicy", V1["RulePolicy"]))[:3]
    assert all(run_branch(j)["discounted_return"] == old[(j.action, j.variant)] for j in jobs)


# --------------------------------------------------------------------------- mechanism A: handoff
def test_handoff_only_after_visible_failure_escalation():  # D-2, D-3, D-4
    cfg = make_config(v1_1__handoff_overhead_minutes=2.0)
    contacts = [make_contact(0, 0.0, human_service_minutes=5.0, **FAIL), make_contact(1, 1.0, human_service_minutes=5.0),
                make_contact(2, 2.0, human_service_minutes=5.0)]  # C2: successful AI
    _, r, _ = run_scripted(cfg, contacts, by_contact({C0: [A.AI], C1: [A.HUMAN], "C00002": [A.AI]}))
    esc, direct = contact(r, C0), contact(r, C1)
    assert esc.escalated and esc.human_direct_minutes == pytest.approx(5.0 + 2.0)
    assert esc.terminal_time == pytest.approx(2.0 + 7.0)
    assert direct.human_direct_minutes == pytest.approx(5.0)
    starts = {e.contact_id: e.data for e in events_of(r, EventType.HUMAN_SERVICE_STARTED)}
    assert starts[C0]["handoff_minutes"] == 2.0 and "handoff_minutes" not in starts[C1]
    assert contact(r, "C00002").outcome is Outcome.RESOLVED_AI
    assert r.episode_reward.human_busy_minutes == pytest.approx(7.0 + 5.0)


def test_no_handoff_for_false_ai_or_its_recontact_routed_human():
    cfg = make_config(v1_1__handoff_overhead_minutes=4.0, recontact__probability=1.0,
                      recontact__mean_extra_delay_minutes=0.0, recontact__human_service_multiplier=1.0)
    root = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=1.0, false_discovery_delay_minutes=10.0,
                        human_service_minutes=5.0)
    _, r, _ = run_scripted(cfg, [root], by_contact({C0: [A.AI], "C00000-R1": [A.HUMAN]}))
    assert not [e for e in events_of(r, EventType.HUMAN_SERVICE_STARTED) if "handoff_minutes" in e.data]


def test_no_handoff_for_callbacks():
    cfg = make_config(v1_1__handoff_overhead_minutes=4.0, callback__b1_5__answer_probability=1.0,
                      callback__b1_5__dropout_probability_per_hour=0.0)
    _, r, _ = run_scripted(cfg, [make_contact(0, 0.0, human_service_minutes=5.0)], fixed(A.CALLBACK))
    assert contact(r, C0).callback_human_minutes == pytest.approx(1.0 + 5.0)


# --------------------------------------------------------------------------- mechanism B: frustration
def test_frustration_scales_remaining_patience_only():  # D-5
    cfg = make_config(humans__n_advisors=1, v1_1__frustration_patience_multiplier=0.5)
    contacts = [make_contact(0, 0.0, human_service_minutes=500.0),
                make_contact(1, 0.0, patience_minutes=10.0, **FAIL)]
    env, r, _ = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], C1: [A.WAIT, A.AI]}))
    # C1: WAIT 2 min (waited 2) -> AI 2 min -> failure at t=4: remaining (10-2)*0.5 = 4 -> abandons at t=8
    ab = events_of(r, EventType.ABANDONMENT, C1)
    assert len(ab) == 1 and ab[0].time == pytest.approx(8.0)
    esc = events_of(r, EventType.AI_ESCALATION, C1)[0].data
    assert esc["remaining_patience_before"] == pytest.approx(8.0) and esc["remaining_patience_after"] == pytest.approx(4.0)
    assert env.twin.contacts[C1].truth.patience_minutes == 10.0  # original patience untouched
    assert contact(r, C1).online_wait_minutes == pytest.approx(2.0 + 4.0)  # time already waited is retained


def test_multiplier_one_keeps_v1_patience():
    contacts = [make_contact(0, 0.0, human_service_minutes=500.0), make_contact(1, 0.0, patience_minutes=10.0, **FAIL)]
    plan = lambda: by_contact({C0: [A.HUMAN], C1: [A.WAIT, A.AI]})
    _, r1, _ = run_scripted(make_config(humans__n_advisors=1), contacts, plan())
    _, r2, _ = run_scripted(make_config(humans__n_advisors=1, v1_1__frustration_patience_multiplier=1.0), contacts, plan())
    assert _events(r1) == _events(r2)
    # cumulative online wait reaches 10: 2 min WAIT + 8 min in queue after the 2-min AI attempt -> t = 12
    assert events_of(r1, EventType.ABANDONMENT, C1)[0].time == pytest.approx(12.0)


# --------------------------------------------------------------------------- mechanism C: hangup
def test_immediate_hangup_is_one_abandonment():  # D-8, D-9
    cfg = make_config(v1_1__immediate_hangup_prob_after_ai_failure=1.0, v1_1__handoff_overhead_minutes=4.0,
                      v1_1__frustration_patience_multiplier=0.4)
    _, r, _ = run_scripted(cfg, [make_contact(0, 0.0, request_value=0.5, **FAIL)], fixed(A.AI))
    s = contact(r, C0)
    assert s.outcome is Outcome.ABANDONED and not s.escalated and not s.human_touched
    ab = events_of(r, EventType.ABANDONMENT, C0)
    assert len(ab) == 1 and ab[0].data["cause"] == "post_ai_failure_hangup" and ab[0].time == pytest.approx(2.0)
    assert not events_of(r, EventType.AI_ESCALATION) and not events_of(r, EventType.HUMAN_QUEUE_ENTERED)
    ep = r.episode_reward
    assert ep.abandonment == pytest.approx(-3 * 1.5) and ep.n_abandonments == 1
    assert ep.callback_failure == 0.0 and ep.unfinished == 0.0 and ep.human_busy_minutes == 0.0
    assert r.metrics["reconciliation"]["balanced"]


def test_hangup_rate_matches_probability_and_uses_own_stream():
    cfg = cell_config(BASE, Cell(0.0, 1.0, 0.05))
    r = run_simulation(cfg, AIFirstPolicy(), seed=1401)
    m = D.v11_metrics(r)
    assert 0.0 < m["hangup_share_of_visible_failures"] < 0.15
    ref = run_simulation(BASE, AIFirstPolicy(), seed=1401)
    ai = lambda res: [(e.contact_id, e.event_type) for e in res.events
                      if e.event_type in (EventType.AI_RESOLVED, EventType.AI_VISIBLE_FAILURE, EventType.AI_FALSE_RESOLUTION)]
    common = dict(ai(ref))
    assert sum(common.get(c) == t for c, t in ai(r)) > 0.9 * len(ai(r))  # AI draws untouched by the hangup stream


# --------------------------------------------------------------------------- invariants across the grid
def test_grid_is_exactly_the_32_preregistered_cells():  # D-12, D-13
    assert len(GRID) == 32 and set(GRID) == {Cell(h, m, p) for h, m, p in itertools.product(HANDOFF, MULTIPLIER, HANGUP)}
    assert HANDOFF == (0.0, 1.0, 2.0, 4.0) and MULTIPLIER == (1.0, 0.8, 0.6, 0.4) and HANGUP == (0.0, 0.05)
    assert isinstance(GRID, tuple)
    with pytest.raises(Exception):
        GRID[0].handoff = 9.0  # type: ignore[misc]
    if (OUT / "sweep" / "grid.json").exists():
        assert json.loads((OUT / "sweep" / "grid.json").read_text(encoding="utf-8")) == grid_manifest()


def test_reward_ai_rates_and_capacity_unchanged_in_every_cell_and_scenario():  # D-10, D-11, D-25
    base_day = ContactGenerator(BASE, RngManager(1401)).generate_day()
    for cfg in [cell_config(BASE, c) for c in GRID] + [scenario_config(BASE, s) for s in SCENARIOS]:
        assert cfg.reward == BASE.reward
        assert cfg.request_types == BASE.request_types and cfg.languages == BASE.languages
        assert cfg.ai == BASE.ai and cfg.ai.max_concurrency == 30  # V1 AI capacity, not swept or activated
    for c in (MAX_CELL, MID_CELL):
        assert ContactGenerator(cell_config(BASE, c), RngManager(1401)).generate_day() == base_day


def test_max_cell_day_reconciles():  # D-9
    r = run_simulation(cell_config(BASE, MAX_CELL), make_policy("RulePolicy", V1["RulePolicy"], 1402), seed=1402)
    assert r.metrics["reconciliation"]["balanced"]
    total = sum(t.reward for t in r.transitions) + r.pre_decision_reward.total
    assert total == pytest.approx(r.episode_reward.total)


# --------------------------------------------------------------------------- rationale and seeds
def test_rationale_frozen_before_results():  # D-14
    meta_path = OUT / "rationale" / "rationale.json"
    if not meta_path.exists():
        pytest.skip("Phase D not run yet")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["sha256"] == rationale_hash() and meta["frozen_before_results"]
    frozen_mtime = os.path.getmtime(meta_path)
    results = [p for p in OUT.rglob("*.json") if "rationale" not in p.parts]
    assert results and all(os.path.getmtime(p) > frozen_mtime for p in results)
    assert (OUT / "rationale" / "twin_v1_1_rationale.md").read_bytes() == RATIONALE_PATH.read_bytes()


def test_phase_d_seeds_disjoint_and_final_untouched():  # D-15, D-16
    assert PHASE_D_AUDIT_SEEDS == tuple(range(1401, 1411)) and PHASE_D_VALIDATION_SEEDS == tuple(range(1411, 1421))
    assert PHASE_D_HEADROOM_SEEDS == tuple(range(1421, 1431))
    for (a, sa), (b, sb) in itertools.combinations(SEED_SETS.items(), 2):
        assert not set(sa) & set(sb), (a, b)
    assert FINAL_TEST_RESERVED == range(2000, 3000)


def test_tuning_and_evaluation_seed_guards():  # D-23
    cfgs = {V1_CELL.cell_id: BASE}
    with pytest.raises(SeedPolicyError):
        D.tune(cfgs, 1, seeds=(1421,))
    with pytest.raises(SeedPolicyError):
        D.tune(cfgs, 1, seeds=(1011,))  # B2 validation seeds are not Phase D validation seeds
    with pytest.raises(SeedPolicyError):
        D.evaluate(cfgs, {"AIFirstPolicy": ("AIFirstPolicy", {})}, 1, seeds=(1411,))
    with pytest.raises(SeedPolicyError):
        D.audit(cfgs, {"AIFirstPolicy": ("AIFirstPolicy", {})}, 1, seeds=(2000,))


# --------------------------------------------------------------------------- scenarios
def test_scenario_5_dynamic_window_and_no_hidden_flag():  # D-17, D-24
    cfg = scenario_config(BASE, "SCENARIO_5")
    wins = [event_windows(cfg, s)["ai_degradation"] for s in (1401, 1402, 1403)]
    assert all(120 <= w[0] <= 420 and w[1] - w[0] == pytest.approx(180) for w in wins) and len({w[0] for w in wins}) == 3
    r = run_simulation(cfg, AIFirstPolicy(), seed=1401)
    lo, hi = wins[0]
    flagged = [e.time for e in r.events if e.event_type is EventType.AI_ATTEMPT_STARTED and e.data.get("degraded")]
    assert flagged and all(lo <= t < hi for t in flagged)
    unflagged = [e.time for e in r.events if e.event_type is EventType.AI_ATTEMPT_STARTED and not e.data.get("degraded")]
    assert not [t for t in unflagged if lo <= t < hi]
    assert all(set(t.state.to_dict()) == set(OBSERVATION_FEATURE_NAMES) for t in r.transitions)
    before = [t for t in r.transitions if t.timestamp < lo]
    base_run = run_simulation(BASE, AIFirstPolicy(), seed=1401)
    assert [t.state for t in before] == [t.state for t in base_run.transitions[: len(before)]]  # invisible before onset


def test_scenario_7_shock_invisible_before_observed_arrivals():  # D-18
    cfg = scenario_config(BASE, "SCENARIO_7")
    lo, hi = event_windows(cfg, 1402)["demand_shock"]
    day = ContactGenerator(cfg, RngManager(1402)).generate_day()
    base = ContactGenerator(BASE, RngManager(1402)).generate_day()
    shock = [c for c in day if c.contact_id.startswith("S")]
    assert shock and all(lo <= c.arrival_time < hi and c.request_type.value == "CHANGE_CANCEL" for c in shock)
    assert set(base) <= set(day)  # base day untouched
    r = run_simulation(cfg, AIFirstPolicy(), seed=1402)
    ref = run_simulation(BASE, AIFirstPolicy(), seed=1402)
    pre = [t.state for t in r.transitions if t.timestamp < lo]
    assert pre and pre == [t.state for t in ref.transitions[: len(pre)]]  # forecast identical until the shock shows
    post = [t.state.forecast_30min for t in r.transitions if lo + 30 <= t.timestamp < hi]
    post_ref = [t.state.forecast_30min for t in ref.transitions if lo + 30 <= t.timestamp < hi]
    assert np.mean(post) > np.mean(post_ref)  # ... and reacts once shock arrivals are observed


def test_scenario_8_components_match_config():  # D-19
    cfg = scenario_config(BASE, "SCENARIO_8")
    assert cfg.humans.n_advisors == 7 and cfg.scenario.extra_load_layers == [0.3]
    assert cfg.events.patience_scale == 0.8 and cfg.events.ai_degradation is not None
    assert cfg.v1_1.is_v1 and cfg.scenario.purpose == "PREDEFINED_V1_SCENARIO"
    with pytest.raises(ValueError):
        scenario_config(cell_config(BASE, MAX_CELL), "SCENARIO_8")
    day = ContactGenerator(cfg, RngManager(1401)).generate_day()
    base = {c.contact_id: c for c in ContactGenerator(BASE, RngManager(1401)).generate_day()}
    same_id = [c for c in day if c.contact_id in base]
    assert all(c.patience_minutes < base[c.contact_id].patience_minutes for c in same_id)


# --------------------------------------------------------------------------- headroom / classification
def test_screening_deterministic_and_worker_independent():  # D-21, D-26
    cfgs = {MAX_CELL.cell_id: cell_config(BASE, MAX_CELL)}
    rule, theta = {MAX_CELL.cell_id: V1["RulePolicy"]}, {MAX_CELL.cell_id: 0.3}
    quota = {"calm": 0, "rising": 0, "peak": 1, "post_peak": 0, "late": 1}
    a = D.build_states(cfgs, (1421,), rule, quota, theta, workers=1)
    assert a == D.build_states(cfgs, (1421,), rule, quota, theta, workers=2)
    cont = {MAX_CELL.cell_id: ("RulePolicy", V1["RulePolicy"])}
    r1 = D.run_headroom(cfgs, a, rule, cont, 2, workers=1)[MAX_CELL.cell_id]
    r2 = D.run_headroom(cfgs, a, rule, cont, 2, workers=2)[MAX_CELL.cell_id]
    assert r1["results"] == r2["results"] and r1["summary"]["classification"] == r2["summary"]["classification"]


def test_audit_reproducible():  # D-26
    cfgs = {MAX_CELL.cell_id: cell_config(BASE, MAX_CELL)}
    pol = {"AIFirstPolicy": ("AIFirstPolicy", {})}
    assert D.audit(cfgs, pol, 1, seeds=(1401,)) == D.audit(cfgs, pol, 1, seeds=(1401,))


def _summary(r, lo, hi=1.0, mean_j=400.0):
    return {"regret": {"RulePolicy": {"cross_fitted_ci": {"mean": r, "lo": lo, "hi": hi}}},
            "mean_J": {"RulePolicy_action": mean_j}, "near_tie_fractions": {"lt_5": 0.9}, "label_distribution": {"TIE": 0.5}}


def test_preregistered_classification_rule():  # D-22
    assert D.classify(_summary(0.6, 0.1))["label"] == "CLEARLY_POSITIVE"
    assert D.classify(_summary(0.2, 0.05))["label"] == "EMERGING"
    assert D.classify(_summary(0.7, -0.1))["label"] == "EMERGING"
    assert D.classify(_summary(0.3, -0.1))["label"] == "NEGLIGIBLE"
    assert D.classify(_summary(-0.2, -0.4))["label"] == "NEGLIGIBLE"
    assert D.classify(_summary(0.6, 0.1))["relative_to_mean_J"] == pytest.approx(0.6 / 400)


def test_preregistered_confirmatory_selection():
    screen = {c.cell_id: {"classification": {"label": "NEGLIGIBLE", "R": -0.1}} for c in GRID}
    sel = D.select_confirmatory(screen)
    ids = [c for c, _ in sel["cells"]]
    assert ids[:3] == [V1_CELL.cell_id, MAX_CELL.cell_id, MID_CELL.cell_id] and len(set(ids)) == 4
    screen["h2_m0.8_p0"] = {"classification": {"label": "EMERGING", "R": 0.3}}
    screen["h1_m0.6_p0.05"] = {"classification": {"label": "CLEARLY_POSITIVE", "R": 0.9}}
    sel = D.select_confirmatory(screen)
    assert sel["cells"][2][0] == "h2_m0.8_p0"  # severity 0.5+0.333 < 0.25+0.667+1
    assert sel["cells"][3][0] == "h1_m0.6_p0.05"


def test_learnability_detects_observable_signal_and_not_noise():
    rng = np.random.default_rng(0)
    states, analyses = [], []
    for i in range(120):
        x = float(rng.random())
        enc = [x] + [float(rng.random()) for _ in range(42)]
        human_better = 3.0 if x > 0.5 else -3.0
        G = {"AI": list(rng.normal(0, 0.3, 8)), "HUMAN": list(rng.normal(human_better, 0.3, 8))}
        sid = f"s{i}"
        states.append({"state_id": sid, "seed": 1421 + i % 6, "encoded": enc, "rule_action": "AI",
                       "observation": {}, "timestamp": 0.0})
        analyses.append({"state_id": sid, "status": "OK", "returns": G, "label": "TIE"})
    res = D.learnability(states, analyses, "sig")
    assert res["learnable_gain_vs_rule"]["lo"] > 0.5 and 0.2 < res["share_changed_from_rule"] < 0.8
    for a in analyses:  # destroy the feature-outcome link
        a["returns"]["HUMAN"] = list(rng.normal(0, 0.3, 8))
    noise = D.learnability(states, analyses, "noise")
    assert abs(noise["learnable_gain_vs_rule"]["mean"]) < 0.3
