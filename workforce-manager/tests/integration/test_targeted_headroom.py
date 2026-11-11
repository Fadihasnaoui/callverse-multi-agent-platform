"""Targeted initial-routing headroom audit: filters, protocol, statistics and labels."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from callverse_workforce.config import load_config
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.domain.observation import OBSERVATION_FEATURE_NAMES
from callverse_workforce.encoding import OBSERVATION_DIM
from callverse_workforce.headroom import stats as HS
from callverse_workforce.phase_e.scenarios import TRAINING_DISTRIBUTION
from callverse_workforce.targeted import audit as TA

OUT = PROJECT_ROOT / "artifacts" / "targeted_initial_headroom"
SRC = PROJECT_ROOT / "src" / "callverse_workforce" / "targeted" / "audit.py"
BASE = load_config()


def _rec(**o):
    obs = {"humans_available": 0, "human_queue_size": 1, "urgency_pred": "LOW", "deadline_remaining_pred": 600.0,
           "waiting_time_so_far": 0.0, "ai_resolution_score": 0.3, "oldest_human_wait": 2.0, "callback_allowed": True,
           "callbacks_pending": 2, "oldest_callback_age": 10.0, "routing_attempt_count": 0, "request_type": "FAQ"}
    mask = {"AI": True, "HUMAN": True, "WAIT": True, "CALLBACK": True}
    obs.update({k: v for k, v in o.items() if k not in ("mask", "reasons", "contact_id")})
    return {"observation": obs, "mask": o.get("mask", mask), "mask_reasons": o.get("reasons", {}),
            "contact_id": o.get("contact_id", "C00001")}


def test_frozen_components_and_no_training():  # T-1..T-5, T-32..T-34
    assert OBSERVATION_DIM == 43 and len(OBSERVATION_FEATURE_NAMES) == 28
    assert TRAINING_DISTRIBUTION["WORST_DAY"] == 0.03  # no scenario oversampling
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    imported = {(n.module, a.name) for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module for a in n.names}
    from_training = {name for mod, name in imported if mod.endswith("phase_e1.training")}
    assert from_training <= {"worker_sampler_seed"}  # only to regenerate past seed streams for the collision audit
    assert not [m for m, _ in imported if "smdp" in m or "baselines.experiments" in m]
    src = SRC.read_text(encoding="utf-8")
    for token in ("learn(", "train_seed(", "build_model(", "SMDPMaskablePPO", "HUMAN_HIGH", "apply_overrides"):
        assert token not in src


def test_initial_state_definition():  # T-6
    assert TA.is_initial(_rec())
    assert not TA.is_initial(_rec(routing_attempt_count=1))  # WAIT re-evaluation
    assert not TA.is_initial(_rec(contact_id="C00001-R1"))  # recontact routing


def test_wait_filter_exact():  # T-8
    assert TA.wait_favourable(_rec())
    for bad in ({"humans_available": 1}, {"human_queue_size": 3}, {"urgency_pred": "HIGH"}, {"deadline_remaining_pred": 59.9},
                {"waiting_time_so_far": 5.1}, {"ai_resolution_score": 0.51}, {"oldest_human_wait": 5.1},
                {"mask": {"AI": True, "HUMAN": True, "WAIT": False, "CALLBACK": True}}):
        assert not TA.wait_favourable(_rec(**bad)), bad
    assert TA.wait_favourable(_rec(human_queue_size=2, deadline_remaining_pred=60.0, ai_resolution_score=0.5, oldest_human_wait=5.0))


def test_callback_filter_exact():  # T-9
    good = dict(human_queue_size=5)
    assert TA.callback_favourable(_rec(**good))
    for bad in ({"human_queue_size": 4}, {"humans_available": 1}, {"urgency_pred": "HIGH"}, {"deadline_remaining_pred": 119.0},
                {"ai_resolution_score": 0.6}, {"callbacks_pending": 6}, {"oldest_callback_age": 31.0}, {"callback_allowed": False},
                {"mask": {"AI": True, "HUMAN": True, "WAIT": True, "CALLBACK": False}},
                {"reasons": {"CALLBACK": ["callback_unavailable"]}}):
        assert not TA.callback_favourable(_rec(**good | bad)), bad


def test_filters_use_observable_fields_only():  # T-10..T-14
    allowed = set(OBSERVATION_FEATURE_NAMES)
    src = SRC.read_text(encoding="utf-8")
    fn_src = src[src.index("def wait_favourable"):src.index("# seeds")]
    used = {n.slice.value for n in ast.walk(ast.parse(fn_src)) if isinstance(n, ast.Subscript)
            and isinstance(n.slice, ast.Constant) and isinstance(n.slice.value, str)}
    assert used - {"WAIT", "CALLBACK", "callback_unavailable"} <= allowed | {"observation", "mask", "mask_reasons"}
    for hidden in ("patience", "truth", "service_minutes", "future", "deadline_at", "shortage_window", "degradation_window"):
        assert hidden not in fn_src
    m = OUT / "observable_field_mapping.json"
    c = OUT / "state_collection.json"
    if m.exists() and c.exists():
        assert os.path.getmtime(m) < os.path.getmtime(c)  # mapping frozen before collection


def test_seed_block_audit_and_final_untouched():  # T-15..T-17
    a = OUT / "seed_audit.json"
    if not a.exists():
        pytest.skip("not run")
    audit = json.loads(a.read_text(encoding="utf-8"))
    lo, hi = audit["seed_block"]
    assert lo >= 300_000 and audit["tried"][-1]["n_collisions"] == 0
    sources = TA.known_seed_sources(PROJECT_ROOT)
    assert not set().union(*sources.values()) & set(range(lo, hi + 1))
    assert not set(range(2000, 3000)) & set(range(lo, hi + 1))
    col = json.loads((OUT / "state_collection.json").read_text(encoding="utf-8"))
    assert all(lo <= d["seed"] <= hi for d in col["days"])


def test_stopping_rule():  # T-18, T-19
    import inspect

    src = inspect.getsource(TA.collect)
    assert "len(wait) >= TARGET_N and len(cb) >= TARGET_N" in src and "MAX_INITIAL_STATES" in src
    assert TA.TARGET_N == 300 and TA.MAX_INITIAL_STATES == 100_000
    col = OUT / "state_collection.json"
    if col.exists():
        c = json.loads(col.read_text(encoding="utf-8"))
        w = json.loads((OUT / "wait_subset" / "states.json").read_text(encoding="utf-8"))
        cb = json.loads((OUT / "callback_subset" / "states.json").read_text(encoding="utf-8"))
        assert (len(w) >= 300 and len(cb) >= 300) or c["initial_states_inspected"] >= 100_000
        assert all(TA.wait_favourable(s | {"mask_reasons": s["mask_reasons"]}) for s in w[:200])
        assert all(TA.callback_favourable(s) for s in cb)
        assert os.path.getmtime(OUT / "preregistration.json") < os.path.getmtime(col)


def test_rule_policy_behaviour_and_continuations():  # T-7, T-23, T-24, T-32
    p = OUT / "preregistration.json"
    if p.exists():
        pre = json.loads(p.read_text(encoding="utf-8"))
        assert pre["state_generation_policy"] == "RulePolicy" and pre["primary_continuation"] == "RulePolicy"
        assert pre["robustness_continuation"] == "AIFirstPolicy" and pre["K"] == 16
        assert pre["state_generation_policy_params"] == {"theta": 0.5, "queue_max": 2, "congested_action": "AI"}
    assert TA.K_FUTURES == 16


def test_paired_futures_identical_across_actions():  # T-21
    from callverse_workforce.headroom.branch import simulate_branch
    from callverse_workforce.headroom.sampling import trajectory
    from callverse_workforce.phase_e.scenarios import day_config
    from callverse_workforce.phase_d.grid import MIXTURE_CELLS

    rule = {"theta": 0.5, "queue_max": 2, "congested_action": "AI"}
    cfg = day_config(BASE, "NORMAL", MIXTURE_CELLS[0])
    rec = next(r for r in trajectory(cfg, 310_001, rule, 0.4) if TA.is_initial(r) and all(r["mask"].values()) and r["timestamp"] > 120)
    st = TA._record(rec, TA.day_spec(310_001).__class__("NORMAL", MIXTURE_CELLS[0], 310_001), "WAIT")
    jobs = [j for j in TA.jobs_for(BASE, [st], rule, ("RulePolicy", rule), k=2)]
    arrivals = {}
    for j in jobs:
        out, env, _, d = simulate_branch(j)
        assert out["status"] == "OK"
        arrivals[(j.action, j.variant)] = [(e.contact_id, e.time) for e in env.twin.events.records[d["events_before"]:]
                                           if e.event_type.value == "ARRIVAL" and e.contact_id.startswith("V")]
    for v in (0, 1):
        assert len({tuple(arrivals[(a, v)]) for a in ("AI", "HUMAN", "WAIT", "CALLBACK")}) == 1
    assert arrivals[("AI", 0)] != arrivals[("AI", 1)]


def test_phase_c_logic_and_crossfit():  # T-25, T-26
    rets = {"WAIT": [5.0] * 8 + [5.2] * 8, "AI": [1.0] * 16, "HUMAN": [2.0] * 16}
    a = HS.analyze_state(rets, {"RulePolicy": "AI"}, "x")
    assert a["label"] == "WAIT_BEST"
    assert TA.crossfit_advantage(rets, "WAIT") == pytest.approx(((5.2 - 2.0) + (5.0 - 2.0)) / 2)
    noisy = {"WAIT": [0.0] * 16, "AI": [1, -1] * 8, "HUMAN": [-1, 1] * 8}
    assert TA.crossfit_advantage(noisy, "WAIT") == pytest.approx(0.0)  # selection bias removed


def test_cluster_bootstrap_resamples_days():  # T-27
    vals = [1.0] * 10 + [0.0] * 10
    days = [1] * 10 + [2] * 10
    ci = HS.cluster_bootstrap_mean(vals, days, "k", b=2000)
    assert ci["lo"] == 0.0 and ci["hi"] == 1.0  # only whole days can be drawn: means are 0, 0.5 or 1


def test_thresholds_and_labels():  # T-20, T-28..T-31, T-35
    assert (TA.CLEAR_MEANINGFUL, TA.ADV_MEANINGFUL, TA.CLEAR_WEAK, TA.LOW_SAMPLE) == (0.10, 0.5, 0.05, 100)
    s = lambda clear, mean, lo: {"clearly_best_share": clear, "crossfit_advantage": {"mean": mean, "ci": {"lo": lo}}}
    assert TA.classify(s(0.12, 0.6, 0.1), 300)["label"] == "MEANINGFUL INITIAL-ROUTING NICHE"
    assert TA.classify(s(0.12, 0.6, -0.1), 300)["label"] == "RARE / WEAK NICHE"
    assert TA.classify(s(0.06, 0.1, -0.1), 300)["label"] == "RARE / WEAK NICHE"
    assert TA.classify(s(0.02, 0.1, -0.1), 300)["label"] == "NO USEFUL INITIAL NICHE"
    assert TA.classify(s(0.9, 9.0, 5.0), 99)["label"] == "TOO RARE TO MATTER AT INITIAL ROUTING"
    assert TA.classify(s(0.12, 0.6, 0.1), 150)["limited_power"]
    assert TA.final_decision("NO USEFUL INITIAL NICHE", "RARE / WEAK NICHE")["verdict"] == "INITIAL ROUTING — AI/HUMAN ONLY; MOVE WAIT/CALLBACK TO V2"
    assert TA.final_decision("MEANINGFUL INITIAL-ROUTING NICHE", "NO USEFUL INITIAL NICHE")["case"] == "B"
    assert TA.final_decision("NO USEFUL INITIAL NICHE", "MEANINGFUL INITIAL-ROUTING NICHE")["case"] == "C"
    assert TA.final_decision("MEANINGFUL INITIAL-ROUTING NICHE", "MEANINGFUL INITIAL-ROUTING NICHE")["case"] == "D"


def test_replay_amendment_before_replay():
    a = OUT / "replay_sample_amendment.json"
    r = OUT / "paired_replays"
    if a.exists() and r.exists():
        assert all(os.path.getmtime(a) < os.path.getmtime(p) for p in r.glob("*.jsonl"))
        assert len(json.loads(a.read_text(encoding="utf-8"))["wait_replay_state_ids"]) == 300
