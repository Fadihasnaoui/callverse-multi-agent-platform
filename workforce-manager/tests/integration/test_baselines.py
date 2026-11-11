"""Phase B2: baseline policies, seed protocol, tuning guards, paired evaluation and diagnostics."""

from __future__ import annotations

import ast
import dataclasses
import itertools
import json
from pathlib import Path

import pytest

from callverse_workforce.baselines import (
    BASELINE_NAMES,
    AIFirstPolicy,
    FixedActionPolicy,
    RandomPolicy,
    RulePolicy,
    ThresholdPolicy,
    make_policy,
)
from callverse_workforce.baselines import experiments as X
from callverse_workforce.baselines.metrics import REWARD_KEYS
from callverse_workforce.config import load_config
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import ACTION_ORDER, EventType, RoutingAction
from callverse_workforce.domain.observation import OBSERVATION_FEATURE_NAMES
from callverse_workforce.seeds import (
    B2_LOAD_CHECK_EVAL_SEEDS,
    B2_NORMAL_EVAL_SEEDS,
    B2_VALIDATION_SEEDS,
    CALIBRATION_SEEDS,
    FINAL_TEST_RESERVED,
    SEED_SETS,
    SeedPolicyError,
    registry_dict,
)
from callverse_workforce.twin import run_simulation
from helpers import make_observation

A = RoutingAction
NORMAL = load_config()
LOAD = load_config(PROJECT_ROOT / "configs" / "load_check.yaml")
FROZEN = {
    "RandomPolicy": {}, "AIFirstPolicy": {}, "ThresholdPolicy": {"theta": 0.5},
    "RulePolicy": {"theta": 0.6, "queue_max": 2, "congested_action": "CALLBACK"},
    "BestFixedPolicy": {"action": "CALLBACK"},
}


def _all_policies(seed=1011):
    return [make_policy(n, FROZEN[n], seed) for n in BASELINE_NAMES] + [
        RulePolicy(0.5, 2, "WAIT"), RulePolicy(0.5, 2, "AI")] + [FixedActionPolicy(a) for a in X.FIXED_ACTIONS]


# --------------------------------------------------------------------------- seeds
def test_seed_registry_is_exact_and_disjoint():
    assert CALIBRATION_SEEDS == tuple(range(1001, 1011))
    assert B2_VALIDATION_SEEDS == tuple(range(1011, 1021))
    assert B2_NORMAL_EVAL_SEEDS == tuple(range(1101, 1121))
    assert B2_LOAD_CHECK_EVAL_SEEDS == tuple(range(1201, 1221))
    assert FINAL_TEST_RESERVED == range(2000, 3000)
    for (a, sa), (b, sb) in itertools.combinations(SEED_SETS.items(), 2):
        assert not set(sa) & set(sb), (a, b)
    assert registry_dict()["final_test_reserved"] == [2000, 2999]


def test_calibration_and_final_test_seeds_are_rejected_everywhere():
    for bad in (1001, 1010, 2000, 2500, 2999, 1101):
        with pytest.raises(SeedPolicyError):
            X.tune_threshold(NORMAL, seeds=(bad,), grid=(0.5,))
    for bad in (1005, 2001, 1015):
        with pytest.raises(SeedPolicyError):
            X.evaluate_frozen(NORMAL, FROZEN, (bad,), "NORMAL")
    with pytest.raises(SeedPolicyError):
        X.evaluate_frozen(LOAD, FROZEN, (1101,), "LOAD_CHECK")  # NORMAL eval seeds are not LOAD_CHECK seeds


# --------------------------------------------------------------------------- policies
def test_every_policy_respects_masks_exhaustively():
    obs_variants = [make_observation(ai_resolution_score=s, humans_available=h, human_queue_size=q)
                    for s in (0.0, 0.55, 1.0) for h in (0, 3) for q in (0, 20)]
    # HUMAN is never masked in V1, so every reachable mask has HUMAN valid.
    masks = [ActionMask(valid=(ai, True, w, cb)) for ai, w, cb in itertools.product((True, False), repeat=3)]
    for policy in _all_policies():
        for obs in obs_variants:
            for mask in masks:
                assert mask.is_valid(policy.predict(obs, mask)), (policy, mask)


@pytest.mark.parametrize("config", [NORMAL, LOAD], ids=["normal", "load_check"])
def test_policies_respect_masks_in_the_twin(config):
    for name in BASELINE_NAMES:
        r = run_simulation(config, make_policy(name, FROZEN[name], 1011), seed=1011)
        assert all(d.action_mask.is_valid(d.action) for d in r.decisions)
        assert r.metrics["reconciliation"]["balanced"]


def test_random_policy_reproducible_and_seed_dependent():
    obs, mask = make_observation(), ActionMask(valid=(True, True, True, True))
    seq = lambda p: [p.predict(obs, mask) for _ in range(50)]
    a, b, c = RandomPolicy(7), RandomPolicy(7), RandomPolicy(8)
    assert seq(a) == seq(b) != seq(c)
    a.reset()
    assert seq(a) == seq(RandomPolicy(7))
    assert set(seq(RandomPolicy(9))) == set(ACTION_ORDER)


def test_deterministic_runner_reproducible():
    for name in BASELINE_NAMES:
        x = run_simulation(NORMAL, make_policy(name, FROZEN[name], 1012), seed=1012)
        y = run_simulation(NORMAL, make_policy(name, FROZEN[name], 1012), seed=1012)
        assert [e.to_dict() for e in x.events] == [e.to_dict() for e in y.events]
        assert x.total_reward == y.total_reward


def test_ai_first_order_and_threshold_fallback_is_human():
    obs = make_observation(ai_resolution_score=0.95)
    assert AIFirstPolicy().predict(obs, ActionMask(valid=(True, True, True, True))) is A.AI
    assert AIFirstPolicy().predict(obs, ActionMask(valid=(False, True, True, True))) is A.HUMAN
    t = ThresholdPolicy(0.6)
    full = ActionMask(valid=(True, True, True, True))
    assert t.predict(make_observation(ai_resolution_score=0.6), full) is A.AI
    assert t.predict(make_observation(ai_resolution_score=0.59), full) is A.HUMAN  # not WAIT / CALLBACK
    assert t.predict(obs, ActionMask(valid=(False, True, True, True))) is A.HUMAN


def test_rule_policy_branches():
    rule = RulePolicy(theta=0.6, queue_max=3, congested_action="CALLBACK")
    full = ActionMask(valid=(True, True, True, True))
    assert rule.predict(make_observation(ai_resolution_score=0.7), full) is A.AI
    assert rule.predict(make_observation(ai_resolution_score=0.2, humans_available=1, human_queue_size=9), full) is A.HUMAN
    assert rule.predict(make_observation(ai_resolution_score=0.2, humans_available=0, human_queue_size=2), full) is A.HUMAN
    congested = make_observation(ai_resolution_score=0.2, humans_available=0, human_queue_size=3)
    assert rule.predict(congested, full) is A.CALLBACK
    assert rule.predict(congested, ActionMask(valid=(True, True, True, False))) is A.HUMAN  # callback masked


def test_fixed_policy_fallback_order():
    obs = make_observation()
    assert FixedActionPolicy("CALLBACK").predict(obs, ActionMask(valid=(True, True, True, False))) is A.HUMAN
    assert FixedActionPolicy("WAIT").predict(obs, ActionMask(valid=(True, True, False, True))) is A.HUMAN
    assert FixedActionPolicy("AI").predict(obs, ActionMask(valid=(False, True, True, True))) is A.HUMAN


class _RecordingObservation:
    """Proxy that records attribute reads and refuses anything outside the 28 observable features."""

    def __init__(self, obs):
        self._obs, self.read = obs, set()

    def __getattr__(self, item):
        if item not in OBSERVATION_FEATURE_NAMES:
            raise AttributeError(f"policy tried to read non-observable attribute {item!r}")
        self.read.add(item)
        return getattr(self._obs, item)


def test_no_policy_accesses_hidden_truth():
    import callverse_workforce.baselines.policies as mod

    tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    forbidden = ("twin", "domain.contact", "config", "intake", "forecasting", "reward")
    assert not [m for m in imported if any(f in m for f in forbidden)], imported
    for policy in _all_policies():
        for score in (0.1, 0.9):
            proxy = _RecordingObservation(make_observation(ai_resolution_score=score, humans_available=0,
                                                           human_queue_size=10))
            policy.predict(proxy, ActionMask(valid=(True, True, True, True)))
            assert proxy.read <= set(OBSERVATION_FEATURE_NAMES)


def test_frozen_policies_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        ThresholdPolicy(0.5).theta = 0.9  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        RulePolicy(0.5, 2, "AI").queue_max = 99  # type: ignore[misc]


# --------------------------------------------------------------------------- tuning
def test_tuning_uses_only_validation_seeds_and_normal_days():
    th = X.tune_threshold(NORMAL, seeds=(1011, 1012), grid=(0.4, 0.8))
    assert th["seeds"] == [1011, 1012] and set(th["seeds"]) <= set(B2_VALIDATION_SEEDS)
    assert all(set(c["per_seed_reward"]) == {"1011", "1012"} for c in th["candidates"])
    best = max(th["candidates"], key=lambda c: c["mean_reward"])
    assert th["selected"] == best["params"]
    rule = X.tune_rule(NORMAL, seeds=(1013,), grid={"theta": (0.6,), "queue_max": (2,), "congested_action": ("AI", "WAIT")})
    assert rule["seeds"] == [1013] and len(rule["candidates"]) == 2
    fixed = X.select_best_fixed(NORMAL, seeds=(1014,), actions=("HUMAN", "AI"))
    assert fixed["seeds"] == [1014] and fixed["selected"]["action"] in ("HUMAN", "AI")
    for fn, kw in ((X.tune_threshold, {"grid": (0.5,)}), (X.tune_rule, {}), (X.select_best_fixed, {})):
        with pytest.raises(SeedPolicyError):
            fn(LOAD, seeds=(1011,), **kw)  # LOAD_CHECK may never be used for tuning
        with pytest.raises(SeedPolicyError):
            fn(NORMAL, seeds=(1011, 1101), **kw)  # evaluation seed leaks into validation


def test_selection_ties_keep_first_candidate():
    rows = [{"params": {"x": 1}, "mean_reward": 5.0}, {"params": {"x": 2}, "mean_reward": 5.0}]
    best = 0
    for i, r in enumerate(rows):
        if r["mean_reward"] > rows[best]["mean_reward"]:
            best = i
    assert best == 0  # same rule as experiments._select


# --------------------------------------------------------------------------- evaluation
@pytest.fixture(scope="module")
def evaluation():
    return X.evaluate_frozen(NORMAL, FROZEN, B2_NORMAL_EVAL_SEEDS[:2], "NORMAL")


def test_selected_parameters_frozen_during_evaluation(evaluation):
    for name in BASELINE_NAMES:
        block = evaluation["policies"][name]
        assert block["params"] == FROZEN[name]
        assert all(d["params"] == FROZEN[name] and d["policy"] == name for d in block["days"])


def test_evaluation_is_paired_and_exogenous_arrivals_identical():
    seed = B2_NORMAL_EVAL_SEEDS[0]

    def exogenous(r):
        return [(e.contact_id, e.time, json.dumps(e.to_dict()["data"], sort_keys=True)) for e in r.events
                if e.event_type in (EventType.ARRIVAL, EventType.INTAKE_PREDICTION) and "-R" not in e.contact_id]

    runs = [run_simulation(NORMAL, make_policy(n, FROZEN[n], seed), seed=seed) for n in BASELINE_NAMES]
    ref = exogenous(runs[0])
    assert all(exogenous(r) == ref for r in runs[1:])
    assert len({round(r.total_reward, 6) for r in runs}) > 1  # the policies genuinely differ


def test_day_metrics_consistency(evaluation):
    for name in BASELINE_NAMES:
        for d in evaluation["policies"][name]["days"]:
            assert d["reconciled"]
            parts = sum(d["reward"][k] for k in REWARD_KEYS if k != "total")
            assert parts == pytest.approx(d["reward"]["total"]) == pytest.approx(d["total_reward"])
            assert sum(d["action_share"].values()) == pytest.approx(1.0)
            assert sum(d["valid_action_pct"].values()) == pytest.approx(1.0)
            assert d["human_routing_rate"] == pytest.approx(d["action_share"]["HUMAN"])
            assert d["dominant_action_share"] == pytest.approx(max(d["action_share"].values()))
            assert 0.0 <= d["hir_instance"] <= 1.0 and 0.0 <= d["hir_root"] <= 1.0


def test_hir_and_routing_rate_are_distinct_metrics():
    r = run_simulation(NORMAL, FixedActionPolicy("HUMAN"), seed=1011)
    from callverse_workforce.baselines.metrics import policy_day_metrics

    d = policy_day_metrics(r)
    assert d["human_routing_rate"] == 1.0
    assert d["hir_instance"] < 1.0  # routed to HUMAN is not the same as served (abandonments)
    assert d["hir_instance"] == pytest.approx(sum(c.human_touched for c in r.contacts) / len(r.contacts))


def test_mask_freedom_statistics(evaluation):
    r = run_simulation(NORMAL, make_policy("RandomPolicy", {}, B2_NORMAL_EVAL_SEEDS[0]), seed=B2_NORMAL_EVAL_SEEDS[0])
    from callverse_workforce.baselines.metrics import policy_day_metrics

    d = policy_day_metrics(r)
    n = len(r.decisions)
    for k in (1, 2, 3, 4):
        assert d["valid_action_pct"][str(k)] == pytest.approx(sum(x.n_valid_actions == k for x in r.decisions) / n)
    assert d["mean_valid_actions"] == pytest.approx(sum(x.n_valid_actions for x in r.decisions) / n)
    assert d["total_reward"] == evaluation["policies"]["RandomPolicy"]["days"][0]["total_reward"]  # same day


def _fake_days(share, n=3, **extra):
    base = {"action_share": share, "callbacks_scheduled": 100, "callback_success": 60, "callback_dropout": 10,
            "callback_max_attempts": 20, "callback_window_expired": 10, "callback_human_minutes": 500.0,
            "callback_reward_contribution": 42.0}
    base.update(extra)
    return [dict(base) for _ in range(n)]


def test_callback_overuse_and_dominant_action_warnings():
    d = X.diagnostics(_fake_days({"AI": 0.1, "HUMAN": 0.2, "WAIT": 0.0, "CALLBACK": 0.7}))
    assert d["callback_share"] == pytest.approx(0.7) and d["callback_overuse"]
    assert X.CALLBACK_WARNING in d["warnings"]
    assert d["callback_detail"]["success_rate"] == pytest.approx(0.6)
    assert d["callback_detail"]["callback_reward_contribution_mean"] == 42.0
    calm = X.diagnostics(_fake_days({"AI": 0.5, "HUMAN": 0.4, "WAIT": 0.0, "CALLBACK": 0.1}))
    assert not calm["callback_overuse"] and not calm["dominant_over_90"] and not calm["warnings"]
    d92 = X.diagnostics(_fake_days({"AI": 0.92, "HUMAN": 0.08, "WAIT": 0.0, "CALLBACK": 0.0}))
    assert d92["dominant_over_90"] and not d92["dominant_over_95"] and not d92["nearly_constant"]
    d97 = X.diagnostics(_fake_days({"AI": 0.0, "HUMAN": 0.97, "WAIT": 0.03, "CALLBACK": 0.0}))
    assert d97["dominant_over_95"] and d97["nearly_constant"] and d97["dominant_action"] == "HUMAN"
    assert any(">95%" in w for w in d97["warnings"])


def test_paired_variability_structure(evaluation):
    p = evaluation["paired"]
    assert p["seeds"] == list(B2_NORMAL_EVAL_SEEDS[:2])
    assert p["diff_vs_best"][p["best_by_mean_reward"]]["mean"] == 0.0
    assert sum(p["per_day_winner_counts"].values()) == 2


ARTIFACTS = PROJECT_ROOT / "artifacts" / "baselines"


@pytest.mark.skipif(not (ARTIFACTS / "validation.json").exists(), reason="run scripts/run_baselines.py first")
def test_persisted_artifacts_respect_the_seed_protocol():
    val = json.loads((ARTIFACTS / "validation.json").read_text(encoding="utf-8"))
    allowed = {str(s) for s in B2_VALIDATION_SEEDS}
    for key in ("threshold", "rule", "best_fixed"):
        assert val[key]["seeds"] == list(B2_VALIDATION_SEEDS)
        assert all(set(c["per_seed_reward"]) == allowed for c in val[key]["candidates"])
    assert [c["params"]["theta"] for c in val["threshold"]["candidates"]] == list(X.THRESHOLD_GRID)
    assert len(val["rule"]["candidates"]) == 27 and len(val["best_fixed"]["candidates"]) == 4
    frozen = json.loads((ARTIFACTS / "frozen_policies.json").read_text(encoding="utf-8"))["frozen"]
    for stem, seeds in (("eval_normal", B2_NORMAL_EVAL_SEEDS), ("eval_load_check", B2_LOAD_CHECK_EVAL_SEEDS)):
        ev = json.loads((ARTIFACTS / f"{stem}.json").read_text(encoding="utf-8"))
        assert ev["seeds"] == list(seeds)
        for name in BASELINE_NAMES:
            assert ev["policies"][name]["params"] == frozen[name]  # identical frozen parameters on both sets
