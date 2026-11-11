"""Phase E: Gym environment, SMDP-MaskablePPO extension, protocol guards, statistics and verdict."""

from __future__ import annotations

import ast
import importlib.util
import itertools
import json
import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sb3_contrib")
pytest.importorskip("gymnasium")

from callverse_workforce.config import load_config  # noqa: E402
from callverse_workforce.config.loader import PROJECT_ROOT  # noqa: E402
from callverse_workforce.domain.enums import ACTION_ORDER  # noqa: E402
from callverse_workforce.phase_d.grid import GRID, MIXTURE_CELLS, MAX_CELL, MID_CELL  # noqa: E402
from callverse_workforce.phase_e import evaluation as EV  # noqa: E402
from callverse_workforce.phase_e import scenarios as SC  # noqa: E402
from callverse_workforce.phase_e import training as TR  # noqa: E402
from callverse_workforce.phase_e.env import REWARD_SCALE, CallVerseRoutingEnv  # noqa: E402
from callverse_workforce.phase_e.smdp import SMDPMaskablePPO, SMDPRolloutBuffer, smdp_gae, transition_discounts  # noqa: E402
from callverse_workforce.seeds import (  # noqa: E402
    FINAL_TEST_RESERVED,
    PPO_DEV_EVAL_SEEDS,
    PPO_VALIDATION_SEEDS,
    SEED_SETS,
    SeedPolicyError,
)

BASE = load_config()
OUT = PROJECT_ROOT / "artifacts" / "ppo_v1_1"
PHASE_E = PROJECT_ROOT / "src" / "callverse_workforce" / "phase_e"
_spec = importlib.util.spec_from_file_location("run_phase_e", PROJECT_ROOT / "scripts" / "run_phase_e.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)


# --------------------------------------------------------------------------- environment definition
def test_exact_phase_d_mixture_frozen():  # E-1, E-2
    assert SC.MIXTURE_PROBABILITIES == (0.25, 0.25, 0.25, 0.25)
    assert [c.cell_id for c in MIXTURE_CELLS] == ["h0_m1_p0", "h1_m0.8_p0", "h2_m0.6_p0.05", "h4_m0.4_p0.05"]
    other = next(c for c in GRID if c not in MIXTURE_CELLS)
    with pytest.raises(ValueError):
        SC.day_config(BASE, "NORMAL", other)
    sampler = SC.TrainingDaySampler(3)
    days = [sampler() for _ in range(3000)]
    assert {d.cell for d in days} == set(MIXTURE_CELLS) and MAX_CELL not in MIXTURE_CELLS[:3]
    assert all(d.seed >= SC.TRAINING_SEED_MIN for d in days)
    fams = {f: sum(d.family == f for d in days) / len(days) for f in SC.TRAINING_DISTRIBUTION}
    assert all(abs(fams[f] - p) < 0.03 for f, p in SC.TRAINING_DISTRIBUTION.items())
    if (OUT / "environment_definition.json").exists():
        env_def = json.loads((OUT / "environment_definition.json").read_text(encoding="utf-8"))
        assert [c["cell_id"] for c in env_def["mixture_cells"]] == [c.cell_id for c in MIXTURE_CELLS]
        assert env_def["mixture_probabilities"] == list(SC.MIXTURE_PROBABILITIES)
    assert MID_CELL not in MIXTURE_CELLS


def test_all_eight_families_exist_and_configs_frozen():  # E-3, E-4
    assert SC.FAMILIES == ("NORMAL", "SUMMER", "RAMADAN", "OMRA_HOLIDAYS", "AI_DEGRADATION", "HUMAN_SHORTAGE",
                           "DEMAND_SHOCK", "WORST_DAY")
    reg_path = OUT / "scenario_registry.json"
    if not reg_path.exists():
        pytest.skip("freeze stage not run")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))["configs"]
    for key, entry in reg.items():
        fam, cid = key.split("|")
        cell = {c.cell_id: c for c in MIXTURE_CELLS}[cid]
        assert SC.config_hash(SC.day_config(BASE, fam, cell)) == entry["sha256"], key


def test_new_family_definitions():
    s = SC.day_config(BASE, "SUMMER", MIXTURE_CELLS[0])
    assert s.forecasting.reference_hourly_arrival_rates == s.scenario.hourly_arrival_rates == SC.SUMMER_PROFILE
    r = SC.day_config(BASE, "RAMADAN", MIXTURE_CELLS[0])
    assert r.forecasting.reference_hourly_arrival_rates == SC.RAMADAN_PROFILE
    o = SC.day_config(BASE, "OMRA_HOLIDAYS", MIXTURE_CELLS[0])
    assert o.scenario.day_type.value == "HOLIDAY" and o.request_types["BOOKING"].share == 0.38
    h = SC.day_config(BASE, "HUMAN_SHORTAGE", MIXTURE_CELLS[0])
    assert h.events.human_shortage.absent_advisors == 3 and h.humans.n_advisors == 8
    n = SC.day_config(BASE, "AI_DEGRADATION", MIXTURE_CELLS[0])
    assert n.forecasting == BASE.forecasting  # non-calendar families keep the historical normal reference
    for fam in SC.FAMILIES:
        assert SC.day_config(BASE, fam, MIXTURE_CELLS[1]).reward == BASE.reward


def test_forecast_leakage_checks_pass():  # E-5, E-6
    from callverse_workforce.phase_e.audit import leakage_checks

    res = leakage_checks(BASE, (100_001,), {"theta": 0.5, "queue_max": 2, "congested_action": "AI"})
    assert res and all(v["ok"] for v in res.values())


def test_human_shortage_reduces_available_advisors_only_during_window():
    from callverse_workforce.phase_e.evaluation import windows
    from callverse_workforce.twin import run_simulation
    from callverse_workforce.baselines.policies import AIFirstPolicy

    cfg = SC.day_config(BASE, "HUMAN_SHORTAGE", MIXTURE_CELLS[0])
    w = windows(cfg, 100_003)["human_shortage"]
    assert 180 <= w[0] <= 420 and w[1] - w[0] == pytest.approx(240)
    r = run_simulation(cfg, AIFirstPolicy(), seed=100_003)
    during = [t.state.humans_available for t in r.transitions if w[0] + 30 < t.timestamp < w[1]]
    assert max(during) <= 5 and r.metrics["reconciliation"]["balanced"]


# --------------------------------------------------------------------------- gymnasium env
def _env(seed=5):
    return CallVerseRoutingEnv(BASE, SC.TrainingDaySampler(seed))


def test_gym_env_spaces_and_episode():  # E-7..E-12, E-17, E-18, E-44, E-46
    env = _env()
    assert env.observation_space.shape == (43,) and env.observation_space.dtype == np.float32
    assert env.action_space.n == 4 and [a.value for a in ACTION_ORDER] == ["AI", "HUMAN", "WAIT", "CALLBACK"]
    obs, info = env.reset(options={"day": SC.DaySpec("NORMAL", MIXTURE_CELLS[0], 100_005)})
    assert obs.shape == (43,) and obs.dtype == np.float32 and env.observation_space.contains(obs)
    assert "day" in info and len(info) == 1  # nothing hidden besides logging metadata
    rng = np.random.default_rng(0)
    total_scaled, raw, done, n = 0.0, 0.0, False, 0
    while not done:
        m = env.action_masks()
        assert m.dtype == bool and m.shape == (4,) and m[1]  # HUMAN always valid
        obs, r, done, trunc, info = env.step(int(rng.choice(np.flatnonzero(m))))
        assert not trunc and info["delta_t_minutes"] >= 0
        assert r == pytest.approx(info["raw_reward"] / REWARD_SCALE)
        assert info["raw_reward"] == pytest.approx(info["reward_breakdown"]["total"])
        total_scaled += r
        raw += info["raw_reward"]
        n += 1
    assert info["termination_reason"] == "END_OF_DAY" and n > 100
    assert info["day_raw_reward"] == pytest.approx(raw + env._env.pre_decision_reward.total)
    assert total_scaled == pytest.approx(raw / 100)
    assert env._env.result().metrics if False else env._env.done


def test_invalid_action_is_impossible():  # E-12
    env = _env()
    env.reset()
    for _ in range(500):
        m = env.action_masks()
        if not m.all():
            bad = int(np.flatnonzero(~m)[0])
            with pytest.raises(ValueError):
                env.step(bad)
            return
        env.step(1)
    pytest.skip("no masked state encountered")


# --------------------------------------------------------------------------- no baseline leakage
def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    return out | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}


def _code_text(path: Path) -> str:
    """Identifiers, attributes and non-docstring string constants (docstrings/comments excluded)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docs = {id(n.body[0].value) for n in ast.walk(tree)
            if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant)}
    parts = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            parts.append(n.id)
        elif isinstance(n, ast.Attribute):
            parts.append(n.attr)
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            parts.append(n.name)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs:
            parts.append(n.value)
    return " ".join(parts) + " ("


def test_training_modules_cannot_access_baseline_logic():  # E-13, E-14, E-16
    for name in ("training.py", "smdp.py", "env.py", "scenarios.py"):
        imps = _imports(PHASE_E / name)
        assert not [m for m in imps if m.startswith(("callverse_workforce.baselines", "callverse_workforce.headroom"))], (name, imps)
        src = _code_text(PHASE_E / name)
        for token in ("rule_action", "aifirst_action", "threshold_action", "make_policy", "RulePolicy(",
                      "imitation", "behavior_cloning", "VecNormalize"):
            assert token not in src, (name, token)


def test_reward_scale_fixed_and_no_vecnormalize():  # E-15, E-16
    assert REWARD_SCALE == 100.0
    for p in PHASE_E.glob("*.py"):
        assert "VecNormalize" not in p.read_text(encoding="utf-8") or p.name == "report.py"


# --------------------------------------------------------------------------- SMDP analytics
def test_smdp_discounts_return_gae_and_terminal():  # E-19..E-22
    dt = np.array([[1.0], [10.0]])
    r = np.array([[2.0], [3.0]])
    v = np.array([[5.0], [4.0]])
    starts = np.array([[1.0], [0.0]])
    disc = transition_discounts(dt, "smdp")
    assert disc[0, 0] == pytest.approx(0.995 ** 1) and disc[1, 0] == pytest.approx(0.995 ** 10)
    lam, last_v = 0.95, np.array([7.0])
    # second transition is the true end of day: no bootstrap
    adv, ret = smdp_gae(r, v, starts, disc, last_v, np.array([True]), lam)
    d1 = 3.0 - 4.0
    d0 = 2.0 + 0.995 * 4.0 - 5.0
    assert adv[1, 0] == pytest.approx(d1)
    assert adv[0, 0] == pytest.approx(d0 + 0.995 * lam * d1)
    assert ret[1, 0] == pytest.approx(3.0)  # Monte-Carlo-exact at the terminal step
    # not terminal: bootstrap with gamma^10 * V(s_next)
    adv2, _ = smdp_gae(r, v, starts, disc, last_v, np.array([False]), lam)
    assert adv2[1, 0] == pytest.approx(3.0 + 0.995 ** 10 * 7.0 - 4.0)
    # lambda = 1 gives the discounted SMDP return R0 = r0 + g^1 * r1
    _, ret1 = smdp_gae(r, v, starts, disc, last_v, np.array([True]), 1.0)
    assert ret1[0, 0] == pytest.approx(2.0 + 0.995 * 3.0)
    # mutation: a fixed per-transition gamma gives a different (wrong) answer
    fixed = transition_discounts(dt, "fixed", gamma_fixed=0.995)
    adv_wrong, _ = smdp_gae(r, v, starts, fixed, last_v, np.array([False]), lam)
    assert adv_wrong[1, 0] != pytest.approx(adv2[1, 0])


def test_episode_boundary_inside_rollout_has_no_bootstrap():
    r = np.array([[1.0], [1.0], [1.0]])
    v = np.array([[0.5], [9.0], [0.5]])
    starts = np.array([[0.0], [1.0], [0.0]])  # a new day starts at step 1: step 0 ends a day
    disc = transition_discounts(np.array([[2.0], [2.0], [2.0]]), "smdp")
    adv, _ = smdp_gae(r, v, starts, disc, np.array([0.0]), np.array([True]), 0.95)
    assert adv[0, 0] == pytest.approx(1.0 - 0.5)  # value of the next day's first state is NOT used


def test_myopic_and_transition_gamma():  # E-23, E-24
    r = np.array([[2.0], [3.0]])
    v = np.array([[5.0], [4.0]])
    disc0 = transition_discounts(np.array([[1.0], [10.0]]), "fixed", gamma_fixed=0.0)
    adv, ret = smdp_gae(r, v, np.zeros((2, 1)), disc0, np.array([7.0]), np.array([False]), 0.95)
    assert np.allclose(ret, r) and np.allclose(adv, r - v)
    plan = OUT / "compute_plan.json"
    if plan.exists():
        p = json.loads(plan.read_text(encoding="utf-8"))
        assert p["transition_ppo_gamma_fixed"] == pytest.approx(0.995 ** p["mean_training_delta_t_minutes"])
        assert p["myopic_gamma"] == 0.0


def test_buffer_and_model_store_delta_t():
    env = _env(11)
    model = SMDPMaskablePPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, seed=0, device="cpu")
    model.learn(64)
    buf = model.rollout_buffer
    assert isinstance(buf, SMDPRolloutBuffer) and buf.delta_t.shape == (64, 1) and buf.delta_t.sum() > 0
    disc = transition_discounts(buf.delta_t, "smdp")
    assert np.all((disc > 0) & (disc <= 1))
    with pytest.raises(ValueError):
        buf.add(np.zeros((1, 43)), np.zeros((1, 1)), np.zeros(1), np.zeros(1), __import__("torch").zeros(1), __import__("torch").zeros(1))


# --------------------------------------------------------------------------- protocol
def test_hyperparameter_grid_and_seed_counts():  # E-25..E-27, E-29, E-30
    assert len(TR.GRID) == 8
    assert {(g["learning_rate"], tuple(g["net_arch"]), g["ent_coef"]) for g in TR.GRID} == set(
        itertools.product((1e-4, 3e-4), ((64, 64), (128, 128)), (0.0, 0.01)))
    assert len(RUN.SEARCH_SEEDS) == 2 and len(RUN.FULL_SEEDS) == 5 and len(RUN.ABLATION_SEEDS) == 3
    assert TR.SEARCH_BUDGET < TR.FULL_BUDGET
    specs = [RUN._spec("x", h, s, TR.SEARCH_BUDGET, "smdp", None, "tmp") for h in TR.GRID for s in RUN.SEARCH_SEEDS]
    assert len(specs) == 16 and {s.budget for s in specs} == {TR.SEARCH_BUDGET}
    assert all({d[2] for d in s.eval_days} <= set(PPO_VALIDATION_SEEDS) for s in specs)  # E-33


def test_seed_sets_disjoint_and_final_untouched():  # E-31, E-32
    for (a, sa), (b, sb) in itertools.combinations(SEED_SETS.items(), 2):
        assert not set(sa) & set(sb), (a, b)
    assert FINAL_TEST_RESERVED == range(2000, 3000)
    assert min(RUN.AUDIT_SEEDS) >= 100_000
    days = SC.fixed_days(PPO_VALIDATION_SEEDS) + SC.fixed_days(PPO_DEV_EVAL_SEEDS)
    assert len({d.seed for d in days}) == 160
    assert [d.family for d in SC.fixed_days(PPO_DEV_EVAL_SEEDS)][::10] == list(SC.FAMILIES)


def test_dev_days_inaccessible_for_validation_and_vice_versa():  # E-34
    with pytest.raises(SeedPolicyError):
        EV.days_for("validation", SC.fixed_days(PPO_DEV_EVAL_SEEDS))
    with pytest.raises(SeedPolicyError):
        EV.days_for("dev", SC.fixed_days(PPO_VALIDATION_SEEDS))
    with pytest.raises(ValueError):
        SC.fixed_days(range(2000, 2080)[:79])


def test_checkpoint_selection_avoids_isolated_spike():  # E-33
    curve = [{"step": s, "checkpoint": f"c{s}", "val_subset_mean_raw_reward": v}
             for s, v in zip(range(1, 8), [100, 110, 200, 100, 150, 160, 150])]
    sel = TR.select_checkpoint(curve)
    assert sel["step"] == 7 and sel["raw"] == 150  # the isolated raw maximum (step 3, 200) is not selected
    assert TR.select_checkpoint(curve[:1])["step"] == 1


def test_protocol_order_artifacts():  # E-28, E-35
    sel, verdict_rule = OUT / "selected_models" / "selected_config.json", OUT / "verdict_rule.json"
    if verdict_rule.exists() and (OUT / "hyperparameter_search").exists():
        runs = list((OUT / "hyperparameter_search").rglob("run.json"))
        assert all(os.path.getmtime(verdict_rule) < os.path.getmtime(r) for r in runs)
    if sel.exists():
        full_runs = list((OUT / "training").rglob("run.json")) + list((OUT / "ablation").rglob("run.json"))
        assert all(os.path.getmtime(sel) < os.path.getmtime(r) for r in full_runs)
    dev = OUT / "development_evaluation" / "dev_opened.json"
    if dev.exists():
        assert os.path.getmtime(verdict_rule) < os.path.getmtime(dev)
        assert os.path.getmtime(OUT / "selected_models" / "selected_checkpoints.json") < os.path.getmtime(dev)


# --------------------------------------------------------------------------- statistics and verdict
def _rows(values, seeds=None):
    seeds = seeds or list(range(1601, 1601 + len(values)))
    return [{"seed": s, "total_reward": v} for s, v in zip(seeds, values)]


def test_paired_day_means_and_verdict_classes():  # E-36..E-41, E-43
    rng = np.random.default_rng(0)
    rule = {1601 + i: 1600 + rng.normal(0, 50) for i in range(80)}
    seeds5 = [_rows([rule[d] + rng.normal(-5, 10) for d in sorted(rule)]) for _ in range(5)]
    mean = EV.ppo_mean_by_day(seeds5)
    assert len(mean) == 80
    d0 = sorted(mean)[0]
    assert mean[d0] == pytest.approx(np.mean([s[0]["total_reward"] for s in seeds5]))
    random_ = {d: 1000.0 for d in rule}
    aifirst = {d: rule[d] - 30 for d in rule}
    v = EV.verdict(mean, rule, random_, aifirst)
    assert v["ppo_minus_rule"]["n_days"] == 80  # 80 paired days, not 400
    assert v["margin"] == pytest.approx(-0.02 * np.mean(list(rule.values())))
    assert v["verdict"] == EV.SUCCESS
    worse = {d: rule[d] - 80 + rng.normal(0, 5) for d in rule}  # -5% but above Random and AIFirst
    assert EV.verdict(worse, rule, random_, {d: rule[d] - 120 for d in rule})["verdict"] == EV.PARTIAL
    bad = {d: 900.0 for d in rule}
    assert EV.verdict(bad, rule, random_, aifirst)["verdict"] == EV.FAILURE
    with pytest.raises(ValueError):
        EV.paired_diff(mean, {k: v for k, v in list(rule.items())[:79]})


def test_seed_noninferiority_count():  # E-42
    rule = {1601 + i: 1600.0 + i for i in range(80)}
    good = _rows([rule[d] - 1 for d in sorted(rule)])
    bad = _rows([rule[d] - 100 for d in sorted(rule)])
    res = EV.seed_noninferiority([good, good, good, bad, bad], rule)
    assert res["noninferior_seed_count"] == 3 and res["n_seeds"] == 5


def test_config_selection_rule():
    cands = [{"index": i, "mean_val_reward": m, "seed_means": [m, m]} for i, m in enumerate([10.0, 30.0, 30.0, 5.0])]
    sel = EV.select_config(cands, random_val_mean=8.0)
    assert sel["selected"]["index"] == 1 and sel["excluded"] == [3]


def test_phase_d_outputs_still_reproducible():  # E-45
    path = PROJECT_ROOT / "artifacts" / "v1_1" / "baselines" / "evaluation_v1_1_cells.json"
    if not path.exists():
        pytest.skip("Phase D artifacts missing")
    ev = json.loads(path.read_text(encoding="utf-8"))
    from callverse_workforce.baselines.policies import make_policy
    from callverse_workforce.phase_d.grid import cell_config
    from callverse_workforce.twin import run_simulation

    frozen = json.loads((PROJECT_ROOT / "artifacts" / "v1_1" / "baselines" / "frozen" / "frozen_v1_1_policies.json")
                        .read_text(encoding="utf-8"))["frozen"]
    r = run_simulation(cell_config(BASE, MAX_CELL), make_policy("RulePolicy", frozen["RulePolicy"], 1421), seed=1421)
    assert r.total_reward == ev["table"][MAX_CELL.cell_id]["RulePolicy"]["per_seed_reward"]["1421"]
