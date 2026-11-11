"""Phase E.1: training-only stabilisation. Problem frozen, optimiser changed, protocol guarded."""

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

from callverse_workforce.config import load_config  # noqa: E402
from callverse_workforce.config.loader import PROJECT_ROOT  # noqa: E402
from callverse_workforce.domain.enums import ACTION_ORDER  # noqa: E402
from callverse_workforce.phase_d.grid import MIXTURE_CELLS  # noqa: E402
from callverse_workforce.phase_e import evaluation as EV  # noqa: E402
from callverse_workforce.phase_e import scenarios as SC  # noqa: E402
from callverse_workforce.phase_e.env import REWARD_SCALE, CallVerseRoutingEnv  # noqa: E402
from callverse_workforce.phase_e.smdp import SMDPMaskablePPO, SMDPRolloutBuffer  # noqa: E402
from callverse_workforce.phase_e1 import evaluation as E1  # noqa: E402
from callverse_workforce.phase_e1 import training as T  # noqa: E402
from callverse_workforce.seeds import (  # noqa: E402
    FINAL_TEST_RESERVED,
    PPO_DEV_EVAL_SEEDS,
    PPO_E1_DEV_SEEDS,
    PPO_VALIDATION_SEEDS,
    SEED_SETS,
    SeedPolicyError,
)

BASE = load_config()
OUT = PROJECT_ROOT / "artifacts" / "ppo_v1_1_e1"
PHASE_E = PROJECT_ROOT / "artifacts" / "ppo_v1_1"
CFG_PATH = str(PROJECT_ROOT / "configs" / "normal_day.yaml")
_spec = importlib.util.spec_from_file_location("run_phase_e1", PROJECT_ROOT / "scripts" / "run_phase_e1.py")
RUN = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RUN)


def _sha(p: Path) -> str:
    import hashlib

    return hashlib.sha256(p.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- frozen problem
def test_phase_e_artifacts_and_verdict_preserved():  # E1-1
    if not (PHASE_E / "development_evaluation" / "verdict.json").exists():
        pytest.skip("Phase E artifacts missing")
    v = json.loads((PHASE_E / "development_evaluation" / "verdict.json").read_text(encoding="utf-8"))
    assert v["verdict"] == EV.FAILURE
    cfg = OUT / "training_config.json"
    if cfg.exists():
        fp = json.loads(cfg.read_text(encoding="utf-8"))["frozen_problem"]
        assert fp["environment_definition_sha256"] == _sha(PHASE_E / "environment_definition.json")
        assert fp["scenario_registry_sha256"] == _sha(PHASE_E / "scenario_registry.json")
        assert os.path.getmtime(PHASE_E / "development_evaluation" / "verdict.json") < os.path.getmtime(cfg)


def test_twin_reward_and_scenarios_unchanged():  # E1-2, E1-3, E1-7, E1-8
    reg = json.loads((PHASE_E / "scenario_registry.json").read_text(encoding="utf-8"))["configs"]
    cells = {c.cell_id: c for c in MIXTURE_CELLS}
    for key, entry in reg.items():
        fam, cid = key.split("|")
        cfg = SC.day_config(BASE, fam, cells[cid])
        assert SC.config_hash(cfg) == entry["sha256"] and cfg.reward == BASE.reward
    env_def = json.loads((PHASE_E / "environment_definition.json").read_text(encoding="utf-8"))
    assert SC.TRAINING_DISTRIBUTION == env_def["training_distribution"] and SC.TRAINING_DISTRIBUTION["WORST_DAY"] == 0.03
    assert REWARD_SCALE == 100.0


def test_phase_e_dev_result_still_reproduces():  # E1-2
    per_day = PHASE_E / "development_evaluation" / "per_day.json"
    if not per_day.exists():
        pytest.skip("Phase E artifacts missing")
    rows = json.loads(per_day.read_text(encoding="utf-8"))["RulePolicy"]
    r0 = rows[0]
    frozen = json.loads((PROJECT_ROOT / "artifacts" / "v1_1" / "baselines" / "frozen" / "frozen_v1_1_policies.json")
                        .read_text(encoding="utf-8"))["frozen"]
    out = EV.eval_job(("RulePolicy", "baseline", ("RulePolicy", frozen["RulePolicy"]), CFG_PATH,
                       (r0["family"], r0["cell"], r0["seed"])))
    assert out["total_reward"] == r0["total_reward"]


def test_masks_observation_and_actions_unchanged():  # E1-4, E1-5, E1-6, E1-41
    env = CallVerseRoutingEnv(BASE, SC.TrainingDaySampler(1))
    obs, _ = env.reset(options={"day": SC.DaySpec("WORST_DAY", MIXTURE_CELLS[3], 100_007)})
    assert obs.shape == (43,) and obs.dtype == np.float32 and env.action_space.n == 4
    assert [a.value for a in ACTION_ORDER] == ["AI", "HUMAN", "WAIT", "CALLBACK"]
    for _ in range(50):
        assert tuple(env.action_masks()) == env._dp.action_mask.valid  # the unchanged V1 minimal masks
        obs, *_ = env.step(1)
    # Amended (V2 Stage 1, approved): exclude exactly the separately specified V2 package; every other module, including
    # all V1.1 / E.1 code, is still scanned. Import isolation of that package is tested in test_v2_isolation.py.
    v2_dir = (PROJECT_ROOT / "src" / "callverse_workforce" / "v2_postfail").resolve()
    files = [p for p in (PROJECT_ROOT / "src").rglob("*.py") if v2_dir not in p.resolve().parents]
    src = "".join(p.read_text(encoding="utf-8") for p in files)
    assert "HUMAN_HIGH" not in src and "HUMAN_NORMAL" not in src  # no V2 action in the V1.1 / E.1 code


# --------------------------------------------------------------------------- pre-registration and protocol order
def test_preregistration_before_training_and_plan_before_training():  # E1-9, E1-28
    h = OUT / "preregistration_hash.json"
    if not h.exists():
        pytest.skip("E.1 not started")
    assert json.loads(h.read_text(encoding="utf-8"))["sha256"] == _sha(PROJECT_ROOT / "docs" / "phase_e1_stabilization_prereg.md")
    plan = OUT / "compute_plan.json"
    assert json.loads(plan.read_text(encoding="utf-8"))["checkpoint_every"] == 50_000
    for p in (OUT / "training").rglob("*"):
        if p.is_file():
            assert os.path.getmtime(h) < os.path.getmtime(p) and os.path.getmtime(plan) < os.path.getmtime(p)


def test_selection_and_dev_order():  # E1-30, E1-31
    sel, dev = OUT / "selected_models" / "selected_checkpoints.json", OUT / "development_evaluation" / "dev_opened.json"
    if dev.exists():
        assert os.path.getmtime(sel) < os.path.getmtime(dev)
    with pytest.raises(SeedPolicyError):
        E1.evaluate_checkpoints(["x.zip"], SC.fixed_days(PPO_E1_DEV_SEEDS), CFG_PATH, 1)
    with pytest.raises(SeedPolicyError):
        E1.evaluate_checkpoints(["x.zip"], SC.fixed_days(PPO_DEV_EVAL_SEEDS), CFG_PATH, 1)
    with pytest.raises(SeedPolicyError):
        E1.evaluate({}, SC.fixed_days(PPO_DEV_EVAL_SEEDS), "e1_dev", CFG_PATH, 1)  # E dev is closed for E.1


# --------------------------------------------------------------------------- training configuration
@pytest.fixture(scope="module")
def model():
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda s=s: T.make_env(CFG_PATH, s) for s in (11, 12, 13, 14, 15, 16, 17, 18)])
    return T.build_model(venv, 501)


def test_ppo_configuration(model):  # E1-10, E1-13..E1-16, E1-19, E1-22..E1-25
    assert T.N_ENVS == 8 and model.n_envs == 8 and model.n_steps == 2048
    assert T.ROLLOUT == 16_384 and model.rollout_buffer.buffer_size * model.n_envs == 16_384
    assert model.batch_size == 512 and T.ROLLOUT // model.batch_size == 32
    assert model.n_epochs == 5 and model.target_kl == 0.015 and model.gae_lambda == 0.95
    assert isinstance(model, SMDPMaskablePPO) and isinstance(model.rollout_buffer, SMDPRolloutBuffer)
    assert model.rollout_buffer.discount_mode == "smdp" and model.rollout_buffer.gamma_minute == 0.995
    pi = [m.out_features for m in model.policy.mlp_extractor.policy_net if hasattr(m, "out_features")]
    vf = [m.out_features for m in model.policy.mlp_extractor.value_net if hasattr(m, "out_features")]
    assert pi == [64, 64] and vf == [128, 128]


def test_learning_rate_and_entropy_schedules():  # E1-17, E1-18, E1-20, E1-21
    assert T.lr_schedule(1.0) == pytest.approx(1e-4)
    assert T.lr_schedule(0.5) == pytest.approx(5e-5)
    assert T.lr_schedule(0.0) == pytest.approx(0.0)
    assert T.ent_schedule(1.0) == pytest.approx(0.01)
    assert T.ent_schedule(0.5) == pytest.approx(0.005)
    assert T.ent_schedule(0.0) == pytest.approx(0.0)


def test_entropy_schedule_applied_during_training():
    from stable_baselines3.common.vec_env import DummyVecEnv

    venv = DummyVecEnv([lambda s=s: T.make_env(CFG_PATH, s) for s in range(2)])
    m = T.ScheduledSMDPMaskablePPO("MlpPolicy", venv, learning_rate=T.lr_schedule, ent_coef=0.01, n_steps=64,
                                   batch_size=64, n_epochs=1, seed=0, device="cpu", target_kl=T.TARGET_KL)
    m.learn(256)
    assert m.ent_coef == pytest.approx(T.ent_schedule(m._current_progress_remaining))
    assert m.ent_coef < 0.01


def test_workers_independent():  # E1-11, E1-12, E1-26
    seeds = {s: [T.worker_sampler_seed(s, w) for w in range(8)] for s in T.E1_SEEDS}
    flat = [x for v in seeds.values() for x in v]
    assert len(set(flat)) == 40
    days = [[SC.TrainingDaySampler(x)().seed for _ in range(3)] for x in seeds[500]]
    assert len({tuple(d) for d in days}) == 8  # workers do not replay the same days
    e1, e2 = T.make_env(CFG_PATH, seeds[500][0]), T.make_env(CFG_PATH, seeds[500][1])
    e1.reset(), e2.reset()
    assert e1._env is not e2._env and e1._env.twin is not e2._env.twin and e1.day != e2.day
    assert not set(T.E1_SEEDS) & {0, 1, 100, 101, 102, 103, 104, 200, 201, 202}


def test_budget():  # E1-27
    assert T.TOTAL_TIMESTEPS <= 3_000_000 and T.TOTAL_TIMESTEPS % T.ROLLOUT == 0
    assert T.TOTAL_TIMESTEPS == 183 * 16_384
    for s in T.E1_SEEDS:
        run = OUT / "training" / f"e1_s{s}" / "run.json"
        if run.exists():
            assert json.loads(run.read_text(encoding="utf-8"))["num_timesteps"] == T.TOTAL_TIMESTEPS


def test_checkpoint_selection_rule_and_80_days():  # E1-29
    ev = [{"actual_step": s, "val_mean_raw_reward": v} for s, v in ((50, 1500.0), (100, 1530.4), (150, 1531.0), (200, 1529.0))]
    sel = T.select_checkpoint(ev)
    assert sel["actual_step"] == 100 and sel["best_val_mean"] == 1531.0  # within 1.0 of the max -> earliest
    for s in T.E1_SEEDS:
        curve = OUT / "validation" / f"e1_s{s}_curve.json"
        if curve.exists():
            assert all(c["n_days"] == 80 for c in json.loads(curve.read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- seeds
def test_seed_sets():  # E1-32, E1-33, E1-34
    assert PPO_E1_DEV_SEEDS == tuple(range(1681, 1761))
    for (a, sa), (b, sb) in itertools.combinations(SEED_SETS.items(), 2):
        assert not set(sa) & set(sb), (a, b)
    assert FINAL_TEST_RESERVED == range(2000, 3000)
    days = SC.fixed_days(PPO_E1_DEV_SEEDS)
    assert [d.family for d in days][::10] == list(SC.FAMILIES) and len(days) == 80


# --------------------------------------------------------------------------- no baseline leakage
def test_training_module_has_no_baseline_access():  # E1-35
    tree = ast.parse((PROJECT_ROOT / "src" / "callverse_workforce" / "phase_e1" / "training.py").read_text(encoding="utf-8"))
    mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert not [m for m in mods if m.startswith(("callverse_workforce.baselines", "callverse_workforce.headroom"))]


# --------------------------------------------------------------------------- verdict and stability
def _rows(vals):
    return [{"seed": 1681 + i, "total_reward": v} for i, v in enumerate(vals)]


def test_same_verdict_implementation_and_80_days():  # E1-36, E1-37
    rng = np.random.default_rng(1)
    rule = {1681 + i: 1550 + rng.normal(0, 40) for i in range(80)}
    seeds = [_rows([rule[d] + rng.normal(-3, 8) for d in sorted(rule)]) for _ in range(5)]
    mean = EV.ppo_mean_by_day(seeds)
    rand, aif = {d: 900.0 for d in rule}, {d: rule[d] - 10 for d in rule}
    v_e1, v_e = E1.primary_verdict(mean, rule, rand, aif), EV.verdict(mean, rule, rand, aif)
    assert v_e1["ppo_minus_rule"] == v_e["ppo_minus_rule"] and v_e1["margin"] == v_e["margin"]
    assert v_e1["ppo_minus_rule"]["n_days"] == 80 and v_e1["verdict"] == E1.SUCCESS
    assert E1.LABEL_MAP[EV.PARTIAL] == E1.PARTIAL and E1.LABEL_MAP[EV.FAILURE] == E1.FAILURE


def test_stability_criterion():  # E1-38, E1-39, E1-40
    rule = {1681 + i: 1500.0 + i for i in range(80)}
    good = lambda off: _rows([rule[d] + off for d in sorted(rule)])
    s = E1.stability([good(-1), good(-2), good(0), good(1), good(-3)], rule)
    assert s["noninferior_seed_count"] == 5 and s["seed_sd"] < 13.5 and s["status"] == "STABLE"
    s3 = E1.stability([good(-1), good(-2), good(0), good(-100), good(-90)], rule)
    assert s3["noninferior_seed_count"] == 3 and s3["status"] == "NOT STABLE"
    spread = E1.stability([good(25), good(-35), good(0), good(20), good(-20)], rule)  # 4/5 non-inferior, SD ~24
    assert spread["noninferior_seed_count"] == 4 and spread["seed_sd"] > 13.5 and spread["status"] == "NOT STABLE"
    assert E1.STABLE_MAX_SD == 13.5 and E1.STABLE_MIN_SEEDS == 4
    assert spread["seed_range"] == pytest.approx(60.0)


def test_interpretation_sentences():
    assert "both competitive and stable" in E1.interpretation(E1.SUCCESS, "STABLE")
    assert "seed sensitivity persists" in E1.interpretation(E1.SUCCESS, "NOT STABLE")
    assert "limited structural headroom" in E1.interpretation(E1.FAILURE, "STABLE")
    assert "motivation for a V2" in E1.interpretation(E1.PARTIAL, "NOT STABLE")
