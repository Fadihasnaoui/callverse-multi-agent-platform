"""V1.1 final test: protocol, seed split, frozen components, statistics and labels (no final-test seed simulated here)."""

from __future__ import annotations

import ast
import hashlib
import json

import numpy as np
import pytest

from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.final_test import v1_1 as FT
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.phase_e import scenarios as SC
from callverse_workforce.phase_e1 import evaluation as E1
from callverse_workforce.seeds import (
    FINAL_TEST_RESERVED,
    FINAL_V1_1_SEEDS,
    V2_RESERVED_SEEDS,
    SeedPolicyError,
    assert_v2_reserved_untouched,
)

OUT = PROJECT_ROOT / "artifacts" / "final_test_v1_1"
PREREG = PROJECT_ROOT / "docs" / "final_test_v1_1_prereg.md"
E1_DIR = PROJECT_ROOT / "artifacts" / "ppo_v1_1_e1"
SCRIPT = PROJECT_ROOT / "scripts" / "run_final_test_v1_1.py"
MODULES = (SCRIPT, PROJECT_ROOT / "src/callverse_workforce/final_test/v1_1.py", PROJECT_ROOT / "src/callverse_workforce/final_test/report.py")
needs_freeze = pytest.mark.skipif(not (OUT / "integrity_checks.json").exists(), reason="final test not frozen")


def _r(p):
    return json.loads(p.read_text(encoding="utf-8"))


def _rows(label, rewards, families=None):
    return [{"label": label, "seed": 2000 + i, "total_reward": float(r), "family": (families or ["NORMAL"] * len(rewards))[i]}
            for i, r in enumerate(rewards)]


# --- 1-2 pre-registration ---------------------------------------------------------------------------------
@needs_freeze
def test_01_prereg_exists_before_test_artifacts():
    pre = _r(OUT / "preregistration_hash.json")
    assert PREREG.exists() and pre["final_test_seeds_simulated_before"] == 0
    flag = OUT / "raw_results" / "final_test_opened.json"
    if flag.exists():
        assert _r(flag)["opened_at_utc"] > pre["frozen_at_utc"]
        assert _r(OUT / "integrity_checks.json")["checked_at_utc"] < _r(flag)["opened_at_utc"]


@needs_freeze
def test_02_prereg_hash_correct():
    assert _r(OUT / "preregistration_hash.json")["sha256"] == hashlib.sha256(PREREG.read_bytes()).hexdigest()


# --- 3-9 scenarios and seeds -------------------------------------------------------------------------------
def test_03_canonical_scenario_registry_reused():
    reg = _r(PROJECT_ROOT / "artifacts" / "ppo_v1_1" / "environment_definition.json")
    assert list(SC.FAMILIES) == reg["families"]
    assert all(d.family in SC.FAMILIES for d in FT.final_days())


def test_04_e1_day_generation_reused():
    src = (PROJECT_ROOT / "src/callverse_workforce/final_test/v1_1.py").read_text(encoding="utf-8")
    assert "EV._config" in src and "E1.eval_job" in src  # same day_config + same E.1 job


def test_05_mixture_logic_reused():
    days = FT.final_days()
    assert all(d.cell in MIXTURE_CELLS for d in days)
    assert [d.cell for d in days[:4]] == list(MIXTURE_CELLS)
    for fam in SC.FAMILIES:
        cells = [d.cell.cell_id for d in days if d.family == fam]
        assert [cells.count(c.cell_id) for c in MIXTURE_CELLS] == [8, 8, 7, 7]


def test_06_thirty_seeds_per_family():
    days = FT.final_days()
    assert {f: sum(d.family == f for d in days) for f in SC.FAMILIES} == {f: 30 for f in SC.FAMILIES}


def test_07_total_240_days():
    assert len(FT.final_days()) == 240 and len({d.seed for d in FT.final_days()}) == 240


def test_08_seed_range_exact():
    assert [d.seed for d in FT.final_days()] == list(range(2000, 2240)) == list(FINAL_V1_1_SEEDS)
    with pytest.raises(ValueError):
        FT.final_days(range(2001, 2241))
    assert FINAL_TEST_RESERVED == range(2000, 3000)


def test_09_v2_reserve_inaccessible():
    assert V2_RESERVED_SEEDS == tuple(range(2240, 3000)) and not set(V2_RESERVED_SEEDS) & set(FINAL_V1_1_SEEDS)
    with pytest.raises(SeedPolicyError):
        assert_v2_reserved_untouched([2240])
    spec = SC.DaySpec("NORMAL", MIXTURE_CELLS[0], 2500)
    with pytest.raises(SeedPolicyError):
        EV.days_for("final_v1_1", [spec])
    with pytest.raises(SeedPolicyError):
        EV.days_for("e1_dev", [SC.DaySpec("NORMAL", MIXTURE_CELLS[0], 2000)])


# --- 10-11 collision audit ---------------------------------------------------------------------------------
def test_10_collision_audit_detects_prior_usage(monkeypatch):
    monkeypatch.setattr(FT, "consumed_seed_sources", lambda root: {"fake_debug_run": {2005, 2300, 1501}})
    a = FT.collision_audit(PROJECT_ROOT)
    assert a["collisions_final"] == {"fake_debug_run": [2005]} and a["n_collisions_final"] == 1
    assert a["collisions_v2_reserved"] == {"fake_debug_run": [2300]}


@needs_freeze
def test_11_zero_collisions_required():
    a = _r(OUT / "seed_collision_audit.json")
    assert a["n_collisions_final"] == 0 and a["n_collisions_v2"] == 0
    assert "final_test_reserved" not in a["sources_checked"]  # the reserved range itself is not "consumption"
    src = SCRIPT.read_text(encoding="utf-8")
    assert "SEED COLLISION — STOP FOR PROTOCOL REVIEW" in src and "choose_block" not in src  # no substitution


# --- 12-15 frozen models / baselines -------------------------------------------------------------------------
@needs_freeze
def test_12_ppo_hashes_match_e1_frozen():
    sel = _r(E1_DIR / "selected_models" / "selected_checkpoints.json")["runs"]
    mh = _r(OUT / "model_hashes.json")
    assert sorted(mh["models"]) == [f"e1_s{s}" for s in range(500, 505)]
    for rid, v in sel.items():
        assert mh["models"][rid]["sha256"] == hashlib.sha256(open(v["path"], "rb").read()).hexdigest()
        assert mh["models"][rid]["selected_step"] == v["actual_step"]


@needs_freeze
def test_13_no_checkpoint_change():
    mh = _r(OUT / "model_hashes.json")
    sel_sha = hashlib.sha256((E1_DIR / "selected_models" / "selected_checkpoints.json").read_bytes()).hexdigest()
    assert mh["selected_checkpoints_sha256"] == sel_sha == mh["e1_dev_opened_selected_checkpoints_sha256"]
    assert "checkpoint changed after freezing" in SCRIPT.read_text(encoding="utf-8")


@needs_freeze
def test_14_deterministic_convention_matches_e1():
    integ = _r(OUT / "integrity_checks.json")
    assert all(v["match"] for v in integ["e1_dev_day_reproduction"].values())
    src = (PROJECT_ROOT / "src/callverse_workforce/phase_e/training.py").read_text(encoding="utf-8")
    assert "deterministic=True" in src


@needs_freeze
def test_15_baseline_hashes_match_frozen():
    bh = _r(OUT / "baseline_hashes.json")
    frozen = PROJECT_ROOT / bh["frozen_file"]
    assert bh["sha256"] == hashlib.sha256(frozen.read_bytes()).hexdigest()
    f = _r(frozen)["frozen"]
    assert bh["policies"]["RulePolicy"][1] == f["RulePolicy"] and bh["policies"]["ThresholdPolicy"][1] == f["ThresholdPolicy"]


# --- 16-18 pairing / independence / per-day averaging ---------------------------------------------------------
def test_16_same_day_randomness_across_policies():
    src = (PROJECT_ROOT / "src/callverse_workforce/phase_e1/evaluation.py").read_text(encoding="utf-8")
    assert "run_simulation(cfg, EV._policy(kind, ref, seed), seed=seed)" in src  # the day seed drives all exogenous streams
    jobs = [(lab, d.seed) for lab in ("a", "b") for d in FT.final_days()]
    assert {s for l, s in jobs if l == "a"} == {s for l, s in jobs if l == "b"}


def test_17_ppo_models_independent():
    usage = [FT.code_usage(p) for p in MODULES]
    calls = set().union(*(u["calls"] for u in usage))
    assert not calls & {"load_state_dict", "state_dict", "vote", "ensemble"}


def test_18_ppo_average_per_day():
    a, b = _rows("x", [10, 20]), _rows("y", [30, 40])
    assert EV.ppo_mean_by_day([a, b]) == {2000: 20.0, 2001: 30.0}


# --- 19-23 primary statistics -----------------------------------------------------------------------------
def _fake_res(delta=0.0, n=240, spread=1.0, aif_agree=0.9):
    rng = np.random.default_rng(0)
    rule = 1500 + rng.normal(0, 30, n)
    res = {"RulePolicy": _rows("RulePolicy", rule)}
    for name, off in (("AIFirstPolicy", -5), ("ThresholdPolicy", -2), ("BestFixedPolicy", -5), ("RandomPolicy", -400)):
        res[name] = _rows(name, rule + off + rng.normal(0, 5, n))
    for i in range(5):
        rows = _rows(f"p{i}", rule + delta + spread * (i - 2) + rng.normal(0, 5, n))
        for r in rows:
            r["agreement"] = {"decisions": 100, "agree_aifirst": int(100 * aif_agree), "agree_rule": 80, "exact_aifirst_trajectory": False}
        res[f"p{i}"] = rows
    return res


def test_19_primary_ci_uses_240_differences():
    st = FT.statistics(_fake_res(), [f"p{i}" for i in range(5)])
    assert st["n_days"] == 240 and st["ppo_minus_rule"]["n_days"] == 240
    bad = _fake_res(n=239)
    with pytest.raises(RuntimeError):
        FT.statistics(bad, [f"p{i}" for i in range(5)])


def test_20_bootstrap_unit_is_day():
    src = FT.statistics.__code__.co_names
    assert "ppo_mean_by_day" in src and "paired_bootstrap" in src
    d = EV.paired_bootstrap([1.0] * 240, key=1)
    assert d["n_days"] == 240


def test_21_margin_is_minus_2pct():
    st = FT.statistics(_fake_res(), [f"p{i}" for i in range(5)])
    assert st["margin"] == pytest.approx(-0.02 * st["rule_mean_reward"])


def test_22_final_verdict_labels():
    labels = [f"p{i}" for i in range(5)]
    assert FT.statistics(_fake_res(delta=0.0), labels)["verdict"] == FT.NONINF == "V1.1 FINAL — PPO NON-INFERIOR TO RULEPOLICY"
    assert FT.statistics(_fake_res(delta=-60.0), labels)["verdict"] == FT.NOT_NONINF == "V1.1 FINAL — PPO NOT NON-INFERIOR TO RULEPOLICY"


def test_23_individual_seed_noninferiority():
    st = FT.statistics(_fake_res(delta=-25.0, spread=10.0), [f"p{i}" for i in range(5)])
    per = st["stability"]["per_seed"]
    assert [p["noninferior"] for p in per] == [p["lo"] > st["margin"] for p in per]


# --- 24-27 stability and closeness ---------------------------------------------------------------------------
def test_24_25_e1_stability_rule_and_threshold_reused():
    assert E1.STABLE_MIN_SEEDS == 4 and E1.STABLE_MAX_SD == 13.5
    assert "stability" in FT.statistics.__code__.co_names
    st = FT.statistics(_fake_res(spread=1.0), [f"p{i}" for i in range(5)])
    assert st["stability_label"] == "TRAINING STABILITY: STABLE"
    st = FT.statistics(_fake_res(spread=12.0), [f"p{i}" for i in range(5)])  # SD ~ 19 > 13.5
    assert st["stability_label"] == "TRAINING STABILITY: NOT STABLE"


def test_26_close_to_aifirst_statistical_condition():
    st = FT.statistics(_fake_res(delta=-5.0, aif_agree=0.5), [f"p{i}" for i in range(5)])  # PPO ~= AIFirst
    assert st["close_to_aifirst"]["A_ci_contains_zero"] and st["close_to_aifirst"]["label"] == "CLOSE TO AI-FIRST: YES"
    st = FT.statistics(_fake_res(delta=20.0, aif_agree=0.5), [f"p{i}" for i in range(5)])
    assert not st["close_to_aifirst"]["A_ci_contains_zero"] and st["close_to_aifirst"]["label"] == "CLOSE TO AI-FIRST: NO"


def test_27_close_to_aifirst_agreement_condition():
    st = FT.statistics(_fake_res(delta=20.0, aif_agree=0.95), [f"p{i}" for i in range(5)])
    assert st["close_to_aifirst"]["B_agreement_ge_95"] and st["close_to_aifirst"]["label"] == "CLOSE TO AI-FIRST: YES"
    st = FT.statistics(_fake_res(delta=20.0, aif_agree=0.94), [f"p{i}" for i in range(5)])
    assert not st["close_to_aifirst"]["B_agreement_ge_95"]


# --- 28-33 guards -----------------------------------------------------------------------------------------
def test_28_subgroup_scenario_cis_cannot_alter_verdict():
    tree = ast.parse((PROJECT_ROOT / "src/callverse_workforce/final_test/report.py").read_text(encoding="utf-8"))
    assigned = {t.value.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
                if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)}
    assert "st" not in assigned  # the report never rewrites the primary statistics
    stat_src = ast.unparse(next(n for n in ast.walk(ast.parse((PROJECT_ROOT / "src/callverse_workforce/final_test/v1_1.py").read_text(encoding="utf-8")))
                               if isinstance(n, ast.FunctionDef) and n.name == "statistics"))
    assert "family" not in stat_src and "fairness" not in stat_src


def test_29_raw_reward_retained():
    st = FT.statistics(_fake_res(), [f"p{i}" for i in range(5)])
    assert st["rule_mean_reward"] > 1000  # raw scale, not /100
    assert "REWARD_SCALE" not in "".join(p.read_text(encoding="utf-8") for p in MODULES)


def test_30_no_training_invoked():
    usage = [FT.code_usage(p) for p in MODULES]
    assert not set().union(*(u["calls"] for u in usage)) & FT.FORBIDDEN_CALLS
    assert "callverse_workforce.phase_e1.training" not in set().union(*(u["modules"] for u in usage))


def test_31_no_baseline_retuning():
    usage = [FT.code_usage(p) for p in MODULES]
    calls = set().union(*(u["calls"] for u in usage))
    assert not calls & {"tune", "grid_search", "select_config", "tune_baselines"}
    assert not any("tuning" in m for u in usage for m in u["modules"])


def test_32_no_v2_execution():
    usage = [FT.code_usage(p) for p in MODULES]
    assert not any("v2" in m.lower() for u in usage for m in u["modules"])
    assert not (PROJECT_ROOT / "src" / "callverse_workforce" / "v2").exists()


def test_33_final_artifacts_cannot_overwrite_history(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_final", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(SystemExit):
        mod.dump(E1_DIR / "development_evaluation" / "verdict.json", {"x": 1})
    with pytest.raises(SystemExit):
        mod.dump(tmp_path / "x.json", {})
    post = OUT / "integrity_post_run.json"
    if post.exists():
        assert _r(post)["unchanged"]
