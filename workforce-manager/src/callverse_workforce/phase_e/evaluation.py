"""Phase E evaluation: paired day-level evaluation of PPO models and frozen baselines, statistics, verdict.

Baselines are used here for COMPARISON ONLY (this module is never imported by the training code).
"""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

import numpy as np

from callverse_workforce.baselines.metrics import policy_day_metrics
from callverse_workforce.baselines.policies import make_policy
from callverse_workforce.config import SimulationConfig, load_config
from callverse_workforce.domain.enums import ACTION_ORDER
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_d.scenarios import period_of
from callverse_workforce.phase_e.scenarios import DaySpec, day_config
from callverse_workforce.phase_e.training import PPORoutingPolicy
from callverse_workforce.seeds import (
    FINAL_V1_1_SEEDS,
    PHASE_F_DEV_SEEDS,
    PHASE_F_TEST_SEEDS,
    PPO_DEV_EVAL_SEEDS,
    PPO_E1_DEV_SEEDS,
    PPO_VALIDATION_SEEDS,
    assert_v2_reserved_untouched,
    require_seeds,
)
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.twin.runner import run_simulation
from callverse_workforce.utils.rng import RngManager

SUCCESS = "PPO V1.1 SUCCESS — COMPETITIVE WITH STRONG BASELINES"
PARTIAL = "PPO V1.1 PARTIAL SUCCESS — LEARNS VALID POLICY BUT BELOW RULE"
FAILURE = "PPO V1.1 FAILURE — TRAINING/FORMULATION ISSUE"
MARGIN_FRACTION = 0.02
BOOTSTRAP_B = 10_000
_MODELS: dict[str, Any] = {}
_CONFIGS: dict[tuple[str, str, str], SimulationConfig] = {}
WINDOW_KIND = {"AI_DEGRADATION": "ai_degradation", "WORST_DAY": "ai_degradation", "HUMAN_SHORTAGE": "human_shortage",
               "DEMAND_SHOCK": "demand_shock"}


def days_for(purpose: str, specs: Sequence[DaySpec]) -> Sequence[DaySpec]:
    allowed = {"validation": PPO_VALIDATION_SEEDS, "dev": PPO_DEV_EVAL_SEEDS, "e1_dev": PPO_E1_DEV_SEEDS,
               "final_v1_1": FINAL_V1_1_SEEDS, "phase_f_dev": PHASE_F_DEV_SEEDS, "phase_f_test": PHASE_F_TEST_SEEDS}[purpose]
    assert_v2_reserved_untouched([d.seed for d in specs])
    require_seeds([d.seed for d in specs], allowed, f"PPO {purpose} evaluation")
    return specs


def _policy(kind: str, ref: Any, seed: int):
    if kind == "ppo":
        if ref not in _MODELS:
            import torch

            from callverse_workforce.phase_e.smdp import SMDPMaskablePPO

            torch.set_num_threads(1)
            _MODELS[ref] = SMDPMaskablePPO.load(ref, device="cpu")
        return PPORoutingPolicy(_MODELS[ref])
    name, params = ref
    return make_policy(name, params, seed)


def _config(base_path: str, day: DaySpec) -> SimulationConfig:
    key = (base_path, day.family, day.cell.cell_id)
    if key not in _CONFIGS:
        _CONFIGS[key] = day_config(load_config(base_path), day.family, day.cell)
    return _CONFIGS[key]


def windows(cfg: SimulationConfig, seed: int) -> dict[str, tuple[float, float] | None]:
    """Hidden event windows (ANALYSIS ONLY)."""
    from callverse_workforce.twin.digital_twin import DigitalTwin

    twin = DigitalTwin(cfg, seed, contacts=[])
    return {"ai_degradation": twin.degradation_window, "human_shortage": twin.shortage_window,
            "demand_shock": ContactGenerator(cfg, RngManager(seed)).shock_window()}


def eval_job(job: tuple[str, str, Any, str, tuple[str, str, int]]) -> dict[str, Any]:
    label, kind, ref, base_path, (family, cell_id, seed) = job
    day = DaySpec(family, {c.cell_id: c for c in MIXTURE_CELLS}[cell_id], seed)
    cfg = _config(base_path, day)
    result = run_simulation(cfg, _policy(kind, ref, seed), seed=seed)
    d = policy_day_metrics(result)
    d |= {"label": label, "family": family, "cell": cell_id, "visible_ai_failures": sum(
        1 for e in result.events if e.event_type.value == "AI_VISIBLE_FAILURE")}
    d["adaptation"] = adaptation(result, cfg, family, seed)
    return d


def adaptation(result, cfg, family: str, seed: int) -> dict[str, Any]:
    """Observable-variable diagnostics; hidden windows used only to label periods for analysis."""
    trs = result.transitions
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]
    by_fail = {}
    for lo, hi in bins:
        sel = [t for t in trs if lo <= t.state.recent_ai_failure_rate < hi]
        if sel:
            by_fail[f"{lo:.1f}-{hi:.1f}"] = {"n": len(sel), "ai_share": sum(t.action.value == "AI" for t in sel) / len(sel)}
    by_humans = {}
    for k in ("0", "1-2", "3+"):
        sel = [t for t in trs if (t.state.humans_available == 0 if k == "0" else
                                  1 <= t.state.humans_available <= 2 if k == "1-2" else t.state.humans_available >= 3)]
        if sel:
            by_humans[k] = {"n": len(sel), "human_share": sum(t.action.value == "HUMAN" for t in sel) / len(sel)}
    out: dict[str, Any] = {"ai_share_by_recent_ai_failure_rate": by_fail, "human_share_by_humans_available": by_humans}
    kind = WINDOW_KIND.get(family)
    if kind:
        w = windows(cfg, seed)[kind]
        per = defaultdict(list)
        for t in trs:
            per[period_of(t.timestamp, w)].append(t)
        out["periods"] = {p: {"n": len(ts), **{f"{a.value}_share": sum(t.action is a for t in ts) / len(ts) for a in ACTION_ORDER},
                              "recent_ai_failure_rate": float(np.mean([t.state.recent_ai_failure_rate for t in ts])),
                              "humans_available": float(np.mean([t.state.humans_available for t in ts])),
                              "human_queue_size": float(np.mean([t.state.human_queue_size for t in ts])),
                              "forecast_30min": float(np.mean([t.state.forecast_30min for t in ts])),
                              "reward_per_decision": float(np.mean([t.reward for t in ts]))}
                          for p, ts in per.items()}
        out["window"] = w
    return out


def pmap(fn, jobs, workers: int, chunksize: int = 2) -> list:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs, chunksize=chunksize))


def evaluate(policies: dict[str, tuple[str, Any]], days: Sequence[DaySpec], purpose: str, base_path: str,
             workers: int) -> dict[str, list[dict[str, Any]]]:
    days = days_for(purpose, days)
    jobs = [(label, kind, ref, base_path, (d.family, d.cell.cell_id, d.seed))
            for label, (kind, ref) in policies.items() for d in days]
    rows = pmap(eval_job, jobs, workers)
    out: dict[str, list] = defaultdict(list)
    for r in rows:
        out[r["label"]].append(r)
    return dict(out)


# ---------------------------------------------------------------------------
# paired statistics and verdict
# ---------------------------------------------------------------------------
def paired_bootstrap(diffs: Sequence[float], key: int = 0, b: int = BOOTSTRAP_B) -> dict[str, float]:
    """Percentile bootstrap over DAYS of the mean paired difference (95%)."""
    d = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(np.random.SeedSequence(20251025, spawn_key=(key,)))
    means = d[rng.integers(0, d.size, size=(b, d.size))].mean(axis=1)
    return {"n_days": int(d.size), "mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "sd": float(d.std(ddof=1)) if d.size > 1 else 0.0}


def by_day(rows: Sequence[dict[str, Any]]) -> dict[int, float]:
    return {r["seed"]: r["total_reward"] for r in rows}


def ppo_mean_by_day(seed_rows: Sequence[Sequence[dict[str, Any]]]) -> dict[int, float]:
    """PPO_mean_d = (1/S) * sum_s Reward(PPO_s, day d) -- one value per day."""
    maps = [by_day(rows) for rows in seed_rows]
    days = sorted(maps[0])
    if any(sorted(m) != days for m in maps):
        raise ValueError("every PPO seed must be evaluated on the same days")
    return {d: float(np.mean([m[d] for m in maps])) for d in days}


def paired_diff(a: dict[int, float], b: dict[int, float]) -> list[float]:
    if sorted(a) != sorted(b):
        raise ValueError("paired comparison needs identical days")
    return [a[d] - b[d] for d in sorted(a)]


def verdict(ppo_mean: dict[int, float], rule: dict[int, float], random: dict[int, float],
            aifirst: dict[int, float]) -> dict[str, Any]:
    """Pre-registered classification (artifacts/ppo_v1_1/verdict_rule.json)."""
    rule_mean = float(np.mean(list(rule.values())))
    margin = -MARGIN_FRACTION * rule_mean
    vs_rule = paired_bootstrap(paired_diff(ppo_mean, rule), key=1)
    vs_random = paired_bootstrap(paired_diff(ppo_mean, random), key=2)
    vs_aifirst = paired_bootstrap(paired_diff(ppo_mean, aifirst), key=3)
    if vs_rule["lo"] > margin:
        label = SUCCESS
    elif vs_random["mean"] > 0 and vs_aifirst["mean"] > 0:
        label = PARTIAL
    else:
        label = FAILURE
    return {"verdict": label, "rule_mean_reward": rule_mean, "margin": margin, "ppo_minus_rule": vs_rule,
            "relative_difference": vs_rule["mean"] / rule_mean, "ppo_minus_random": vs_random,
            "ppo_minus_aifirst": vs_aifirst}


def seed_noninferiority(seed_rows: Sequence[Sequence[dict[str, Any]]], rule: dict[int, float]) -> dict[str, Any]:
    margin = -MARGIN_FRACTION * float(np.mean(list(rule.values())))
    per = []
    for i, rows in enumerate(seed_rows):
        ci = paired_bootstrap(paired_diff(by_day(rows), rule), key=100 + i)
        per.append(ci | {"noninferior": ci["lo"] > margin})
    return {"margin": margin, "per_seed": per, "noninferior_seed_count": sum(p["noninferior"] for p in per),
            "n_seeds": len(per)}


def select_config(candidates: Sequence[dict[str, Any]], random_val_mean: float) -> dict[str, Any]:
    """Pre-registered: exclude candidates with any search seed below RandomPolicy's validation mean
    (catastrophic); select the highest mean validation raw reward over both seeds; ties -> grid order."""
    eligible = [c for c in candidates if min(c["seed_means"]) > random_val_mean]
    pool = eligible or list(candidates)
    best = pool[0]
    for c in pool[1:]:
        if c["mean_val_reward"] > best["mean_val_reward"] + 1e-9:
            best = c
    return {"selected": best, "excluded": [c["index"] for c in candidates if c not in eligible]}
