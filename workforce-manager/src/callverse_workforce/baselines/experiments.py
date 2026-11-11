"""Phase B2: baseline tuning (validation seeds only), freezing, and paired evaluation.

Protocol
--------
1. Tune ThresholdPolicy, RulePolicy and BestFixedPolicy on NORMAL validation seeds 1011-1020 only;
   select the candidate with the highest mean center-level reward (ties -> first in grid order).
2. Freeze the selected parameters.
3. Evaluate all five frozen baselines on the SAME seeds (paired): NORMAL 1101-1120, LOAD_CHECK 1201-1220.
Every policy faces the same day for a given seed: exogenous arrivals and all per-contact latent draws
come from seed-keyed RNG streams that do not depend on routing.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

import numpy as np

from callverse_workforce.baselines.metrics import REWARD_KEYS, policy_day_metrics
from callverse_workforce.baselines.policies import BASELINE_NAMES, make_policy
from callverse_workforce.config.models import SimulationConfig
from callverse_workforce.domain.enums import ACTION_ORDER, CallbackReason, Channel, LanguageDialect
from callverse_workforce.seeds import (
    B2_LOAD_CHECK_EVAL_SEEDS,
    B2_NORMAL_EVAL_SEEDS,
    B2_VALIDATION_SEEDS,
    SeedPolicyError,
    require_seeds,
)
from callverse_workforce.twin.runner import run_simulation

THRESHOLD_GRID: tuple[float, ...] = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
RULE_GRID: dict[str, tuple[Any, ...]] = {
    "theta": (0.50, 0.60, 0.70),
    "queue_max": (2, 5, 10),
    "congested_action": ("CALLBACK", "WAIT", "AI"),
}
FIXED_ACTIONS: tuple[str, ...] = ("AI", "HUMAN", "WAIT", "CALLBACK")
NORMAL_PURPOSE = "SYNTHETIC_NORMAL_DAY_V1"

CALLBACK_OVERUSE_SHARE = 0.30
DOMINANT_WARN = (0.90, 0.95)
CALLBACK_WARNING = "Potential CALLBACK over-use under current SYNTHETIC V1 reward assumptions."
CALLBACK_PENALTY_NOTE = "MAX_ATTEMPTS / WINDOW_EXPIRED = -1 may underprice unresolved customers (unchanged)."

SCALAR_KEYS = (
    "total_reward", "instances", "roots", "resolution_rate_root", "resolution_rate_instance", "hir_root",
    "hir_instance", "human_routing_rate", "human_busy_minutes", "human_utilization", "decisions",
    "dominant_action_share", "abandonment_rate", "missed_deadline_rate", "unfinished_rate", "online_wait_mean",
    "online_wait_p90", "ai_attempts_completed", "false_ai_rate", "ai_visible_failure_rate", "ai_escalations",
    "wait_limit_count", "abandonments_in_wait", "callbacks_scheduled", "callback_success", "callback_dropout",
    "callback_max_attempts", "callback_window_expired", "callback_unfinished", "callback_success_rate",
    "callback_human_minutes", "callback_reward_contribution", "mean_valid_actions",
)


# ---------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------
def _run_one(job: tuple[SimulationConfig, str, dict[str, Any], int]) -> dict[str, Any]:
    config, name, params, seed = job
    result = run_simulation(config, make_policy(name, params, seed), seed=seed)
    day = policy_day_metrics(result)
    day["policy"] = name
    day["params"] = dict(params)
    return day


def run_policy_days(
    config: SimulationConfig, name: str, params: dict[str, Any], seeds: Sequence[int], workers: int = 1
) -> list[dict[str, Any]]:
    jobs = [(config, name, params, s) for s in seeds]
    if workers <= 1:
        return [_run_one(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_one, jobs))


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


# ---------------------------------------------------------------------------
# validation / tuning (NORMAL validation seeds only)
# ---------------------------------------------------------------------------
def _check_validation(config: SimulationConfig, seeds: Sequence[int], purpose: str) -> tuple[int, ...]:
    if config.scenario.purpose != NORMAL_PURPOSE:
        raise SeedPolicyError(f"{purpose}: tuning is allowed on NORMAL days only, got {config.scenario.purpose}")
    return require_seeds(seeds, B2_VALIDATION_SEEDS, purpose)


def _select(name: str, candidates: Sequence[dict[str, Any]], config, seeds, workers) -> dict[str, Any]:
    rows = []
    for params in candidates:
        days = run_policy_days(config, name, params, seeds, workers)
        rewards = [d["total_reward"] for d in days]
        rows.append({
            "params": params,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
            "per_seed_reward": {str(d["seed"]): d["total_reward"] for d in days},
            "resolution_rate_root": float(np.mean([d["resolution_rate_root"] for d in days])),
            "hir_root": float(np.mean([d["hir_root"] for d in days])),
            "abandonment_rate": float(np.mean([d["abandonment_rate"] for d in days])),
            "online_wait_p90": float(np.mean([d["online_wait_p90"] for d in days])),
            "action_share": {a.value: float(np.mean([d["action_share"][a.value] for d in days])) for a in ACTION_ORDER},
        })
    best = 0
    for i, row in enumerate(rows):  # strict '>' keeps the first candidate on ties
        if row["mean_reward"] > rows[best]["mean_reward"]:
            best = i
    return {"policy": name, "seeds": list(seeds), "selection_metric": "mean total reward (center-level)",
            "candidates": rows, "selected": dict(rows[best]["params"])}


def tune_threshold(config, seeds=B2_VALIDATION_SEEDS, grid=THRESHOLD_GRID, workers=1) -> dict[str, Any]:
    seeds = _check_validation(config, seeds, "ThresholdPolicy tuning")
    return _select("ThresholdPolicy", [{"theta": t} for t in grid], config, seeds, workers)


def tune_rule(config, seeds=B2_VALIDATION_SEEDS, grid=RULE_GRID, workers=1) -> dict[str, Any]:
    seeds = _check_validation(config, seeds, "RulePolicy tuning")
    keys = list(grid)
    candidates = [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]
    return _select("RulePolicy", candidates, config, seeds, workers)


def select_best_fixed(config, seeds=B2_VALIDATION_SEEDS, actions=FIXED_ACTIONS, workers=1) -> dict[str, Any]:
    seeds = _check_validation(config, seeds, "BestFixedPolicy selection")
    return _select("BestFixedPolicy", [{"action": a} for a in actions], config, seeds, workers)


def validate_and_freeze(config, seeds=B2_VALIDATION_SEEDS, workers=1, **grids) -> dict[str, Any]:
    threshold = tune_threshold(config, seeds, grids.get("threshold_grid", THRESHOLD_GRID), workers)
    rule = tune_rule(config, seeds, grids.get("rule_grid", RULE_GRID), workers)
    fixed = select_best_fixed(config, seeds, grids.get("fixed_actions", FIXED_ACTIONS), workers)
    frozen = {
        "RandomPolicy": {"rng": "dedicated policy stream keyed by (day seed, 'policy:RandomPolicy')"},
        "AIFirstPolicy": {"order": ["AI", "HUMAN", "CALLBACK", "WAIT"]},
        "ThresholdPolicy": threshold["selected"],
        "RulePolicy": rule["selected"],
        "BestFixedPolicy": fixed["selected"],
    }
    return {"validation_seeds": list(seeds), "frozen": frozen,
            "threshold": threshold, "rule": rule, "best_fixed": fixed}


# ---------------------------------------------------------------------------
# paired evaluation of frozen policies
# ---------------------------------------------------------------------------
EVAL_SETS = {"NORMAL": B2_NORMAL_EVAL_SEEDS, "LOAD_CHECK": B2_LOAD_CHECK_EVAL_SEEDS}


def evaluate_frozen(
    config: SimulationConfig, frozen: dict[str, dict[str, Any]], seeds: Sequence[int], label: str, workers: int = 1
) -> dict[str, Any]:
    seeds = require_seeds(seeds, EVAL_SETS[label], f"{label} evaluation")
    policies = {}
    for name in BASELINE_NAMES:
        days = run_policy_days(config, name, frozen[name], seeds, workers)
        policies[name] = {"params": frozen[name], "days": days, "aggregate": aggregate_days(days),
                          "diagnostics": diagnostics(days)}
    return {"label": label, "purpose": config.scenario.purpose, "seeds": list(seeds), "policies": policies,
            "paired": paired_variability(policies)}


def aggregate_days(days: Sequence[dict[str, Any]]) -> dict[str, Any]:
    agg: dict[str, Any] = {k: _mean_std([d[k] for d in days]) for k in SCALAR_KEYS}
    agg["action_share"] = {a.value: _mean_std([d["action_share"][a.value] for d in days]) for a in ACTION_ORDER}
    agg["valid_action_pct"] = {k: _mean_std([d["valid_action_pct"][k] for d in days]) for k in ("4", "3", "2", "1")}
    agg["reward"] = {k: _mean_std([d["reward"][k] for d in days]) for k in REWARD_KEYS}
    agg["callback_reasons_total"] = {r.value: sum(d["callback_reasons"][r.value] for d in days) for r in CallbackReason}
    agg["fairness"] = {
        group: {
            key: {m: _mean_std([d["fairness"][group][key][m] for d in days])
                  for m in ("resolution_rate", "hir_instance", "wait_mean", "wait_p90", "abandonment_rate")}
            for key in keys
        }
        for group, keys in (("language", [x.value for x in LanguageDialect]), ("channel", [x.value for x in Channel]))
    }
    return agg


def diagnostics(days: Sequence[dict[str, Any]]) -> dict[str, Any]:
    share = {a.value: float(np.mean([d["action_share"][a.value] for d in days])) for a in ACTION_ORDER}
    dominant = max(share, key=share.get)
    dom_share = share[dominant]
    out: dict[str, Any] = {
        "action_share_mean": share,
        "callback_share": share["CALLBACK"],
        "dominant_action": dominant,
        "dominant_action_share": dom_share,
        "dominant_over_90": dom_share > DOMINANT_WARN[0],
        "dominant_over_95": dom_share > DOMINANT_WARN[1],
        "nearly_constant": dom_share > DOMINANT_WARN[1],
        "warnings": [],
    }
    if dom_share > DOMINANT_WARN[0]:
        level = ">95%" if dom_share > DOMINANT_WARN[1] else ">90%"
        out["warnings"].append(f"Dominant action {dominant} chosen in {dom_share:.1%} of decisions ({level}): "
                               "policy is nearly constant.")
    overuse = share["CALLBACK"] >= CALLBACK_OVERUSE_SHARE or dominant == "CALLBACK"
    out["callback_overuse"] = overuse
    if overuse:
        scheduled = sum(d["callbacks_scheduled"] for d in days)
        out["callback_detail"] = {
            "action_share": share["CALLBACK"],
            "success_rate": sum(d["callback_success"] for d in days) / scheduled if scheduled else 0.0,
            "dropout_rate": sum(d["callback_dropout"] for d in days) / scheduled if scheduled else 0.0,
            "max_attempts_rate": sum(d["callback_max_attempts"] for d in days) / scheduled if scheduled else 0.0,
            "window_expired_rate": sum(d["callback_window_expired"] for d in days) / scheduled if scheduled else 0.0,
            "callback_human_minutes_mean": float(np.mean([d["callback_human_minutes"] for d in days])),
            "callback_reward_contribution_mean": float(np.mean([d["callback_reward_contribution"] for d in days])),
        }
        out["warnings"] += [CALLBACK_WARNING, CALLBACK_PENALTY_NOTE]
    return out


def paired_variability(policies: dict[str, dict[str, Any]]) -> dict[str, Any]:
    names = list(policies)
    rewards = {n: np.array([d["total_reward"] for d in policies[n]["days"]]) for n in names}
    seeds = [d["seed"] for d in policies[names[0]]["days"]]
    for n in names:
        assert [d["seed"] for d in policies[n]["days"]] == seeds, "evaluation must be paired"
    best = max(names, key=lambda n: rewards[n].mean())
    vs_best = {}
    for n in names:
        diff = rewards[n] - rewards[best]
        vs_best[n] = {**_mean_std(diff), "days_better_than_best": int((diff > 0).sum()),
                      "days_worse_than_best": int((diff < 0).sum())}
    return {
        "seeds": seeds,
        "best_by_mean_reward": best,
        "per_day_reward": {n: {str(s): float(r) for s, r in zip(seeds, rewards[n])} for n in names},
        "diff_vs_best": vs_best,
        "per_day_winner_counts": dict(sorted(
            {n: int(sum(1 for i in range(len(seeds)) if max(names, key=lambda m: rewards[m][i]) == n))
             for n in names}.items())),
    }
