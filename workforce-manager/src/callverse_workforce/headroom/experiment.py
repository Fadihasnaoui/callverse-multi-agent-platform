"""Phase C orchestration: nested loads, state sampling, replay plan, parallel branches, analysis.

Determinism: every branch job is a pure function of (config, seed, target, action, variant, policies).
Jobs are built in a fixed order and executed with ``ProcessPoolExecutor.map`` (order-preserving), so
results do not depend on the worker count; ``workers <= 1`` runs serially in-process.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from callverse_workforce.config import SimulationConfig
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.domain.enums import ACTION_ORDER
from callverse_workforce.headroom.branch import BranchJob, StateKey, policy_spec, run_branch
from callverse_workforce.headroom.loads import LOAD_LEVELS, headroom_config
from callverse_workforce.headroom.sampling import QUOTA, STRATA, sample_states
from callverse_workforce.headroom.stats import NEAR_TIE, analyze_state, cluster_bootstrap_mean, describe
from callverse_workforce.seeds import HEADROOM_EXTENSION_SEEDS, HEADROOM_PRIMARY_SEEDS, require_seeds
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.utils.rng import RngManager

K_FUTURES = 8
FROZEN_PATH = PROJECT_ROOT / "artifacts" / "baselines" / "frozen_policies.json"
FROZEN_THRESHOLD_THETA = 0.30
ROUGH_LABEL = "ROUGH DAILY HEADROOM ORDER OF MAGNITUDE (local regrets are not additive; not an achievable gain)"
# Pre-registered (written to replay_plan.json before C1 runs): extend to 1311-1320 only if, for some load
# level, the day-clustered 95% CI of the mean CROSS-FITTED Rule regret per eligible decision contains both
# NEAR_ZERO_REGRET and MEANINGFUL_REGRET, i.e. the data cannot tell "near zero" from "meaningful" apart.
NEAR_ZERO_REGRET = 0.1
MEANINGFUL_REGRET = 0.5


# ---------------------------------------------------------------------------
# frozen B2 policies
# ---------------------------------------------------------------------------
def load_frozen(path: Path = FROZEN_PATH) -> dict[str, dict[str, Any]]:
    frozen = json.loads(path.read_text(encoding="utf-8"))["frozen"]
    if frozen["ThresholdPolicy"]["theta"] != FROZEN_THRESHOLD_THETA:
        raise RuntimeError("frozen B2 ThresholdPolicy theta must remain 0.30")
    return frozen


# ---------------------------------------------------------------------------
# nested loads
# ---------------------------------------------------------------------------
def nested_manifest(base: SimulationConfig, seed: int) -> dict[str, Any]:
    days = {load: ContactGenerator(headroom_config(base, load), RngManager(seed)).generate_day() for load in LOAD_LEVELS}
    ids = {load: [c.contact_id for c in contacts] for load, contacts in days.items()}
    by_id = {load: {c.contact_id: c for c in contacts} for load, contacts in days.items()}
    n13 = set(ids["NORMAL"]) <= set(ids["NORMAL_X1_3"])
    n35 = set(ids["NORMAL_X1_3"]) <= set(ids["NORMAL_X1_5"])
    same13 = all(by_id["NORMAL"][i] == by_id["NORMAL_X1_3"][i] for i in ids["NORMAL"])
    same35 = all(by_id["NORMAL_X1_3"][i] == by_id["NORMAL_X1_5"][i] for i in ids["NORMAL_X1_3"])
    return {"seed": seed, "counts": {load: len(v) for load, v in ids.items()},
            "normal_subset_of_x1_3": n13, "x1_3_subset_of_x1_5": n35,
            "shared_contacts_identical": same13 and same35, "contact_ids": ids}


# ---------------------------------------------------------------------------
# sampling and jobs
# ---------------------------------------------------------------------------
def build_samples(base: SimulationConfig, seeds: Sequence[int], rule: dict[str, Any],
                  loads: Iterable[str] = LOAD_LEVELS) -> dict[str, Any]:
    seeds = require_seeds(seeds, HEADROOM_PRIMARY_SEEDS + HEADROOM_EXTENSION_SEEDS, "headroom sampling")
    out = {}
    for load in loads:
        cfg = headroom_config(base, load)
        states, metas = [], []
        for s in seeds:
            st, meta = sample_states(cfg, load, s, rule)
            states += st
            metas.append(meta)
        out[load] = {"states": states, "meta": metas}
    return out


def state_key(st: dict[str, Any]) -> StateKey:
    return StateKey(st["target_index"], st["decision_id"], st["contact_id"], st["timestamp"], st["observation"],
                    tuple(st["mask"][a.value] for a in ACTION_ORDER))


def make_jobs(cfg: SimulationConfig, states: Sequence[dict[str, Any]], rule: dict[str, Any],
              continuation: tuple[str, dict[str, Any]], k: int = K_FUTURES) -> list[BranchJob]:
    pre = policy_spec("RulePolicy", rule)
    cont = policy_spec(*continuation)
    jobs = []
    for st in states:
        key = state_key(st)
        for a in ACTION_ORDER:
            if not st["mask"][a.value]:
                continue  # an invalid action is never forced
            for variant in range(k):
                jobs.append(BranchJob(st["state_id"], cfg, st["seed"], key, a.value, variant, pre, cont))
    return jobs


def execute(jobs: Sequence[BranchJob], workers: int = 1, chunksize: int = 4) -> list[dict[str, Any]]:
    if workers <= 1:
        return [run_branch(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_branch, jobs, chunksize=chunksize))


def benchmark(base: SimulationConfig, samples: dict[str, Any], rule: dict[str, Any], per_load: int = 6) -> dict[str, Any]:
    out = {}
    for load, block in samples.items():
        cfg = headroom_config(base, load)
        states = block["states"]
        picks = [states[i] for i in np.linspace(0, len(states) - 1, per_load).astype(int)]
        jobs = [j for st in picks for j in make_jobs(cfg, [st], rule, ("RulePolicy", rule), k=1)[:2]]
        t0 = time.perf_counter()
        execute(jobs, workers=1)
        dt = time.perf_counter() - t0
        out[load] = {"benchmark_branches": len(jobs), "seconds_total": dt, "seconds_per_branch": dt / len(jobs)}
    return out


def plan(samples: dict[str, Any], bench: dict[str, Any], workers: int, k: int = K_FUTURES) -> dict[str, Any]:
    loads = {}
    for load, block in samples.items():
        pairs = sum(sum(st["mask"].values()) for st in block["states"])
        n = pairs * k
        sec = bench[load]["seconds_per_branch"] * n
        loads[load] = {"states": len(block["states"]), "state_action_pairs": pairs, "branches": n,
                       "max_branches_if_all_valid": len(block["states"]) * 4 * k,
                       "estimated_sequential_seconds": sec, "estimated_parallel_seconds": sec / max(1, workers)}
    total = sum(v["branches"] for v in loads.values())
    seq = sum(v["estimated_sequential_seconds"] for v in loads.values())
    return {
        "K": k, "workers": workers, "per_load": loads, "C1_branches": total, "C2_branches": total,
        "C1_estimated_sequential_seconds": seq, "C1_estimated_parallel_seconds": seq / max(1, workers),
        "pre_registered_extension_rule": {
            "metric": "mean cross-fitted Rule regret per eligible decision (Rule continuation, MEDIUM noise)",
            "ci": "95% bootstrap over days (clusters)",
            "near_zero": NEAR_ZERO_REGRET, "meaningful": MEANINGFUL_REGRET,
            "extend_if": "for any load, CI contains both thresholds",
        },
    }


# ---------------------------------------------------------------------------
# analysis
# ---------------------------------------------------------------------------
def band_ai(s: float) -> str:
    return "low(<0.4)" if s < 0.4 else ("medium(0.4-0.7)" if s < 0.7 else "high(>=0.7)")


def band_util(u: float) -> str:
    return "low(<=50%)" if u <= 0.5 else ("medium(50-<100%)" if u < 1.0 else "high(100%)")


def band_queue(q: int) -> str:
    return "empty/small(<=1)" if q <= 1 else "congested(>=2)"


def band_wait(w: float) -> str:
    return "short(<0.5)" if w < 0.5 else ("medium(0.5-2)" if w < 2.0 else "long(>=2)")


CONTEXT = {
    "ai_score": lambda st: band_ai(st["ai_resolution_score"]),
    "human_utilization": lambda st: band_util(st["human_utilization"]),
    "human_queue": lambda st: band_queue(st["human_queue_size"]),
    "urgency": lambda st: st["urgency_pred"],
    "waiting_time": lambda st: band_wait(st["waiting_time_so_far"]),
    "request_type": lambda st: st["request_type"],
    "time_of_day": lambda st: st["stratum"],
}
LABELS = [f"{a.value}_BEST" for a in ACTION_ORDER] + ["TIE"]


def analyze(states: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]], k: int = K_FUTURES) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    failed: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if r["status"] != "OK":
            failed[r["state_id"]].append(r.get("reason", r["status"]))
            continue
        grouped[r["state_id"]][r["action"]][r["variant"]] = r["discounted_return"]
    out = []
    for st in states:
        sid = st["state_id"]
        if sid in failed or len(grouped.get(sid, {})) < 2:
            out.append({"state_id": sid, "status": "FAILED", "reasons": failed.get(sid, ["missing"])})
            continue
        returns = {a: [grouped[sid][a][v] for v in range(k)] for a in grouped[sid]}
        pol = {"RulePolicy": st["rule_action"], "AIFirstPolicy": st["aifirst_action"],
               "ThresholdPolicy": st["threshold_action"]}
        res = analyze_state(returns, pol, sid)
        out.append({"state_id": sid, "status": "OK", "returns": returns, **res})
    return out


def _share(labels: list[str]) -> dict[str, float]:
    c = Counter(labels)
    return {l: c.get(l, 0) / len(labels) if labels else 0.0 for l in LABELS}


def summarize(states: Sequence[dict[str, Any]], analyses: Sequence[dict[str, Any]], metas: Sequence[dict[str, Any]],
              results: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    by_id = {st["state_id"]: st for st in states}
    ok = [a for a in analyses if a["status"] == "OK"]
    days = [by_id[a["state_id"]]["seed"] for a in ok]
    labels = [a["label"] for a in ok]
    eligible_per_day = float(np.mean([m["eligible_decisions"] for m in metas])) if metas else 0.0

    def jv(a, act):
        return a["J"][act]

    regret = {}
    for pol in ("RulePolicy", "AIFirstPolicy", "ThresholdPolicy"):
        raw = [a["regret"][pol]["raw"] for a in ok]
        clear = [a["regret"][pol]["clear"] for a in ok]
        cf = [a["regret"][pol]["cross_fitted"] for a in ok]
        regret[pol] = {
            "raw": describe(raw), "clear": describe(clear), "cross_fitted": describe(cf),
            "raw_ci": cluster_bootstrap_mean(raw, days, key + pol + "raw") if raw else {},
            "clear_ci": cluster_bootstrap_mean(clear, days, key + pol + "clear") if clear else {},
            "cross_fitted_ci": cluster_bootstrap_mean(cf, days, key + pol + "cf") if cf else {},
            "fraction_policy_is_best": float(np.mean([a["regret"][pol]["is_best"] for a in ok])) if ok else 0.0,
            "fraction_raw_below_1": float(np.mean(np.array(raw) < 1.0)) if raw else 0.0,
            "fraction_clear_zero": float(np.mean(np.array(clear) == 0.0)) if clear else 0.0,
            "rough_daily_raw": (float(np.mean(raw)) * eligible_per_day) if raw else 0.0,
            "rough_daily_cross_fitted": (float(np.mean(cf)) * eligible_per_day) if cf else 0.0,
            "rough_daily_label": ROUGH_LABEL,
        }
    adv = {}
    for alt in ("HUMAN", "WAIT", "CALLBACK"):
        rows = [a["advantage_vs_ai"][alt] for a in ok if alt in a["advantage_vs_ai"]]
        means = [r["mean"] for r in rows]
        adv[alt] = {**describe(means),
                    "positive_fraction": float(np.mean(np.array(means) > 0)) if rows else 0.0,
                    "clearly_positive_fraction": float(np.mean([r["clearly_positive"] for r in rows])) if rows else 0.0,
                    "clearly_negative_fraction": float(np.mean([r["clearly_negative"] for r in rows])) if rows else 0.0}
    context = {}
    for name, fn in CONTEXT.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for a in ok:
            groups[fn(by_id[a["state_id"]])].append(a)
        context[name] = {
            g: {"n": len(rows), "labels": _share([r["label"] for r in rows]),
                "mean_adv_vs_ai": {alt: (float(np.mean(v)) if (v := [r["advantage_vs_ai"][alt]["mean"] for r in rows
                                                                   if alt in r["advantage_vs_ai"]]) else None)
                                   for alt in ("HUMAN", "WAIT", "CALLBACK")},
                "mean_rule_regret_raw": float(np.mean([r["regret"]["RulePolicy"]["raw"] for r in rows])),
                "mean_rule_regret_cross_fitted": float(np.mean([r["regret"]["RulePolicy"]["cross_fitted"] for r in rows]))}
            for g, rows in sorted(groups.items())
        }
    gaps = [a["J"][a["best"]] - a["J"][a["second"]] for a in ok]
    masked = Counter(act for st in states for act, v in st["mask"].items() if not v)
    sample_dist = {name: dict(Counter(fn(st) for st in states)) for name, fn in CONTEXT.items()}
    branch_status = Counter(r["status"] for r in results)
    return {
        "n_sampled_states": len(states),
        "n_analyzed_states": len(ok),
        "reconstruction_success_rate": len(ok) / len(states) if states else 0.0,
        "branch_replays": len(results),
        "branch_status": dict(branch_status),
        "all_forced_once": all(r.get("forced_action_applied", True) for r in results if r["status"] == "OK"),
        "all_rewards_reconcile": all(r.get("reward_reconciles", True) for r in results if r["status"] == "OK"),
        "all_terminals_reconcile": all(r.get("terminal_reconciles", True) for r in results if r["status"] == "OK"),
        "eligible_decisions_per_day_mean": eligible_per_day,
        "mean_J": {
            "oracle_best_forced_action": float(np.mean([jv(a, a["best"]) for a in ok])),
            "RulePolicy_action": float(np.mean([jv(a, a["regret"]["RulePolicy"]["action"]) for a in ok])),
            "AIFirstPolicy_action": float(np.mean([jv(a, a["regret"]["AIFirstPolicy"]["action"]) for a in ok])),
            "ThresholdPolicy_action": float(np.mean([jv(a, a["regret"]["ThresholdPolicy"]["action"]) for a in ok])),
            "note": "per-state best forced action; NOT a deployable oracle policy",
        },
        "label_distribution": _share(labels),
        "label_counts": dict(Counter(labels)),
        "near_tie_fractions": {f"lt_{int(x)}": float(np.mean(np.array(gaps) < x)) for x in NEAR_TIE},
        "best_minus_second": describe(gaps),
        "regret": regret,
        "advantage_vs_ai": adv,
        "context": context,
        "mask_freedom": {"n_valid": dict(Counter(st["n_valid"] for st in states)),
                         "masked_action_counts": dict(masked)},
        "sample_distribution": sample_dist,
        "disagreement": disagreement(states, ok),
    }


def disagreement(states: Sequence[dict[str, Any]], ok: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from callverse_workforce.headroom.stats import paired_interval

    by_id = {st["state_id"]: st for st in states}
    rows = []
    for a in ok:
        st = by_id[a["state_id"]]
        ra, fa = st["rule_action"], st["aifirst_action"]
        if ra == fa:
            continue
        iv = paired_interval(np.subtract(a["returns"][ra], a["returns"][fa]), a["state_id"] + "dis")
        verdict = "RULE_CLEARLY_BETTER" if iv["t_lo"] > 0 else ("AIFIRST_CLEARLY_BETTER" if iv["t_hi"] < 0 else "TIE")
        rows.append({"state_id": a["state_id"], "rule_action": ra, "aifirst_action": fa, "J": a["J"],
                     "rule_minus_aifirst": iv["mean"], "verdict": verdict, "label": a["label"],
                     "rule_regret": a["regret"]["RulePolicy"]["raw"], "aifirst_regret": a["regret"]["AIFirstPolicy"]["raw"]})
    return {
        "n": len(rows),
        "share_of_states": len(rows) / len(ok) if ok else 0.0,
        "action_pairs": dict(Counter(f"{r['rule_action']}_vs_{r['aifirst_action']}" for r in rows)),
        "verdicts": dict(Counter(r["verdict"] for r in rows)),
        "labels": dict(Counter(r["label"] for r in rows)),
        "rule_minus_aifirst": describe([r["rule_minus_aifirst"] for r in rows]),
        "mean_rule_regret": float(np.mean([r["rule_regret"] for r in rows])) if rows else 0.0,
        "mean_aifirst_regret": float(np.mean([r["aifirst_regret"] for r in rows])) if rows else 0.0,
        "states": rows,
    }


def extension_needed(summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    verdicts = {}
    for load, s in summaries.items():
        ci = s["regret"]["RulePolicy"]["cross_fitted_ci"]
        ambiguous = ci["lo"] <= NEAR_ZERO_REGRET and ci["hi"] >= MEANINGFUL_REGRET
        verdicts[load] = {"ci": ci, "ambiguous": ambiguous}
    return {"extend": any(v["ambiguous"] for v in verdicts.values()), "per_load": verdicts,
            "rule": f"extend iff some CI contains both {NEAR_ZERO_REGRET} and {MEANINGFUL_REGRET}"}


def subset_states(states: Sequence[dict[str, Any]], fraction_quota: dict[str, int]) -> list[dict[str, Any]]:
    """Deterministic stratified subset: the first n states (by target index) of each (day, stratum)."""
    keep = []
    seen: dict[tuple[int, str], int] = defaultdict(int)
    for st in sorted(states, key=lambda s: (s["seed"], s["target_index"])):
        k = (st["seed"], st["stratum"])
        if seen[k] < fraction_quota[st["stratum"]]:
            keep.append(st)
            seen[k] += 1
    return keep


SUBSET_QUOTA = {"calm": 1, "rising": 2, "peak": 4, "post_peak": 1, "late": 2}  # 10 of the 30 states per day
assert set(SUBSET_QUOTA) == set(QUOTA) == {s for s, _ in STRATA}
