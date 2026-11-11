"""Phase D orchestration: V1.1 audit, baseline retuning, headroom screening / confirmation, learnability.

All protocol choices are pre-registered in docs/twin_v1_1_rationale.md. Every job is a pure function of
its inputs and runs through one order-preserving process pool, so results do not depend on workers.
"""

from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Sequence

import numpy as np

from callverse_workforce.baselines.experiments import FIXED_ACTIONS, RULE_GRID, THRESHOLD_GRID
from callverse_workforce.baselines.metrics import policy_day_metrics
from callverse_workforce.baselines.policies import BASELINE_NAMES, make_policy
from callverse_workforce.config import SimulationConfig
from callverse_workforce.domain.enums import ACTION_ORDER, EventType, Urgency
from callverse_workforce.headroom import experiment as H
from callverse_workforce.headroom.branch import run_branch
from callverse_workforce.headroom.sampling import sample_states
from callverse_workforce.headroom.stats import cluster_bootstrap_mean
from callverse_workforce.phase_d.grid import GRID, MAX_CELL, MID_CELL, MIXTURE_CELLS, V1_CELL, Cell
from callverse_workforce.phase_d.scenarios import event_windows, period_of
from callverse_workforce.seeds import (
    PHASE_D_AUDIT_SEEDS,
    PHASE_D_EXTENSION_SEEDS,
    PHASE_D_HEADROOM_SEEDS,
    PHASE_D_VALIDATION_SEEDS,
    require_seeds,
)
from callverse_workforce.twin.runner import run_simulation

MEANINGFUL = 0.5
SCREEN_DAYS = PHASE_D_HEADROOM_SEEDS[:3]
SCREEN_K = 4
CONFIRM_K = 8
LEARN_FEATURES = ("ai_resolution_score", "recent_ai_failure_rate", "humans_available", "human_utilization",
                  "human_queue_size", "waiting_time_so_far", "urgency_pred", "deadline_remaining_pred",
                  "request_type", "forecast_30min", "routing_attempt_count", "time")


# ---------------------------------------------------------------------------
# generic parallel map
# ---------------------------------------------------------------------------
def pmap(fn: Callable, jobs: Sequence, workers: int, chunksize: int = 4) -> list:
    if workers <= 1:
        return [fn(j) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, jobs, chunksize=chunksize))


def day_job(job: tuple[SimulationConfig, str, dict[str, Any], int, str]) -> dict[str, Any]:
    """Run one policy day and return B2 metrics + V1.1 operational metrics."""
    cfg, name, params, seed, tag = job
    result = run_simulation(cfg, make_policy(name, params, seed), seed=seed)
    d = policy_day_metrics(result) | v11_metrics(result)
    d.pop("fairness", None)
    return d | {"policy": name, "params": dict(params), "tag": tag}


def v11_metrics(result) -> dict[str, Any]:
    ev = Counter(e.event_type for e in result.events)
    hangups = sum(1 for e in result.events if e.event_type is EventType.ABANDONMENT
                  and e.data.get("cause") == "post_ai_failure_hangup")
    handoff = [e.data["handoff_minutes"] for e in result.events
               if e.event_type is EventType.HUMAN_SERVICE_STARTED and "handoff_minutes" in e.data]
    escalated = [c for c in result.contacts if c.escalated]
    entered = {e.contact_id for e in result.events if e.event_type is EventType.HUMAN_QUEUE_ENTERED}
    q = [c.human_queue_minutes for c in result.contacts if c.contact_id in entered]
    direct = sum(c.human_direct_minutes for c in result.contacts)
    return {
        "visible_ai_failures": ev[EventType.AI_VISIBLE_FAILURE],
        "escalations": ev[EventType.AI_ESCALATION],
        "escalated_served": sum(1 for c in escalated if c.human_touched),
        "handoff_minutes_added": float(sum(handoff)),
        "handoff_minutes_per_served_escalation": float(np.mean(handoff)) if handoff else 0.0,
        "post_ai_failure_hangups": hangups,
        "hangup_share_of_visible_failures": hangups / ev[EventType.AI_VISIBLE_FAILURE] if ev[EventType.AI_VISIBLE_FAILURE] else 0.0,
        "escalated_abandoned_in_queue": sum(1 for c in escalated if c.outcome.value == "ABANDONED"),
        "human_queue_minutes_p90": float(np.percentile(q, 90)) if q else 0.0,
        "human_service_inflation": float(sum(handoff)) / (direct - float(sum(handoff))) if direct > sum(handoff) else 0.0,
    }


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.mean([r[key] for r in rows])) if rows else 0.0


def _std(rows: Sequence[dict[str, Any]], key: str) -> float:
    return float(np.std([r[key] for r in rows], ddof=1)) if len(rows) > 1 else 0.0


# ---------------------------------------------------------------------------
# D1 audit
# ---------------------------------------------------------------------------
AUDIT_KEYS = ("visible_ai_failures", "escalations", "handoff_minutes_added", "handoff_minutes_per_served_escalation",
              "post_ai_failure_hangups", "hangup_share_of_visible_failures", "escalated_abandoned_in_queue",
              "abandonment_rate", "human_busy_minutes", "human_utilization", "human_queue_minutes_p90",
              "resolution_rate_root", "false_ai_rate", "total_reward", "human_service_inflation", "hir_root")


def audit(base_cfgs: dict[str, SimulationConfig], audit_policies: dict[str, tuple[str, dict]], workers: int,
          seeds: Sequence[int] = PHASE_D_AUDIT_SEEDS) -> dict[str, Any]:
    seeds = require_seeds(seeds, PHASE_D_AUDIT_SEEDS, "V1.1 audit")
    jobs = [(cfg, name, params, s, f"{cid}|{label}") for cid, cfg in base_cfgs.items()
            for label, (name, params) in audit_policies.items() for s in seeds]
    rows = pmap(day_job, jobs, workers)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["tag"]].append(r)
    cells = {}
    for tag, rs in by.items():
        cid, label = tag.split("|")
        entry = {k: {"mean": _mean(rs, k), "std": _std(rs, k)} for k in AUDIT_KEYS}
        entry["reconciled_all_days"] = all(r["reconciled"] for r in rs)
        cells.setdefault(cid, {})[label] = entry | {"flags": audit_flags(entry, base_cfgs[cid])}
    return {"seeds": list(seeds), "policies": {k: list(v) for k, v in audit_policies.items()}, "cells": cells}


def audit_flags(e: dict[str, Any], cfg: SimulationConfig) -> list[str]:
    flags = []
    if e["hangup_share_of_visible_failures"]["mean"] > 0.5:
        flags.append("NEARLY_ALL_FAILED_AI_CUSTOMERS_HANG_UP")
    if e["human_utilization"]["mean"] > 1.0 + 1e-9:
        flags.append("IMPOSSIBLE_HUMAN_UTILIZATION")
    if not e["reconciled_all_days"]:
        flags.append("TERMINAL_RECONCILIATION_FAILURE")
    h = cfg.v1_1.handoff_overhead_minutes
    if e["handoff_minutes_added"]["mean"] > 0 and abs(e["handoff_minutes_per_served_escalation"]["mean"] - h) > 1e-9:
        flags.append("HANDOFF_ACCOUNTING_MISMATCH")
    if e["human_service_inflation"]["mean"] > 0.5:
        flags.append("ABSURD_SERVICE_TIME_INFLATION")
    return flags


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------
def _candidates(name: str) -> list[dict[str, Any]]:
    if name == "ThresholdPolicy":
        return [{"theta": t} for t in THRESHOLD_GRID]
    if name == "RulePolicy":
        keys = list(RULE_GRID)
        return [dict(zip(keys, v)) for v in itertools.product(*(RULE_GRID[k] for k in keys))]
    return [{"action": a} for a in FIXED_ACTIONS]


def tune(cfgs: dict[str, SimulationConfig], workers: int, seeds: Sequence[int] = PHASE_D_VALIDATION_SEEDS,
         label: str = "PRIMARY_MIXTURE") -> dict[str, Any]:
    """Select Threshold / Rule / BestFixed by mean reward over all (config, validation seed) pairs."""
    seeds = require_seeds(seeds, PHASE_D_VALIDATION_SEEDS, f"{label} tuning")
    out = {"label": label, "configs": list(cfgs), "seeds": list(seeds), "policies": {}}
    jobs, keys = [], []
    for name in ("ThresholdPolicy", "RulePolicy", "BestFixedPolicy"):
        for i, params in enumerate(_candidates(name)):
            for cid, cfg in cfgs.items():
                for s in seeds:
                    jobs.append((cfg, name, params, s, f"{name}|{i}|{cid}"))
    rows = pmap(day_job, jobs, workers)
    by: dict[tuple[str, int], list] = defaultdict(list)
    for r in rows:
        name, i, _ = r["tag"].split("|")
        by[(name, int(i))].append(r)
    for name in ("ThresholdPolicy", "RulePolicy", "BestFixedPolicy"):
        cands = _candidates(name)
        table = [{"params": cands[i], "mean_reward": _mean(by[(name, i)], "total_reward"),
                  "std_reward": _std(by[(name, i)], "total_reward"),
                  "resolution_rate_root": _mean(by[(name, i)], "resolution_rate_root"),
                  "hir_root": _mean(by[(name, i)], "hir_root"),
                  "ai_share": float(np.mean([r["action_share"]["AI"] for r in by[(name, i)]]))}
                 for i in range(len(cands))]
        best = 0
        for i, row in enumerate(table):
            if row["mean_reward"] > table[best]["mean_reward"]:
                best = i
        out["policies"][name] = {"candidates": table, "selected": dict(cands[best])}
    return out


EVAL_KEYS = ("total_reward", "resolution_rate_root", "hir_root", "hir_instance", "human_routing_rate",
             "human_busy_minutes", "human_utilization", "abandonment_rate", "missed_deadline_rate", "false_ai_rate",
             "online_wait_p90", "post_ai_failure_hangups", "handoff_minutes_added", "escalations")


def evaluate(cfgs: dict[str, SimulationConfig], policies: dict[str, tuple[str, dict]], workers: int,
             seeds: Sequence[int] = PHASE_D_HEADROOM_SEEDS) -> dict[str, Any]:
    seeds = require_seeds(seeds, PHASE_D_HEADROOM_SEEDS, "Phase D evaluation")
    jobs = [(cfg, name, params, s, f"{cid}|{label}") for cid, cfg in cfgs.items()
            for label, (name, params) in policies.items() for s in seeds]
    rows = pmap(day_job, jobs, workers)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["tag"]].append(r)
    table: dict[str, dict[str, Any]] = {}
    for tag, rs in by.items():
        cid, label = tag.split("|")
        table.setdefault(cid, {})[label] = (
            {k: {"mean": _mean(rs, k), "std": _std(rs, k)} for k in EVAL_KEYS}
            | {"action_share": {a.value: float(np.mean([r["action_share"][a.value] for r in rs])) for a in ACTION_ORDER},
               "per_seed_reward": {str(r["seed"]): r["total_reward"] for r in rs},
               "reward": {k: float(np.mean([r["reward"][k] for r in rs])) for k in rs[0]["reward"]}}
        )
    return {"seeds": list(seeds), "policies": {k: list(v) for k, v in policies.items()}, "table": table}


# ---------------------------------------------------------------------------
# headroom
# ---------------------------------------------------------------------------
def sample_job(job: tuple[SimulationConfig, str, int, dict, dict, float]):
    cfg, label, seed, rule, quota, theta = job
    return sample_states(cfg, label, seed, rule, quota, theta)


def build_states(cfgs: dict[str, SimulationConfig], seeds: Sequence[int], rule_by_cfg: dict[str, dict],
                 quota: dict[str, int], theta_by_cfg: dict[str, float], workers: int) -> dict[str, Any]:
    require_seeds(seeds, PHASE_D_HEADROOM_SEEDS + PHASE_D_EXTENSION_SEEDS, "Phase D headroom sampling")
    jobs = [(cfg, cid, s, rule_by_cfg[cid], quota, theta_by_cfg[cid]) for cid, cfg in cfgs.items() for s in seeds]
    out: dict[str, Any] = {cid: {"states": [], "meta": []} for cid in cfgs}
    for (cfg, cid, s, *_), (states, meta) in zip(jobs, pmap(sample_job, jobs, workers)):
        out[cid]["states"] += states
        out[cid]["meta"].append(meta)
    return out


def run_headroom(cfgs: dict[str, SimulationConfig], samples: dict[str, Any], rule_by_cfg: dict[str, dict],
                 continuation_by_cfg: dict[str, tuple[str, dict]], k: int, workers: int) -> dict[str, Any]:
    jobs, owner = [], []
    for cid, cfg in cfgs.items():
        js = H.make_jobs(cfg, samples[cid]["states"], rule_by_cfg[cid], continuation_by_cfg[cid], k)
        jobs += js
        owner += [cid] * len(js)
    results = pmap(run_branch, jobs, workers)
    per: dict[str, list] = defaultdict(list)
    for cid, r in zip(owner, results):
        per[cid].append(r)
    out = {}
    for cid in cfgs:
        states = samples[cid]["states"]
        analyses = H.analyze(states, per[cid], k)
        summary = H.summarize(states, analyses, samples[cid]["meta"], per[cid], f"D|{cid}|{k}")
        out[cid] = {"summary": summary | {"classification": classify(summary)}, "analyses": analyses,
                    "results": per[cid]}
    return out


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    """Pre-registered interpretation rule (rationale section 9)."""
    ci = summary["regret"]["RulePolicy"]["cross_fitted_ci"]
    r, lo = ci["mean"], ci["lo"]
    if lo > 0 and r >= MEANINGFUL:
        label = "CLEARLY_POSITIVE"
    elif (lo > 0 and r < MEANINGFUL) or (r >= MEANINGFUL and lo <= 0):
        label = "EMERGING"
    else:
        label = "NEGLIGIBLE"
    mean_j = summary["mean_J"]["RulePolicy_action"]
    return {"label": label, "R": r, "ci_lo": lo, "ci_hi": ci["hi"],
            "relative_to_mean_J": r / mean_j if mean_j else None,
            "near_tie_lt_5": summary["near_tie_fractions"]["lt_5"], "tie_share": summary["label_distribution"]["TIE"]}


def select_confirmatory(screen: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered D3 selection rules (rationale section 8)."""
    cells = {c.cell_id: c for c in GRID}
    chosen = [(V1_CELL.cell_id, "V1 reference"), (MAX_CELL.cell_id, "maximum severity")]
    positive = sorted((cells[cid] for cid, s in screen.items()
                       if s["classification"]["label"] in ("EMERGING", "CLEARLY_POSITIVE")), key=lambda c: c.order_key)
    boundary = positive[0] if positive else MID_CELL
    chosen.append((boundary.cell_id, "boundary (lowest-severity positive)" if positive else "boundary fallback: mid-grid"))
    ranked = sorted(screen, key=lambda cid: (-screen[cid]["classification"]["R"], cells[cid].order_key))
    taken = {c for c, _ in chosen}
    high = next(cid for cid in ranked if cid not in taken)
    chosen.append((high, "highest screening cross-fitted Rule regret"))
    return {"cells": chosen, "positive_screened": [c.cell_id for c in positive]}


# ---------------------------------------------------------------------------
# learnability (diagnostic only; never a routing policy)
# ---------------------------------------------------------------------------
def learnability(states: Sequence[dict[str, Any]], analyses: Sequence[dict[str, Any]], key: str,
                 ridge: float = 1.0) -> dict[str, Any]:
    """Leave-one-day-out ridge prediction of J(a) - J(Rule action) from the 43-D encoded observation.

    Labels use futures 0..K/2-1, evaluation uses the other futures on held-out days, so neither the day nor
    the future realisation used for scoring was seen in training.
    """
    by_id = {s["state_id"]: s for s in states}
    ok = [a for a in analyses if a["status"] == "OK"]
    if not ok:
        return {"n": 0}
    K = len(next(iter(ok[0]["returns"].values())))
    tr_f, ev_f = slice(0, K // 2), slice(K // 2, K)
    X = np.array([by_id[a["state_id"]]["encoded"] for a in ok], dtype=float)
    X = np.hstack([X, np.ones((len(ok), 1))])
    days = np.array([by_id[a["state_id"]]["seed"] for a in ok])
    rule = [by_id[a["state_id"]]["rule_action"] for a in ok]

    def target(i, a, part):
        g = ok[i]["returns"]
        return float(np.mean(np.subtract(g[a][part], g[rule[i]][part])))

    gains, changed, gains_ai = np.zeros(len(ok)), np.zeros(len(ok), dtype=bool), np.zeros(len(ok))
    for d in np.unique(days):
        train, test = np.flatnonzero(days != d), np.flatnonzero(days == d)
        models = {}
        for a in (x.value for x in ACTION_ORDER):
            rows = [i for i in train if a in ok[i]["returns"]]
            if len(rows) < 5:
                continue
            A, y = X[rows], np.array([target(i, a, tr_f) for i in rows])
            models[a] = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y)
        for i in test:
            valid = list(ok[i]["returns"])
            pred = {a: (float(X[i] @ models[a]) if a in models else 0.0) for a in valid}
            pred[rule[i]] = 0.0
            choice = max(valid, key=lambda a: (pred[a], a == rule[i]))
            changed[i] = choice != rule[i]
            gains[i] = target(i, choice, ev_f)
            gains_ai[i] = target(i, "AI", ev_f) if "AI" in valid else 0.0
    ci = cluster_bootstrap_mean(gains, list(days), key + "learn")
    return {"n": len(ok), "learnable_gain_vs_rule": ci, "share_changed_from_rule": float(changed.mean()),
            "always_ai_gain_vs_rule": cluster_bootstrap_mean(gains_ai, list(days), key + "ai"),
            "ridge": ridge, "label_futures": f"0..{K // 2 - 1}", "eval_futures": f"{K // 2}..{K - 1}",
            "feature_separation": feature_separation(states, ok)}


def _feature(st: dict[str, Any], name: str) -> float:
    o = st["observation"]
    if name == "urgency_pred":
        return float([u.value for u in Urgency].index(o["urgency_pred"]))
    if name == "request_type":
        return 1.0 if o["request_type"] == "COMPLAINT_INCIDENT" else 0.0  # complaint indicator
    if name == "time":
        return float(st["timestamp"])
    return float(o[name])


def feature_separation(states, ok) -> dict[str, Any]:
    """Standardised mean difference (HUMAN_BEST - AI_BEST) of observable features, descriptive only."""
    by_id = {s["state_id"]: s for s in states}
    groups = {lab: [by_id[a["state_id"]] for a in ok if a["label"] == lab] for lab in ("AI_BEST", "HUMAN_BEST")}
    out = {"n_AI_BEST": len(groups["AI_BEST"]), "n_HUMAN_BEST": len(groups["HUMAN_BEST"]), "smd": {}}
    if min(len(g) for g in groups.values()) < 3:
        return out
    for f in LEARN_FEATURES:
        a = np.array([_feature(s, f) for s in groups["AI_BEST"]])
        h = np.array([_feature(s, f) for s in groups["HUMAN_BEST"]])
        sd = np.sqrt((a.var(ddof=1) + h.var(ddof=1)) / 2) or 1.0
        out["smd"][f] = float((h.mean() - a.mean()) / sd)
    return out


# ---------------------------------------------------------------------------
# scenario period analysis (hidden windows used for ANALYSIS only)
# ---------------------------------------------------------------------------
def period_job(job: tuple[SimulationConfig, str, dict, int, str]) -> dict[str, Any]:
    cfg, name, params, seed, kind = job
    result = run_simulation(cfg, make_policy(name, params, seed), seed=seed)
    window = event_windows(cfg, seed)[kind]
    per: dict[str, list] = defaultdict(list)
    for t in result.transitions:
        per[period_of(t.timestamp, window)].append(t)
    out = {"seed": seed, "policy": name, "window": window, "total_reward": result.total_reward, "periods": {}}
    for p, ts in per.items():
        n = len(ts)
        out["periods"][p] = {
            "decisions": n,
            "action_share": {a.value: sum(t.action is a for t in ts) / n for a in ACTION_ORDER},
            "recent_ai_failure_rate": float(np.mean([t.state.recent_ai_failure_rate for t in ts])),
            "human_queue_size": float(np.mean([t.state.human_queue_size for t in ts])),
            "forecast_30min": float(np.mean([t.state.forecast_30min for t in ts])),
            "recent_arrival_rate": float(np.mean([t.state.recent_arrival_rate for t in ts])),
            "reward_per_decision": float(np.mean([t.reward for t in ts])),
        }
    return out


def periods_of_states(states: Sequence[dict[str, Any]], analyses: Sequence[dict[str, Any]], cfg: SimulationConfig,
                      kind: str) -> dict[str, Any]:
    windows = {s: event_windows(cfg, s)[kind] for s in {st["seed"] for st in states}}
    by_id = {s["state_id"]: s for s in states}
    groups: dict[str, list] = defaultdict(list)
    for a in analyses:
        if a["status"] == "OK":
            st = by_id[a["state_id"]]
            groups[period_of(st["timestamp"], windows[st["seed"]])].append(a)
    out = {}
    for p, rows in groups.items():
        labels = Counter(r["label"] for r in rows)
        out[p] = {"n": len(rows), "labels": {l: labels.get(l, 0) / len(rows) for l in H.LABELS},
                  "rule_regret_raw": float(np.mean([r["regret"]["RulePolicy"]["raw"] for r in rows])),
                  "rule_regret_cross_fitted": float(np.mean([r["regret"]["RulePolicy"]["cross_fitted"] for r in rows])),
                  "aifirst_regret_cross_fitted": float(np.mean([r["regret"]["AIFirstPolicy"]["cross_fitted"] for r in rows])),
                  "mean_adv_human_vs_ai": float(np.mean([r["advantage_vs_ai"]["HUMAN"]["mean"] for r in rows
                                                         if "HUMAN" in r["advantage_vs_ai"]] or [0.0]))}
    return {"windows": {str(k): v for k, v in windows.items()}, "periods": out}


def check_cells_from_ids(ids: Sequence[str]) -> list[Cell]:
    by = {c.cell_id: c for c in GRID}
    return [by[i] for i in ids]


assert set(BASELINE_NAMES) == {"RandomPolicy", "AIFirstPolicy", "ThresholdPolicy", "RulePolicy", "BestFixedPolicy"}
