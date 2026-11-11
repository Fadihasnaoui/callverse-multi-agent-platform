"""Phase E.1 checkpoint evaluation, development evaluation, stability criterion and diagnostics.

The PRIMARY verdict reuses the Phase E implementation (``phase_e.evaluation``) unchanged; only the labels
and the development day set (1681-1760) differ.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

import numpy as np

from callverse_workforce.baselines.metrics import policy_day_metrics
from callverse_workforce.domain.enums import ACTION_ORDER
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.phase_e.scenarios import DaySpec
from callverse_workforce.twin.runner import run_simulation

SUCCESS = "PPO V1.1 E.1 SUCCESS — COMPETITIVE WITH STRONG BASELINES"
PARTIAL = "PPO V1.1 E.1 PARTIAL SUCCESS — VALID BUT BELOW RULE"
FAILURE = "PPO V1.1 E.1 FAILURE — STABILIZATION INSUFFICIENT"
STABLE_MIN_SEEDS = 4
STABLE_MAX_SD = 13.5
LABEL_MAP = {EV.SUCCESS: SUCCESS, EV.PARTIAL: PARTIAL, EV.FAILURE: FAILURE}
_CELLS = {c.cell_id: c for c in MIXTURE_CELLS}

BINS = {
    "human_utilization": [(0.0, 0.5, "<50%"), (0.5, 1.0, "50-<100%"), (1.0, 9.0, "100%")],
    "human_queue_size": [(0, 1, "0"), (1, 3, "1-2"), (3, 1e9, "3+")],
    "humans_available": [(0, 1, "0"), (1, 3, "1-2"), (3, 1e9, "3+")],
    "forecast_30min": [(0, 30, "<30"), (30, 45, "30-45"), (45, 1e9, "45+")],
    "deadline_remaining_pred": [(-1e9, 180, "<3h"), (180, 1440, "3h-24h"), (1440, 1e18, ">24h")],
    "ai_resolution_score": [(0.0, 0.4, "<0.4"), (0.4, 0.7, "0.4-0.7"), (0.7, 1.01, ">=0.7")],
    "recent_ai_failure_rate": [(0.0, 0.2, "<0.2"), (0.2, 0.4, "0.2-0.4"), (0.4, 0.6, "0.4-0.6"), (0.6, 1.01, ">=0.6")],
}


def _bin(name: str, x: float) -> str:
    for lo, hi, label in BINS[name]:
        if lo <= x < hi:
            return label
    return "other"


def decision_diagnostics(result) -> dict[str, dict[str, dict[str, int]]]:
    """Action counts conditioned on observable features (counts; aggregate by summation)."""
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for t in result.transitions:
        o = t.state
        a = t.action.value
        for name in BINS:
            b = _bin(name, getattr(o, name))
            out[name][b][a] += 1
            out[name][b]["n"] += 1
        out["urgency_pred"][o.urgency_pred.value][a] += 1
        out["urgency_pred"][o.urgency_pred.value]["n"] += 1
    return {k: {b: dict(v) for b, v in d.items()} for k, d in out.items()}


def eval_job(job: tuple[str, str, Any, str, tuple[str, str, int]]) -> dict[str, Any]:
    """Same metrics as Phase E (EV.eval_job) plus decision diagnostics."""
    label, kind, ref, base_path, (family, cell_id, seed) = job
    day = DaySpec(family, _CELLS[cell_id], seed)
    cfg = EV._config(base_path, day)
    result = run_simulation(cfg, EV._policy(kind, ref, seed), seed=seed)
    d = policy_day_metrics(result) | {"label": label, "family": family, "cell": cell_id}
    d["visible_ai_failures"] = sum(1 for e in result.events if e.event_type.value == "AI_VISIBLE_FAILURE")
    d["false_ai_count"] = sum(1 for e in result.events if e.event_type.value == "AI_FALSE_RESOLUTION")
    d["adaptation"] = EV.adaptation(result, cfg, family, seed)
    d["decision_diagnostics"] = decision_diagnostics(result)
    return d


def evaluate(policies: dict[str, tuple[str, Any]], days: Sequence[DaySpec], purpose: str, base_path: str,
             workers: int) -> dict[str, list[dict[str, Any]]]:
    days = EV.days_for(purpose, days)
    jobs = [(label, kind, ref, base_path, (d.family, d.cell.cell_id, d.seed))
            for label, (kind, ref) in policies.items() for d in days]
    rows = EV.pmap(eval_job, jobs, workers)
    out: dict[str, list] = defaultdict(list)
    for r in rows:
        out[r["label"]].append(r)
    return dict(out)


def checkpoint_job(job: tuple[str, str, tuple[str, str, int]]) -> dict[str, Any]:
    """Deterministic raw validation reward of one checkpoint on one day (same convention as Phase E)."""
    path, base_path, (family, cell_id, seed) = job
    r = run_simulation(EV._config(base_path, DaySpec(family, _CELLS[cell_id], seed)),
                       EV._policy("ppo", path, seed), seed=seed)
    share = {a.value: 0 for a in ACTION_ORDER}
    for d in r.decisions:
        share[d.action.value] += 1
    n = max(1, len(r.decisions))
    return {"path": path, "seed": seed, "family": family, "raw_reward": r.total_reward,
            "action_share": {k: v / n for k, v in share.items()}}


def evaluate_checkpoints(paths: Sequence[str], days: Sequence[DaySpec], base_path: str, workers: int) -> dict[str, Any]:
    EV.days_for("validation", days)
    jobs = [(p, base_path, (d.family, d.cell.cell_id, d.seed)) for p in paths for d in days]
    rows = EV.pmap(checkpoint_job, jobs, workers, chunksize=8)
    by: dict[str, list] = defaultdict(list)
    for r in rows:
        by[r["path"]].append(r)
    return {p: {"val_mean_raw_reward": float(np.mean([r["raw_reward"] for r in by[p]])), "n_days": len(by[p]),
                "action_share": {a.value: float(np.mean([r["action_share"][a.value] for r in by[p]])) for a in ACTION_ORDER},
                "family_mean": {f: float(np.mean([r["raw_reward"] for r in by[p] if r["family"] == f]))
                                for f in sorted({r["family"] for r in by[p]})}}
            for p in paths}


def primary_verdict(ppo_mean, rule, random_, aifirst) -> dict[str, Any]:
    v = EV.verdict(ppo_mean, rule, random_, aifirst)  # identical Phase E implementation
    return v | {"phase_e_label": v["verdict"], "verdict": LABEL_MAP[v["verdict"]]}


def stability(seed_rows: Sequence[Sequence[dict[str, Any]]], rule: dict[int, float]) -> dict[str, Any]:
    ni = EV.seed_noninferiority(seed_rows, rule)  # identical per-seed implementation as Phase E
    means = [p["mean"] for p in ni["per_seed"]]
    sd = float(np.std(means, ddof=1))
    stable = ni["noninferior_seed_count"] >= STABLE_MIN_SEEDS and sd <= STABLE_MAX_SD
    return ni | {"seed_means": means, "seed_sd": sd, "seed_range": float(max(means) - min(means)),
                 "criterion": f">= {STABLE_MIN_SEEDS}/5 seeds non-inferior AND seed SD <= {STABLE_MAX_SD}",
                 "status": "STABLE" if stable else "NOT STABLE"}


def interpretation(primary: str, status: str) -> str:
    success = primary == SUCCESS
    if success and status == "STABLE":
        return ("Phase E failed the original pre-registered evaluation. Phase E.1, a separately pre-registered "
                "training-only experiment, showed that PPO can become both competitive and stable without changing "
                "the routing problem.")
    if success:
        return ("PPO became competitive on average, but training reliability remains insufficient because seed "
                "sensitivity persists.")
    if status == "STABLE":
        return ("Training stabilization succeeded, but the remaining performance gap is consistent with the limited "
                "structural headroom previously identified in Phases C and D.")
    return ("The tested optimization improvements were insufficient. Combined with Phase C/D headroom results, this "
            "strengthens the motivation for a V2 routing formulation.")


def sum_diagnostics(rows: Sequence[dict[str, Any]], family: str | None = None) -> dict[str, Any]:
    agg: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for r in rows:
        if family and r["family"] != family:
            continue
        for feat, bins in r["decision_diagnostics"].items():
            for b, counts in bins.items():
                for k, v in counts.items():
                    agg[feat][b][k] += v
    return {f: {b: {"n": c["n"], **{a: c.get(a, 0) / c["n"] for a in ("AI", "HUMAN", "WAIT", "CALLBACK")}}
                for b, c in sorted(bins.items())} for f, bins in agg.items()}
