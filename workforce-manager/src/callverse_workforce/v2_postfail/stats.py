"""V2 headroom statistics (spec §8). Reuses the frozen Phase C / targeted-audit functions unchanged:
``headroom.stats.analyze_state`` (labels, cross-fitted regret), ``clearly_positive`` / ``paired_interval`` / ``T975``
and ``cluster_bootstrap_mean``; thresholds are the ``targeted/audit.py`` constants."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from callverse_workforce.headroom.stats import analyze_state, cluster_bootstrap_mean, clearly_positive, describe
from callverse_workforce.targeted.audit import ADV_MEANINGFUL, CLEAR_MEANINGFUL, CLEAR_WEAK
from callverse_workforce.v2_postfail.actions import V2_ACTIONS

MIN_STATES_FOR_POWER = 1000


def _G(returns: dict[str, Sequence[float]]) -> dict[str, np.ndarray]:
    order = [a for a in V2_ACTIONS if a in returns]  # fixed action order => deterministic tie-breaks
    return {a: np.asarray(returns[a], dtype=float) for a in order}


def folds(K: int):
    """The authoritative Phase C split / swap of headroom.stats.analyze_state."""
    half = K // 2
    f = (np.arange(half), np.arange(half, K))
    return (f, f[::-1])


def h1_heldout(returns: dict[str, Sequence[float]], baseline_action: str) -> float:
    """R2 (spec §8.2a): per fold, select ONE alternative on the selection fold, test it vs the baseline action with
    the frozen ``clearly_positive`` on the held-out fold only; average the two out-of-fold results (0, 0.5, 1)."""
    G = _G(returns)
    alts = [a for a in G if a != baseline_action]
    if not alts or baseline_action not in G:
        return 0.0
    K = len(G[baseline_action])
    out = []
    for sel, ev in folds(K):
        a_star = max(alts, key=lambda a: (G[a][sel].mean(), -alts.index(a)))  # frozen argmax tie-break pattern
        out.append(1.0 if clearly_positive(G[a_star][ev] - G[baseline_action][ev]) else 0.0)
    return float(np.mean(out))


def analyze(states: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    """states: [{state_id, day, family, urgency, baseline_action, returns{action: [K returns]}}] for ONE comparison row."""
    rows = []
    for s in states:
        res = analyze_state(dict(_G(s["returns"])), {"X": s["baseline_action"]}, s["state_id"])
        rows.append({"state_id": s["state_id"], "day": s["day"], "family": s["family"], "urgency": s["urgency"],
                     "h1": h1_heldout(s["returns"], s["baseline_action"]), "adv": res["regret"]["X"]["cross_fitted"],
                     "label": res["label"], "best": res["best"], "gap": res["J"][res["best"]] - res["J"][res["second"]],
                     "baseline_action": s["baseline_action"]})
    out = summarize(rows, key)
    out["rows"] = rows
    return out


def summarize(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n_states": 0, "classification": {"label": "NONE", "reason": "no states"}}
    days = [r["day"] for r in rows]
    h1 = [r["h1"] for r in rows]
    adv = [r["adv"] for r in rows]
    out = {"n_states": n, "unique_days": len(set(days)),
           "h1_share": float(np.mean(h1)), "h1_ci": cluster_bootstrap_mean(h1, days, key + "|h1"),
           "h2_mean_adv": float(np.mean(adv)), "h3_ci": cluster_bootstrap_mean(adv, days, key + "|adv"),
           "best_action_raw": dict(Counter(r["best"] for r in rows)), "labels": dict(Counter(r["label"] for r in rows)),
           "gap": describe([r["gap"] for r in rows]), "share_gap_gt_0_5": float(np.mean([r["gap"] > 0.5 for r in rows])),
           "by_family": _strata(rows, "family"), "by_urgency": _strata(rows, "urgency")}
    out["classification"] = classify(out)
    return out


def _strata(rows, field) -> dict[str, Any]:
    g = defaultdict(list)
    for r in rows:
        g[r[field]].append(r)
    return {k: {"n": len(v), "h1_share": float(np.mean([x["h1"] for x in v])), "mean_adv": float(np.mean([x["adv"] for x in v]))}
            for k, v in sorted(g.items())}


def classify(s: dict[str, Any]) -> dict[str, Any]:
    h1 = s["h1_share"] >= CLEAR_MEANINGFUL
    h2 = s["h2_mean_adv"] > ADV_MEANINGFUL
    h3 = s["h3_ci"]["lo"] > 0
    if h1 and h2 and h3:
        label = "MEANINGFUL"
    elif s["h1_share"] >= CLEAR_WEAK or (h2 and h3):
        label = "WEAK"
    else:
        label = "NONE"
    return {"label": label, "H1": h1, "H2": h2, "H3": h3, "limited_power": s["n_states"] < MIN_STATES_FOR_POWER}


def mc_stability(primary_rows: Sequence[dict[str, Any]], k64_rows: Sequence[dict[str, Any]], key: str = "") -> dict[str, Any]:
    """Pre-registered stability rule (spec §8.3): on the SAME pre-selected 25 % subsample of states, compute H1, H2 and
    H3 from the K = 32 rows and from the K = 64 rows; if ANY of the three pass/fail verdicts differs -> UNSTABLE."""
    b = {r["state_id"]: r for r in k64_rows}
    sub32 = [r for r in primary_rows if r["state_id"] in b]
    sub64 = [b[r["state_id"]] for r in sub32]
    if not sub32 or len(sub32) != len(b):
        return {"n": len(sub32), "status": "UNSTABLE", "mc_unstable": True, "reason": "subsample mismatch or empty"}
    c32, c64 = classify(summarize(sub32, key + "|k32sub")), classify(summarize(sub64, key + "|k64sub"))
    diff = [h for h in ("H1", "H2", "H3") if c32[h] != c64[h]]
    unstable = bool(diff)
    return {"n": len(sub32), "k32_verdicts": {h: c32[h] for h in ("H1", "H2", "H3")},
            "k64_verdicts": {h: c64[h] for h in ("H1", "H2", "H3")}, "differing": diff,
            "status": "UNSTABLE" if unstable else "STABLE", "mc_unstable": unstable}


def decision_gate(class_b1: dict, class_b2: dict, mc_b1: dict, mc_b2: dict, aging_violations: int) -> dict[str, Any]:
    m1, m2 = class_b1["label"] == "MEANINGFUL", class_b2["label"] == "MEANINGFUL"
    stable = not (mc_b1["mc_unstable"] or mc_b2["mc_unstable"])
    aging_ok = aging_violations == 0
    if not stable:
        verdict = "UNSTABLE — PPO V2 TRAINING NOT JUSTIFIED"
    elif m1 and m2 and aging_ok:
        verdict = "PPO V2 TRAINING CAN BE CONSIDERED"
    elif m1 and not m2:
        verdict = "THE CONTEXTUAL RULE ALREADY CAPTURES THE MEASURED ADVANTAGE — PPO V2 TRAINING NOT JUSTIFIED"
    else:
        verdict = "PPO V2 TRAINING NOT YET JUSTIFIED"
    return {"meaningful_vs_B1": m1, "meaningful_vs_B2": m2, "mc_stable": stable, "aging_guarantee": aging_ok, "verdict": verdict}
