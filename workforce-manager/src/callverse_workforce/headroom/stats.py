"""Small-sample paired statistics for forced-action headroom.

For a state s with valid actions A(s) and K paired futures, G[a][k] is the discounted return of action a
under future k (the same future k for every action -> common random numbers).

* J(s,a) = mean_k G[a][k].
* Best/second = top two actions by J. D_k = G[best][k] - G[second][k] (paired).
* Classification: Student-t 95% interval on mean(D) with K-1 df (exact under normality of D; K=8 is too small
  for asymptotics). If the lower bound > 0 -> "<BEST>_BEST", else "TIE". A paired percentile bootstrap
  interval is reported alongside (not used for classification: with K=8 it is anti-conservative).
* Raw regret_pi(s) = J_max - J(pi(s)) is biased UPWARD (winner's curse: max of noisy means). We therefore
  also report
    - clear regret: raw regret if the paired t-interval of G[best]-G[pi] excludes 0, else 0;
    - cross-fitted regret: choose argmax on futures {0..3}, evaluate on {4..7} (and vice versa), average.
      It is unbiased for the gain of the noisy-argmax rule over pi and can be negative.
"""

from __future__ import annotations

import zlib
from typing import Any, Sequence

import numpy as np

# two-sided 97.5% Student-t quantiles by degrees of freedom
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 24: 2.064, 29: 2.045}
NEAR_TIE = (5.0, 10.0, 25.0)
BOOTSTRAP_B = 2000


def t_quantile(df: int) -> float:
    if df in T975:
        return T975[df]
    return T975[max(k for k in T975 if k <= df)] if df < 30 else 1.96


def paired_interval(diffs: Sequence[float], seed_key: str = "") -> dict[str, float]:
    d = np.asarray(diffs, dtype=float)
    k = d.size
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if k > 1 else 0.0
    half = t_quantile(k - 1) * sd / np.sqrt(k) if k > 1 else float("inf")
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))
    boot = d[rng.integers(0, k, size=(BOOTSTRAP_B, k))].mean(axis=1)
    return {"mean": mean, "sd": sd, "t_lo": mean - half, "t_hi": mean + half,
            "boot_lo": float(np.percentile(boot, 2.5)), "boot_hi": float(np.percentile(boot, 97.5))}


def clearly_positive(diffs: Sequence[float]) -> bool:
    return paired_interval(diffs)["t_lo"] > 0


def analyze_state(returns: dict[str, Sequence[float]], policy_actions: dict[str, str], state_id: str = "") -> dict[str, Any]:
    """returns: action -> K paired discounted returns (valid actions only)."""
    acts = list(returns)
    G = {a: np.asarray(returns[a], dtype=float) for a in acts}
    K = len(next(iter(G.values())))
    if any(len(v) != K for v in G.values()):
        raise ValueError("all actions need the same K paired futures")
    J = {a: float(G[a].mean()) for a in acts}
    order = sorted(acts, key=lambda a: (-J[a], acts.index(a)))
    best, second = order[0], order[1]
    gap = paired_interval(G[best] - G[second], state_id)
    label = f"{best}_BEST" if gap["t_lo"] > 0 else "TIE"
    out: dict[str, Any] = {
        "J": J,
        "J_sd": {a: float(G[a].std(ddof=1)) for a in acts},
        "best": best,
        "second": second,
        "gap": gap,
        "label": label,
        "near_tie": {f"lt_{int(x)}": (J[best] - J[second]) < x for x in NEAR_TIE},
        "advantage_vs_ai": {},
        "regret": {},
    }
    if "AI" in G:
        for a in acts:
            if a != "AI":
                iv = paired_interval(G[a] - G["AI"], state_id + a)
                out["advantage_vs_ai"][a] = {"mean": iv["mean"], "t_lo": iv["t_lo"], "t_hi": iv["t_hi"],
                                             "clearly_positive": iv["t_lo"] > 0, "clearly_negative": iv["t_hi"] < 0}
    half = K // 2
    folds = (np.arange(half), np.arange(half, K))
    for name, a_pi in policy_actions.items():
        raw = J[best] - J[a_pi]
        clear = raw if (a_pi != best and paired_interval(G[best] - G[a_pi])["t_lo"] > 0) else 0.0
        cf = []
        for sel, ev in (folds, folds[::-1]):
            choice = max(acts, key=lambda a: (G[a][sel].mean(), -acts.index(a)))
            cf.append(float(G[choice][ev].mean() - G[a_pi][ev].mean()))
        out["regret"][name] = {"action": a_pi, "raw": raw, "clear": clear, "cross_fitted": float(np.mean(cf)),
                               "is_best": a_pi == best}
    return out


def describe(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def cluster_bootstrap_mean(values: Sequence[float], clusters: Sequence[Any], key: str, b: int = 4000) -> dict[str, float]:
    """95% CI of a mean, resampling whole days (clusters) to respect within-day correlation."""
    vals = np.asarray(values, dtype=float)
    labels = list(dict.fromkeys(clusters))
    idx = {c: np.flatnonzero(np.asarray(clusters) == c) for c in labels}
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    means = []
    for _ in range(b):
        pick = rng.integers(0, len(labels), size=len(labels))
        sel = np.concatenate([idx[labels[i]] for i in pick])
        means.append(vals[sel].mean())
    return {"mean": float(vals.mean()), "lo": float(np.percentile(means, 2.5)), "hi": float(np.percentile(means, 97.5))}
