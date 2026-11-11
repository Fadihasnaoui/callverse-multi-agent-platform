"""Targeted initial-routing headroom audit for WAIT and CALLBACK (diagnostic only).

Frozen environment (Phase E V1.1 mixture x natural scenario distribution), frozen RulePolicy behaviour,
Phase C counterfactual replay (``headroom.branch``) and Phase C statistics (``headroom.stats``). States are
selected with OBSERVABLE routing-time fields only (``RoutingObservation`` and the action mask).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from callverse_workforce.config import SimulationConfig
from callverse_workforce.domain.enums import ACTION_ORDER
from callverse_workforce.headroom.branch import BranchJob, StateKey, policy_spec
from callverse_workforce.headroom.sampling import trajectory
from callverse_workforce.headroom.stats import analyze_state, cluster_bootstrap_mean, paired_interval
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e.scenarios import MIXTURE_PROBABILITIES, TRAINING_DISTRIBUTION, DaySpec, day_config

K_FUTURES = 16
TARGET_N = 300
MAX_INITIAL_STATES = 100_000
LOW_SAMPLE = 100
BLOCK = 10_000
SEED_FLOOR = 300_000
CLEAR_MEANINGFUL = 0.10
CLEAR_WEAK = 0.05
ADV_MEANINGFUL = 0.5

# Pre-registered filters: (implementation field / predicate) -- every field is part of RoutingObservation or mask.
OBSERVABLE_FIELD_MAPPING = {
    "WAIT valid under the existing action mask": "mask['WAIT']",
    "humans_available == 0": "observation.humans_available",
    "human_queue_size <= 2 / >= 5": "observation.human_queue_size",
    "predicted urgency is NOT HIGH": "observation.urgency_pred",
    "deadline_remaining_pred >= 60 / >= 120 min": "observation.deadline_remaining_pred",
    "waiting_time_so_far <= 5 min": "observation.waiting_time_so_far",
    "ai_resolution_score <= 0.50": "observation.ai_resolution_score",
    "oldest_human_wait <= 5 min": "observation.oldest_human_wait (exists exactly; observable centre state)",
    "CALLBACK valid": "mask['CALLBACK']",
    "callback_allowed == True": "observation.callback_allowed",
    "callback service currently available": "no 'callback_unavailable' in mask reasons (ServiceAvailability: callback "
                                            "service enabled and pending < max_pending; observable at routing time)",
    "callbacks_pending <= 5": "observation.callbacks_pending",
    "oldest_callback_age <= 30 min": "observation.oldest_callback_age (exists exactly; observable centre state)",
    "INITIAL routing decision": "observation.routing_attempt_count == 0 and contact is not a recontact instance "
                                "(contact_id has no '-R' suffix); WAIT re-evaluations have routing_attempt_count >= 1",
}
UNAVAILABLE_FIELDS: list[str] = []  # every requested field exists exactly as an observable field


def is_initial(rec: dict[str, Any]) -> bool:
    return rec["observation"]["routing_attempt_count"] == 0 and "-R" not in rec["contact_id"]


def wait_favourable(rec: dict[str, Any]) -> bool:
    o, m = rec["observation"], rec["mask"]
    return (m["WAIT"] and o["humans_available"] == 0 and o["human_queue_size"] <= 2 and o["urgency_pred"] != "HIGH"
            and o["deadline_remaining_pred"] >= 60 and o["waiting_time_so_far"] <= 5 and o["ai_resolution_score"] <= 0.50
            and o["oldest_human_wait"] <= 5)


def callback_favourable(rec: dict[str, Any]) -> bool:
    o, m = rec["observation"], rec["mask"]
    return (m["CALLBACK"] and o["callback_allowed"] and "callback_unavailable" not in rec["mask_reasons"].get("CALLBACK", [])
            and o["humans_available"] == 0 and o["human_queue_size"] >= 5 and o["urgency_pred"] != "HIGH"
            and o["deadline_remaining_pred"] >= 120 and o["ai_resolution_score"] <= 0.50 and o["callbacks_pending"] <= 5
            and o["oldest_callback_age"] <= 30)


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------
def day_spec(seed: int) -> DaySpec:
    """Natural (frozen Phase E training) scenario distribution; family/cell drawn from a seed-keyed stream."""
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0x7A,)))
    fams = list(TRAINING_DISTRIBUTION)
    fam = fams[int(rng.choice(len(fams), p=np.array([TRAINING_DISTRIBUTION[f] for f in fams])))]
    cell = MIXTURE_CELLS[int(rng.choice(4, p=np.array(MIXTURE_PROBABILITIES)))]
    return DaySpec(fam, cell, seed)


def known_seed_sources(root: Path) -> dict[str, set[int]]:
    """Every day seed any previous experiment used or could have used (logged + regenerated sampler streams)."""
    from callverse_workforce.phase_e.scenarios import TrainingDaySampler
    from callverse_workforce.phase_e1.training import worker_sampler_seed
    from callverse_workforce.seeds import FINAL_TEST_RESERVED, SEED_SETS

    src: dict[str, set[int]] = {name: set(v) for name, v in SEED_SETS.items()}
    src["final_test_reserved"] = set(FINAL_TEST_RESERVED)
    src["phase_e_audit_days"] = set(range(100_001, 100_011))
    src["ppo_init_seeds"] = {0, 1, 100, 101, 102, 103, 104, 200, 201, 202, 500, 501, 502, 503, 504}
    logged: set[int] = set()
    for p in list((root / "artifacts" / "ppo_v1_1").rglob("training_episodes.jsonl")) + \
            list((root / "artifacts" / "ppo_v1_1_e1").rglob("training_episodes.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                logged.add(int(json.loads(line)["day"]["seed"]))
    src["logged_training_days"] = logged

    def stream(sampler_seed: int, n: int) -> set[int]:
        s = TrainingDaySampler(sampler_seed)
        return {s().seed for _ in range(n)}

    regen: set[int] = set()
    for ps in (0, 1, 100, 101, 102, 103, 104, 200, 201, 202):  # Phase E runs: TrainingDaySampler(seed*7919+13)
        regen |= stream(ps * 7919 + 13, 2000)
    for ps in (500, 501, 502, 503, 504, 999, 9999):  # E.1 runs + E.1 benchmarks: 8 workers each
        for w in range(8):
            regen |= stream(worker_sampler_seed(ps, w), 1000)
    for ss in (424_242, 9, 10, 1, 2, 3, 5, 11, 12, 13, 14, 15, 16, 17, 18):  # delta_t sample, benchmarks, tests
        regen |= stream(ss, 1000)
    src["regenerated_sampler_streams"] = regen
    return src


def choose_block(sources: dict[str, set[int]]) -> dict[str, Any]:
    """First BLOCK-wide seed block >= SEED_FLOOR with zero collision (decided before any state collection)."""
    every = set().union(*sources.values())
    tried = []
    start = SEED_FLOOR
    while True:
        block = range(start, start + BLOCK)
        hits = sorted(every.intersection(block))
        tried.append({"block": [start, start + BLOCK - 1], "collisions": hits[:20], "n_collisions": len(hits)})
        if not hits:
            return {"seed_block": [start, start + BLOCK - 1], "tried": tried,
                    "sources_checked": {k: len(v) for k, v in sources.items()}}
        start += BLOCK


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------
def collect(base: SimulationConfig, rule: dict[str, Any], seeds: Iterable[int]) -> dict[str, Any]:
    """Sequential days with frozen RulePolicy; stop when BOTH subsets >= 300 or 100k initial states inspected."""
    wait, cb = [], []
    inspected, days = 0, []
    counts = Counter()
    for seed in seeds:
        day = day_spec(seed)
        cfg = day_config(base, day.family, day.cell)
        traj = trajectory(cfg, seed, rule, threshold_theta=0.4)
        n_init = 0
        for rec in traj:
            if not is_initial(rec):
                continue
            if inspected >= MAX_INITIAL_STATES:
                break
            inspected += 1
            n_init += 1
            for label, fn, bucket in (("WAIT", wait_favourable, wait), ("CALLBACK", callback_favourable, cb)):
                if fn(rec):
                    bucket.append(_record(rec, day, label))
        counts[day.family] += 1
        days.append(day.to_dict() | {"initial_states": n_init})
        if (len(wait) >= TARGET_N and len(cb) >= TARGET_N) or inspected >= MAX_INITIAL_STATES:
            break
    return {"wait": wait, "callback": cb, "initial_states_inspected": inspected, "days": days,
            "families_simulated": dict(counts), "stopped_because": "both subsets reached 300" if
            (len(wait) >= TARGET_N and len(cb) >= TARGET_N) else "100,000 initial states inspected"}


def _record(rec: dict[str, Any], day: DaySpec, label: str) -> dict[str, Any]:
    o = rec["observation"]
    return {"state_id": f"{label}|{day.seed}|{rec['target_index']}", "subset": label, "seed": day.seed,
            "family": day.family, "cell": day.cell.cell_id, "target_index": rec["target_index"],
            "decision_id": rec["decision_id"], "contact_id": rec["contact_id"], "timestamp": rec["timestamp"],
            "observation": o, "mask": rec["mask"], "mask_reasons": rec["mask_reasons"],
            "rule_action": rec["rule_action"], "aifirst_action": rec["aifirst_action"],
            "threshold_action": rec["threshold_action"], "qualified_as": f"{label}-favourable (pre-registered filter)"}


def prevalence(col: dict[str, Any]) -> dict[str, Any]:
    n = col["initial_states_inspected"]
    out = {"initial_states_inspected": n, "days_simulated": len(col["days"]), "families_simulated": col["families_simulated"]}
    for name in ("wait", "callback"):
        rs = col[name]
        out[name] = {"qualifying": len(rs), "prevalence": len(rs) / n if n else 0.0,
                     "unique_days": len({r["seed"] for r in rs}),
                     "families": dict(Counter(r["family"] for r in rs)),
                     "request_types": dict(Counter(r["observation"]["request_type"] for r in rs)),
                     "urgency": dict(Counter(r["observation"]["urgency_pred"] for r in rs)),
                     "human_queue": dict(Counter(r["observation"]["human_queue_size"] for r in rs)),
                     "callbacks_pending": dict(Counter(r["observation"]["callbacks_pending"] for r in rs)),
                     "waiting_time_so_far": _desc([r["observation"]["waiting_time_so_far"] for r in rs]),
                     "ai_score": _desc([r["observation"]["ai_resolution_score"] for r in rs]),
                     "deadline_remaining": _desc([r["observation"]["deadline_remaining_pred"] for r in rs])}
    return out


def _desc(v: Sequence[float]) -> dict[str, float]:
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}


# ---------------------------------------------------------------------------
# replay jobs
# ---------------------------------------------------------------------------
def jobs_for(base: SimulationConfig, states: Sequence[dict[str, Any]], rule: dict[str, Any],
             continuation: tuple[str, dict[str, Any]], k: int = K_FUTURES) -> list[BranchJob]:
    pre, cont = policy_spec("RulePolicy", rule), policy_spec(*continuation)
    configs: dict[tuple[str, str], SimulationConfig] = {}
    cells = {c.cell_id: c for c in MIXTURE_CELLS}
    jobs = []
    for st in states:
        key = (st["family"], st["cell"])
        if key not in configs:
            configs[key] = day_config(base, st["family"], cells[st["cell"]])
        sk = StateKey(st["target_index"], st["decision_id"], st["contact_id"], st["timestamp"], st["observation"],
                      tuple(st["mask"][a.value] for a in ACTION_ORDER))
        for a in ACTION_ORDER:
            if st["mask"][a.value]:
                for variant in range(k):
                    jobs.append(BranchJob(st["state_id"], configs[key], st["seed"], sk, a.value, variant, pre, cont))
    return jobs


# ---------------------------------------------------------------------------
# statistics (Phase C functions reused: analyze_state, paired_interval, cluster_bootstrap_mean)
# ---------------------------------------------------------------------------
def crossfit_advantage(returns: dict[str, Sequence[float]], target: str) -> float:
    """J(target) - J(best competitor): competitor chosen on one half of the futures, evaluated on the other
    half (and swapped) -- same split/swap logic as Phase C cross-fitted regret."""
    acts = [a for a in returns if a != target]
    G = {a: np.asarray(v, dtype=float) for a, v in returns.items()}
    K = len(G[target])
    folds = (np.arange(K // 2), np.arange(K // 2, K))
    vals = []
    for sel, ev in (folds, folds[::-1]):
        comp = max(acts, key=lambda a: (G[a][sel].mean(), -acts.index(a)))
        vals.append(float(G[target][ev].mean() - G[comp][ev].mean()))
    return float(np.mean(vals))


def analyze_subset(states: Sequence[dict[str, Any]], results: Sequence[dict[str, Any]], target: str,
                   key: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    failed = set()
    for r in results:
        if r["status"] != "OK":
            failed.add(r["state_id"])
        else:
            grouped[r["state_id"]][r["action"]][r["variant"]] = r["discounted_return"]
    rows = []
    for st in states:
        sid = st["state_id"]
        if sid in failed or target not in grouped.get(sid, {}):
            continue
        rets = {a: [grouped[sid][a][v] for v in range(K_FUTURES)] for a in grouped[sid]}
        res = analyze_state(rets, {"RulePolicy": st["rule_action"], "AIFirstPolicy": st["aifirst_action"]}, sid)
        pair = {}
        for other in rets:
            if other != target:
                iv = paired_interval(np.subtract(rets[target], rets[other]), sid + other)
                pair[other] = {"mean": iv["mean"], "clearly_positive": iv["t_lo"] > 0, "clearly_negative": iv["t_hi"] < 0}
        rows.append({"state_id": sid, "seed": st["seed"], "family": st["family"], "obs": st["observation"],
                     "returns": rets, "label": res["label"], "J": res["J"], "gap": res["J"][res["best"]] - res["J"][res["second"]],
                     "crossfit_advantage": crossfit_advantage(rets, target), "pairwise": pair,
                     "rule_regret_cf": res["regret"]["RulePolicy"]["cross_fitted"]})
    return summarize_subset(rows, target, key, n_states=len(states), n_failed=len(failed))


def summarize_subset(rows: Sequence[dict[str, Any]], target: str, key: str, n_states: int, n_failed: int = 0) -> dict[str, Any]:
    days = [r["seed"] for r in rows]
    labels = Counter(r["label"] for r in rows)
    n = len(rows)
    clear = [1.0 if r["label"] == f"{target}_BEST" else 0.0 for r in rows]
    adv = [r["crossfit_advantage"] for r in rows]
    out: dict[str, Any] = {
        "target": target, "n_states": n_states, "n_analyzed": n, "n_failed_reconstruction": n_failed,
        "unique_days": len(set(days)),
        "label_shares": {l: labels.get(l, 0) / n for l in [f"{a.value}_BEST" for a in ACTION_ORDER] + ["TIE"]} if n else {},
        "clearly_best_share": float(np.mean(clear)) if n else 0.0,
        "clearly_best_share_ci": cluster_bootstrap_mean(clear, days, key + "clear") if n else {},
        "crossfit_advantage": _desc(adv) | ({"ci": cluster_bootstrap_mean(adv, days, key + "adv")} if n else {}),
        "best_second_gap": _desc([r["gap"] for r in rows]),
        "pairwise": {}, "strata": {},
    }
    for other in {o for r in rows for o in r["pairwise"]}:
        sel = [r for r in rows if other in r["pairwise"]]
        means = [r["pairwise"][other]["mean"] for r in sel]
        out["pairwise"][f"{target}-{other}"] = _desc(means) | {
            "ci": cluster_bootstrap_mean(means, [r["seed"] for r in sel], key + other),
            "clearly_positive_share": float(np.mean([r["pairwise"][other]["clearly_positive"] for r in sel])),
            "clearly_negative_share": float(np.mean([r["pairwise"][other]["clearly_negative"] for r in sel]))}
    strata = {
        "urgency": lambda o: o["urgency_pred"], "human_queue": lambda o: str(o["human_queue_size"]),
        "ai_score": lambda o: "<0.2" if o["ai_resolution_score"] < 0.2 else ("0.2-0.35" if o["ai_resolution_score"] < 0.35 else "0.35-0.5"),
        "deadline": lambda o: "<3h" if o["deadline_remaining_pred"] < 180 else ("3-24h" if o["deadline_remaining_pred"] < 1440 else ">24h"),
        "request_type": lambda o: o["request_type"], "callbacks_pending": lambda o: str(o["callbacks_pending"]),
    }
    for name, fn in strata.items():
        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            groups[fn(r["obs"])].append(r)
        out["strata"][name] = {g: {"n": len(v), "clearly_best": float(np.mean([x["label"] == f"{target}_BEST" for x in v])),
                                   "tie": float(np.mean([x["label"] == "TIE" for x in v])),
                                   "mean_crossfit_adv": float(np.mean([x["crossfit_advantage"] for x in v]))}
                               for g, v in sorted(groups.items())}
    fam: dict[str, list] = defaultdict(list)
    for r in rows:
        fam[r["family"]].append(r)
    out["strata"]["family"] = {g: {"n": len(v), "clearly_best": float(np.mean([x["label"] == f"{target}_BEST" for x in v])),
                                   "mean_crossfit_adv": float(np.mean([x["crossfit_advantage"] for x in v]))}
                               for g, v in sorted(fam.items())}
    out["classification"] = classify(out, n_states)
    return out


def classify(s: dict[str, Any], n_qualifying: int) -> dict[str, Any]:
    """Pre-registered niche labels (sections 12, 21, 22)."""
    if n_qualifying < LOW_SAMPLE:
        return {"label": "TOO RARE TO MATTER AT INITIAL ROUTING", "limited_power": False}
    clear = s["clearly_best_share"]
    adv = s["crossfit_advantage"]
    mean, lo = adv["mean"], adv["ci"]["lo"]
    meaningful = clear >= CLEAR_MEANINGFUL and mean > ADV_MEANINGFUL and lo > 0
    positive_gain = mean > ADV_MEANINGFUL and lo > 0
    if meaningful:
        label = "MEANINGFUL INITIAL-ROUTING NICHE"
    elif clear >= CLEAR_WEAK or positive_gain:  # edge case (gain without share) pre-registered as weak
        label = "RARE / WEAK NICHE"
    else:
        label = "NO USEFUL INITIAL NICHE"
    return {"label": label, "limited_power": n_qualifying < TARGET_N, "clearly_best_share": clear,
            "crossfit_adv_mean": mean, "crossfit_adv_ci_lo": lo}


def final_decision(wait_label: str, cb_label: str) -> dict[str, str]:
    w = wait_label == "MEANINGFUL INITIAL-ROUTING NICHE"
    c = cb_label == "MEANINGFUL INITIAL-ROUTING NICHE"
    if w and c:
        return {"case": "D", "verdict": "INITIAL ROUTING — KEEP AI/HUMAN/WAIT/CALLBACK",
                "statement": "The four-action initial-routing formulation is empirically justified. Future work may investigate "
                             "whether the existing observable state is sufficient for a learned policy to identify these niches reliably."}
    if w:
        return {"case": "B", "verdict": "INITIAL ROUTING — KEEP AI/HUMAN + WAIT ONLY",
                "statement": "Initial routing supports AI/HUMAN/WAIT. CALLBACK should be moved to later V2 decision points."}
    if c:
        return {"case": "C", "verdict": "INITIAL ROUTING — KEEP AI/HUMAN + CALLBACK ONLY",
                "statement": "Initial routing supports AI/HUMAN/CALLBACK. WAIT should be moved to later V2 decision points."}
    return {"case": "A", "verdict": "INITIAL ROUTING — AI/HUMAN ONLY; MOVE WAIT/CALLBACK TO V2",
            "statement": "The initial routing problem is effectively AI/HUMAN. WAIT and CALLBACK should not be retained as "
                         "first-routing actions merely for action diversity. They should instead be reconsidered at later "
                         "decision points in V2."}

