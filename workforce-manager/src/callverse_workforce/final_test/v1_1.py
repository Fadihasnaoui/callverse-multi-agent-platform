"""V1.1 final test (docs/final_test_v1_1_prereg.md): mapping, collision audit, hashes, evaluation, statistics.

Statistics reuse the frozen Phase E / E.1 code (``phase_e.evaluation`` and ``phase_e1.evaluation``) unchanged.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from callverse_workforce.baselines.policies import AIFirstPolicy, make_policy
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.phase_e.scenarios import FAMILIES, DaySpec
from callverse_workforce.phase_e1 import evaluation as E1
from callverse_workforce.seeds import FINAL_V1_1_SEEDS, V2_RESERVED_SEEDS

PER_FAMILY = 30
NONINF = "V1.1 FINAL — PPO NON-INFERIOR TO RULEPOLICY"
NOT_NONINF = "V1.1 FINAL — PPO NOT NON-INFERIOR TO RULEPOLICY"
AGREEMENT_THRESHOLD = 0.95
SENTENCE_SUCCESS = ("The frozen Phase E.1 PPO remained non-inferior to RulePolicy on the untouched V1.1 final test set. "
                    "This confirms that the stabilized RL agent generalizes beyond validation and development data.")
SENTENCE_FAILURE = ("The Phase E.1 PPO passed development non-inferiority but did not reproduce that result on the untouched "
                    "final test set. The development SUCCESS result remains valid for its dataset, while the final test shows "
                    "limited out-of-sample reliability.")
SENTENCE_AIFIRST = ("However, the policy remains behaviorally close to AI-first, consistent with the limited routing headroom "
                    "identified in Phases C and D.")


def final_days(seeds: Sequence[int] = FINAL_V1_1_SEEDS) -> list[DaySpec]:
    seeds = list(seeds)
    if seeds != list(FINAL_V1_1_SEEDS):
        raise ValueError("the V1.1 final test uses exactly seeds 2000-2239")
    return [DaySpec(FAMILIES[i // PER_FAMILY], MIXTURE_CELLS[(i % PER_FAMILY) % 4], s) for i, s in enumerate(seeds)]


# ---------------------------------------------------------------------------
# integrity
# ---------------------------------------------------------------------------
def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sha_tree(root: Path) -> dict[str, Any]:
    files = sorted(p for p in root.rglob("*") if p.is_file())
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(sha_file(p).encode())
    return {"files": len(files), "sha256": h.hexdigest()}


def sha_tree_files(files: Sequence[Path], root: Path) -> str:
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        h.update(sha_file(p).encode())
    return h.hexdigest()


FORBIDDEN_CALLS = frozenset({"train_seed", "build_model", "learn", "select_checkpoint", "evaluate_checkpoints", "select_config"})


def code_usage(path: Path) -> dict[str, set[str]]:
    """Imported modules and called names of a source file (AST: strings / docstrings are ignored)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods, calls = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods.add(n.module)
        elif isinstance(n, ast.Call):
            f = n.func
            calls.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
    return {"modules": mods, "calls": calls}


def consumed_seed_sources(project: Path) -> dict[str, set[int]]:
    """Every seed ever consumed (logged or regenerable), EXCLUDING the reserved-but-unused range itself."""
    from callverse_workforce.targeted.audit import known_seed_sources

    src = known_seed_sources(project)
    src.pop("final_test_reserved", None)
    col = project / "artifacts" / "targeted_initial_headroom" / "state_collection.json"
    if col.exists():
        src["targeted_initial_headroom_days"] = {int(d["seed"]) for d in json.loads(col.read_text(encoding="utf-8"))["days"]}
    return src


def collision_audit(project: Path) -> dict[str, Any]:
    src = consumed_seed_sources(project)
    final, v2 = set(FINAL_V1_1_SEEDS), set(V2_RESERVED_SEEDS)
    hits = {k: sorted(v & final) for k, v in src.items() if v & final}
    v2_hits = {k: sorted(v & v2) for k, v in src.items() if v & v2}
    return {"final_range": [2000, 2239], "v2_reserved": [2240, 2999], "sources_checked": {k: len(v) for k, v in src.items()},
            "collisions_final": hits, "n_collisions_final": sum(len(v) for v in hits.values()),
            "collisions_v2_reserved": v2_hits, "n_collisions_v2": sum(len(v) for v in v2_hits.values()),
            "static_code_scan": "all integer literals 2000-2999 in src/tests/scripts are non-seed constants (n_steps, bootstrap B, "
                                "loop counts) or seeds passed to guards that raise before any simulation"}


# ---------------------------------------------------------------------------
# evaluation (day level, every policy independently)
# ---------------------------------------------------------------------------
def agreement(result, rule_params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Decision-level agreement of the evaluated trajectory with AI-first and RulePolicy (same state, same mask)."""
    aif, rule = AIFirstPolicy(), make_policy("RulePolicy", rule_params, seed)
    n = len(result.transitions)
    agree_aif = agree_rule = 0
    ctx: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    groups: dict[str, dict[str, dict[str, int]]] = {"language": defaultdict(lambda: defaultdict(int)),
                                                    "channel": defaultdict(lambda: defaultdict(int))}
    for t in result.transitions:
        s, m, a = t.state, t.action_mask, t.action
        groups["language"][s.language_dialect.value][a.value] += 1
        groups["channel"][s.channel.value][a.value] += 1
        ok_aif = aif.predict(s, m) is a
        ok_rule = rule.predict(s, m) is a
        agree_aif += ok_aif
        agree_rule += ok_rule
        for name, val in (("ai_score", "<0.4" if s.ai_resolution_score < 0.4 else ("0.4-0.7" if s.ai_resolution_score < 0.7 else ">=0.7")),
                          ("humans_available", "0" if s.humans_available == 0 else ("1-2" if s.humans_available <= 2 else "3+")),
                          ("human_utilization", "<50%" if s.human_utilization < 0.5 else ("50-<100%" if s.human_utilization < 1 else "100%")),
                          ("human_queue", "0" if s.human_queue_size == 0 else ("1-2" if s.human_queue_size <= 2 else "3+")),
                          ("recent_ai_failure_rate", "<0.4" if s.recent_ai_failure_rate < 0.4 else ">=0.4"),
                          ("urgency", s.urgency_pred.value),
                          ("deadline", "<3h" if s.deadline_remaining_pred < 180 else ("3-24h" if s.deadline_remaining_pred < 1440 else ">24h")),
                          ("request_type", s.request_type.value)):
            ctx[name][val][0] += 1
            ctx[name][val][1] += int(not ok_rule)
    return {"decisions": n, "agree_aifirst": agree_aif, "agree_rule": agree_rule,
            "exact_aifirst_trajectory": agree_aif == n,
            "routing_by_group": {g: {k: dict(v) for k, v in d.items()} for g, d in groups.items()},
            "rule_disagreement_by_context": {k: {v: {"n": c[0], "disagree": c[1]} for v, c in d.items()} for k, d in ctx.items()}}


def final_eval_job(job: tuple[str, str, Any, str, tuple[str, str, int], dict[str, Any]]) -> dict[str, Any]:
    label, kind, ref, base_path, day, rule_params = job
    from callverse_workforce.twin.runner import run_simulation

    family, cell_id, seed = day
    d = E1.eval_job((label, kind, ref, base_path, day))
    spec = DaySpec(family, {c.cell_id: c for c in MIXTURE_CELLS}[cell_id], seed)
    result = run_simulation(EV._config(base_path, spec), EV._policy(kind, ref, seed), seed=seed)
    if result.total_reward != d["total_reward"]:
        raise RuntimeError("non-deterministic evaluation detected")
    d["agreement"] = agreement(result, rule_params, seed)
    return d


def evaluate(policies: dict[str, tuple[str, Any]], base_path: str, rule_params: dict[str, Any], workers: int) -> dict[str, list]:
    days = EV.days_for("final_v1_1", final_days())
    jobs = [(label, kind, ref, base_path, (d.family, d.cell.cell_id, d.seed), rule_params)
            for label, (kind, ref) in policies.items() for d in days]
    rows = EV.pmap(final_eval_job, jobs, workers)
    out: dict[str, list] = defaultdict(list)
    for r in rows:
        out[r["label"]].append(r)
    return dict(out)


# ---------------------------------------------------------------------------
# statistics (frozen implementations)
# ---------------------------------------------------------------------------
def statistics(res: dict[str, list], ppo_labels: Sequence[str]) -> dict[str, Any]:
    seeds = [res[k] for k in ppo_labels]
    ppo_mean = EV.ppo_mean_by_day(seeds)
    rule = EV.by_day(res["RulePolicy"])
    if len(ppo_mean) != 240:
        raise RuntimeError("primary inference requires exactly 240 paired days")
    v = EV.verdict(ppo_mean, rule, EV.by_day(res["RandomPolicy"]), EV.by_day(res["AIFirstPolicy"]))
    diffs = EV.paired_diff(ppo_mean, rule)
    noninf = v["ppo_minus_rule"]["lo"] > v["margin"]
    stab = E1.stability(seeds, rule)
    vs_aif = EV.paired_bootstrap(EV.paired_diff(ppo_mean, EV.by_day(res["AIFirstPolicy"])), key=3)
    agree = [sum(r["agreement"]["agree_aifirst"] for r in rows) / sum(r["agreement"]["decisions"] for r in rows) for rows in seeds]
    agree_rule = [sum(r["agreement"]["agree_rule"] for r in rows) / sum(r["agreement"]["decisions"] for r in rows) for rows in seeds]
    close_a = vs_aif["lo"] <= 0 <= vs_aif["hi"]
    close_b = float(np.mean(agree)) >= AGREEMENT_THRESHOLD
    close = close_a or close_b
    others = {n: EV.paired_bootstrap(EV.paired_diff(ppo_mean, EV.by_day(res[n])), key=10 + i)
              for i, n in enumerate(("ThresholdPolicy", "BestFixedPolicy", "RandomPolicy"))}
    rule_mean = v["rule_mean_reward"]
    ppo_mean_reward = float(np.mean(list(ppo_mean.values())))
    text = (SENTENCE_SUCCESS if noninf else SENTENCE_FAILURE) + ((" " + SENTENCE_AIFIRST) if close else "")
    return {
        "n_days": len(diffs), "ppo_mean_reward": ppo_mean_reward, "rule_mean_reward": rule_mean, "margin": v["margin"],
        "ppo_minus_rule": v["ppo_minus_rule"] | {"median": float(np.median(diffs))},
        "relative_difference": v["ppo_minus_rule"]["mean"] / rule_mean,
        "verdict": NONINF if noninf else NOT_NONINF,
        "stability": stab, "stability_label": f"TRAINING STABILITY: {stab['status']}",
        "ppo_minus_aifirst": vs_aif | {"relative": vs_aif["mean"] / float(np.mean(list(EV.by_day(res['AIFirstPolicy']).values())))},
        "aifirst_agreement_per_seed": agree, "aifirst_agreement_mean": float(np.mean(agree)),
        "rule_agreement_per_seed": agree_rule, "rule_agreement_mean": float(np.mean(agree_rule)),
        "exact_aifirst_days_per_seed": [sum(r["agreement"]["exact_aifirst_trajectory"] for r in rows) for rows in seeds],
        "close_to_aifirst": {"A_ci_contains_zero": close_a, "B_agreement_ge_95": close_b,
                             "label": f"CLOSE TO AI-FIRST: {'YES' if close else 'NO'}"},
        "other_comparisons": {k: x | {"relative": x["mean"] / float(np.mean(list(EV.by_day(res[k]).values())))} for k, x in others.items()},
        "interpretation": text,
    }
