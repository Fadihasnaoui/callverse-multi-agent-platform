"""Executable V2 headroom protocol (docs/v2_headroom_spec.md). Defines constants and harness functions only;
running them on DEV seeds requires a separate explicit authorization (v2_postfail/pipeline.py guards).

Seed roles (amended, disjoint):
* evaluation 4000-4079 (8 families x 10, cells (i % 10) % 4): POST_AI_FAILURE states for H1/H2/H3;
* calibration 4080-4089 (10 NORMAL days, same cell rotation): ONLY the world-model reference f_ref (rule A1).
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np

from callverse_workforce.agent.perception import public_context_from, to_observable
from callverse_workforce.agent.types import DecisionType
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.phase_e.scenarios import FAMILIES, DaySpec
from callverse_workforce.seeds import V2_HEADROOM_CAL_SEEDS, V2_HEADROOM_EVAL_SEEDS, SeedPolicyError, require_seeds
from callverse_workforce.v2_postfail.policies import ContinuationPolicy
from callverse_workforce.v2_postfail.snapshot import take_snapshot
from callverse_workforce.v2_postfail.twin import A_AGE_PRIMARY, A_AGE_SENSITIVITY, V2Environment, V2Twin
from callverse_workforce.v2_postfail.world import branch_seed, run_branch

K_PRIMARY = 32
K_STABILITY = 64
STABILITY_SUBSAMPLE = 0.25
STABILITY_SEED = 20261027
STATES_PER_DAY = 20
STATE_SAMPLE_SEED = 20261028
EVAL_SEEDS = V2_HEADROOM_EVAL_SEEDS
CAL_SEEDS = V2_HEADROOM_CAL_SEEDS
DEV_SEEDS = EVAL_SEEDS  # evaluation days (kept name used by the Stage-1 tests)
PER_FAMILY = 10
CAL_FAMILY = "NORMAL"
GAMMA_PER_MINUTE = 0.995
F_REF_MIN_AI = 10
F_REF_WINDOW = 30.0
F_REF_STEP = 15.0
ROWS = {  # spec §7.1a: (setting, comparison) -> (INITIAL policy, POST policy)
    ("primary", "B1"): ("RulePolicy", "B1"), ("primary", "B2"): ("RulePolicy", "B2"),
    ("robustness", "B1"): ("AIFirstPolicy", "B1"), ("robustness", "B2"): ("AIFirstPolicy", "B2"),
}


class ReplayDivergence(RuntimeError):
    """Pass 2 of state collection did not reproduce pass 1 (fail closed)."""


class FRefError(RuntimeError):
    """f_ref cannot be computed by the frozen A1 procedure (fail closed)."""


def assert_disjoint(eval_seeds: Sequence[int] = EVAL_SEEDS, cal_seeds: Sequence[int] = CAL_SEEDS) -> None:
    overlap = sorted(set(eval_seeds) & set(cal_seeds))
    if overlap:
        raise SeedPolicyError(f"calibration / evaluation seed overlap: {overlap[:5]}")


def eval_days() -> list[DaySpec]:
    """Evaluation days: positional mapping over EXACTLY the 80 evaluation seeds."""
    require_seeds(EVAL_SEEDS, V2_HEADROOM_EVAL_SEEDS, "V2 headroom evaluation")
    assert_disjoint()
    return [DaySpec(FAMILIES[i // PER_FAMILY], MIXTURE_CELLS[(i % PER_FAMILY) % 4], s) for i, s in enumerate(EVAL_SEEDS)]


dev_days = eval_days


def cal_days() -> list[DaySpec]:
    """Calibration days: an explicit mapping of its own (never the evaluation index); 10 NORMAL days, cells h0 h1 h2 h4
    h0 h1 h2 h4 h0 h1 (h0 x3, h1 x3, h2 x2, h4 x2) = the cell pattern of the NORMAL evaluation days."""
    require_seeds(CAL_SEEDS, V2_HEADROOM_CAL_SEEDS, "V2 headroom calibration")
    assert_disjoint()
    return [DaySpec(CAL_FAMILY, MIXTURE_CELLS[(i % PER_FAMILY) % 4], s) for i, s in enumerate(CAL_SEEDS)]


def _window_rate(env, t: float) -> tuple[float, int]:
    """A1 window at instant t: observable events in [t - 30, t]; n = AI_APPARENT_RESOLUTION + AI_VISIBLE_FAILURE;
    numerator = AI_VISIBLE_FAILURE + FALSE_AI_DISCOVERED; rate = min(1, num / n) (0 if n = 0, excluded later)."""
    ev = [e for e in (to_observable(r) for r in env.twin.events.records) if e is not None and e.timestamp >= t - F_REF_WINDOW]
    done = sum(1 for e in ev if e.event_type in ("AI_APPARENT_RESOLUTION", "AI_VISIBLE_FAILURE"))
    fail = sum(1 for e in ev if e.event_type in ("AI_VISIBLE_FAILURE", "FALSE_AI_DISCOVERED"))
    return (min(1.0, fail / done) if done else 0.0), done


def _b1_env(day: DaySpec, base_path: str, rule_params: dict[str, Any], a_age: float):
    cfg = EV._config(base_path, day)
    return cfg, V2Environment(V2Twin(cfg, day.seed, a_age=a_age)), ContinuationPolicy("RulePolicy", "B1", rule_params)


def calibration_trace(day: DaySpec, base_path: str, rule_params: dict[str, Any], a_age: float = A_AGE_PRIMARY) -> list[tuple[float, float, int]]:
    """Calibration-only live run (INITIAL RulePolicy, POST B1): A1 window rates at the first decision at or after each
    15-minute boundary. Collects NO evaluation snapshot. Only calibration seeds are accepted."""
    require_seeds([day.seed], CAL_SEEDS, "V2 f_ref calibration")
    _, env, pol = _b1_env(day, base_path, rule_params, a_age)
    trace, next_t = [], F_REF_STEP
    dp = env.next_decision()
    while dp is not None:
        if dp.timestamp >= next_t:
            rate, n = _window_rate(env, dp.timestamp)
            trace.append((dp.timestamp, rate, n))
            while next_t <= dp.timestamp:
                next_t += F_REF_STEP
        dp = env.step(pol.act(dp))
    return trace


def live_b1_day(day: DaySpec, base_path: str, rule_params: dict[str, Any], a_age: float = A_AGE_PRIMARY,
                capture: set[int] | None = None) -> dict[str, Any]:
    """Evaluation live run under B1 (INITIAL Rule, POST HUMAN_NORMAL): count eligible POST states (first POST decision,
    wait_count == 0) and capture observable snapshots of the selected indices. Computes NO f_ref."""
    require_seeds([day.seed], EVAL_SEEDS, "V2 evaluation state collection")
    cfg, env, pol = _b1_env(day, base_path, rule_params, a_age)
    public = public_context_from(cfg, EV.load_config(base_path))
    dp = env.next_decision()
    n_post, snaps, eligible_contacts = 0, [], []
    while dp is not None:
        if dp.decision_type is DecisionType.POST_AI_FAILURE and dp.observation.wait_count == 0:
            eligible_contacts.append(dp.contact_id)
            if capture is not None and n_post in capture:
                b2_action = ContinuationPolicy("RulePolicy", "B2", rule_params).act(dp)
                snaps.append({"state_index": n_post, "contact_id": dp.contact_id, "timestamp": dp.timestamp,
                              "snapshot": take_snapshot(env.twin, dp.contact_id, public), "valid": list(dp.mask.valid_actions),
                              "b2_action": b2_action, "urgency": dp.observation.base.urgency_pred.value})
            n_post += 1
        dp = env.step(pol.act(dp))
    return {"n_post": n_post, "snapshots": snaps, "eligible_contacts": eligible_contacts}


def sample_indices(day_seed: int, n_post: int) -> list[int]:
    """Outcome-blind fixed-seed sample of STATES_PER_DAY eligible POST states (all if n_post <= 20), uniform, without
    replacement."""
    if n_post <= STATES_PER_DAY:
        return list(range(n_post))
    rng = np.random.default_rng(np.random.SeedSequence(STATE_SAMPLE_SEED, spawn_key=(day_seed,)))
    return sorted(rng.choice(n_post, STATES_PER_DAY, replace=False).tolist())


def collect_day(day: DaySpec, base_path: str, rule_params: dict[str, Any],
                live_fn: Callable[..., dict[str, Any]] = live_b1_day) -> dict[str, Any]:
    """Two-pass collection with an exact replay check: pass 1 counts eligible states and fixes the indices; pass 2
    replays the day and captures them; pass 2 must reproduce pass 1 (count, eligible contacts, indices) exactly."""
    first = live_fn(day, base_path, rule_params)
    idx = sample_indices(day.seed, first["n_post"])
    second = live_fn(day, base_path, rule_params, capture=set(idx))
    got = [s["state_index"] for s in second["snapshots"]]
    if second["n_post"] != first["n_post"]:
        raise ReplayDivergence(f"day {day.seed}: eligible count {first['n_post']} vs replay {second['n_post']}")
    if second.get("eligible_contacts") != first.get("eligible_contacts"):
        raise ReplayDivergence(f"day {day.seed}: eligible contact sequence differs on replay")
    if got != idx:
        raise ReplayDivergence(f"day {day.seed}: captured indices {got} != selected {idx}")
    for s in second["snapshots"]:
        if s["snapshot"].target_contact_id != s["contact_id"] or first["eligible_contacts"][s["state_index"]] != s["contact_id"]:
            raise ReplayDivergence(f"day {day.seed}: snapshot {s['state_index']} inconsistent with pass 1")
    return {"day": (day.family, day.cell.cell_id, day.seed), "n_post": first["n_post"], "indices": idx,
            "snapshots": second["snapshots"]}


def f_ref_from_traces(cal_traces: Sequence[Sequence[tuple[float, float, int]]]) -> dict[str, Any]:
    """A1: median of the window rates with n >= 10 completed AI attempts, pooled over all calibration traces; windows
    with n < 10 (incl. 0) are excluded; fail closed if no eligible window remains."""
    windows = [w for tr in cal_traces for w in tr]
    vals = [r for (_, r, n) in windows if n >= F_REF_MIN_AI]
    if not vals:
        raise FRefError("no eligible A1 window (n >= 10): f_ref cannot be computed")
    f = float(np.median(vals))
    if not (0.0 <= f < 1.0):
        raise FRefError(f"f_ref {f} outside [0, 1): the world-model multiplier (1 - f_obs) / (1 - f_ref) is undefined")
    return {"f_ref": f, "n_windows": len(windows), "n_eligible": len(vals), "n_excluded": len(windows) - len(vals)}


def stability_subsample(state_ids: Sequence[str]) -> list[str]:
    """K = 64 subsample: drawn ONCE from the sorted COLLECTED state ids (25 %, seed 20261027, without replacement),
    saved to dev/k64_state_ids.json before any branch; later stages load the file, never recompute."""
    ids = sorted(state_ids)
    rng = np.random.default_rng(STABILITY_SEED)
    k = max(1, int(round(STABILITY_SUBSAMPLE * len(ids))))
    return sorted(rng.choice(ids, k, replace=False).tolist())


def state_returns(snap_rec: dict[str, Any], day: DaySpec, base, row: tuple[str, str], rule_params, k: int, world_model: str,
                  a_age: float, f_ref: float) -> dict[str, Any]:
    """All valid actions x K futures for one state under ONE continuation row (identical continuation and futures).
    ``f_ref`` enters only the hypothetical-world AI multiplier (agent.whatif.ai_multiplier); it is not a B2 threshold
    and not a decision-state feature."""
    cont = ContinuationPolicy(*row, rule_params)
    cache: dict = {}
    snap = snap_rec["snapshot"]
    state_id = f"{day.seed}:{snap_rec['state_index']}"
    returns, extra = {}, {"aging_violations": 0, "aging_overrides": 0, "classes": []}
    for a in snap_rec["valid"]:
        rs = [run_branch(snap, base, a, branch_seed(day.seed, snap_rec["state_index"], j), cont, world_model, a_age, f_ref, cache)
              for j in range(k)]
        if any(r["status"] != "OK" for r in rs):
            return {"status": "BRANCH_FAILED", "state_id": state_id, "action": a}
        returns[a] = [r["discounted_return"] for r in rs]
        extra["aging_violations"] += sum(r["aging_violations"] for r in rs)
        extra["aging_overrides"] += sum(r["aging_overrides"] for r in rs)
        extra["classes"].append({a: [r["classes"] for r in rs]})
    baseline_action = "HUMAN_NORMAL" if row[1] == "B1" else snap_rec["b2_action"]
    return {"status": "OK", "state_id": state_id, "day": day.seed, "family": day.family,
            "urgency": snap_rec["urgency"], "baseline_action": baseline_action, "returns": returns,
            "continuation": list(cont.spec), "k": k, "world_model": world_model, "a_age": a_age} | extra
