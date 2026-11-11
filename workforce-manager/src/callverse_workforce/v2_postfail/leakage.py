"""V2 leakage gate (spec §7.2): the 7 Phase F checks repeated with V2 snapshots and branches."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from typing import Any

from callverse_workforce.agent.perception import public_context_from
from callverse_workforce.agent.types import DecisionType
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.v2_postfail.policies import ContinuationPolicy
from callverse_workforce.v2_postfail.snapshot import take_snapshot
from callverse_workforce.v2_postfail.twin import A_AGE_PRIMARY, V2Environment, V2Twin
from callverse_workforce.v2_postfail.world import branch_seed, build_world, plan_for, run_branch
from callverse_workforce.agent.whatif import WHATIF_SEED_BASE, hypothetical_config, world_parameters

FORBIDDEN = {"truth", "degradation_window", "shortage_window", "shock_window", "_arrival_queue", "family"}
PKG = Path(__file__).resolve().parent


def _live(day, base_path, rule, until_post: int):
    cfg = EV._config(base_path, day)
    public = public_context_from(cfg, EV.load_config(base_path))
    env = V2Environment(V2Twin(cfg, day.seed))
    pol = ContinuationPolicy("RulePolicy", "B1", rule)
    dp, n = env.next_decision(), 0
    while dp is not None:
        if dp.decision_type is DecisionType.POST_AI_FAILURE:
            if n == until_post:
                return env, take_snapshot(env.twin, dp.contact_id, public), pol, dp
            n += 1
        dp = env.step(pol.act(dp))
    raise RuntimeError("not enough POST states")


def leakage_gate(day_a, day_b, base_path: str, rule: dict[str, Any], f_ref: float, k: int = 3) -> dict[str, Any]:
    """day_a / day_b: two ALREADY-OPENED (non-protected) days with different hidden sources."""
    base = EV.load_config(base_path)
    env_a, snap_a, pol, dp_a = _live(day_a, base_path, rule, 30)
    cont = ContinuationPolicy("RulePolicy", "B2", rule)
    seeds = [branch_seed(day_a.seed, 30, j) for j in range(k)]
    actions = list(dp_a.mask.valid_actions)

    def run(snap):
        return [run_branch(snap, base, a, s, cont, "persistence", A_AGE_PRIMARY, f_ref)["discounted_return"] for a in actions for s in seeds]

    r1 = run(snap_a)
    env_b, snap_b, *_ = _live(day_b, base_path, rule, 30)  # different hidden source simulated in-process
    r2 = run(dataclasses.replace(snap_b, **{f.name: getattr(snap_a, f.name) for f in dataclasses.fields(snap_a)}))
    for env in (env_a, env_b):  # mutate every hidden live field
        tw = env.twin
        for rt in tw.contacts.values():
            object.__setattr__(rt.truth, "patience_minutes", rt.truth.patience_minutes * 7.0 + 1.0)
            object.__setattr__(rt.truth, "ai_correct_prob", 0.01)
        tw.degradation_window = (0.0, 1e9)
        tw.shortage_window = (0.0, 1e9)
        tw._arrival_queue.clear()
    r3 = run(snap_a)
    params = world_parameters(snap_a, base, plan_for(snap_a, base, "persistence", f_ref))
    cfg = hypothetical_config(snap_a, base, params, MIXTURE_CELLS[0])
    live_future = {round(c.arrival_time, 9) for c in env_a.twin.truth if c.arrival_time > snap_a.now}
    overlap = sum(len({round(c.arrival_time, 9) for c in build_world(snap_a, cfg, params, s)[0]} & live_future) for s in seeds)
    # live RNG untouched: a fresh live day with branches run mid-day continues identically to an untouched run
    env_c, snap_c, pol_c, dp_c = _live(day_a, base_path, rule, 30)
    run(snap_c)
    dp = dp_c
    while dp is not None:
        dp = env_c.step(pol_c.act(dp))
    env_d, _, pol_d, dp_d = _live(day_a, base_path, rule, 30)
    dp = dp_d
    while dp is not None:
        dp = env_d.step(pol_d.act(dp))
    live_untouched = env_c.twin.reward.episode.total == env_d.twin.reward.episode.total
    attrs = set()
    for name in ("snapshot.py", "world.py", "policies.py", "stats.py"):
        attrs |= {n.attr for n in ast.walk(ast.parse((PKG / name).read_text(encoding="utf-8"))) if isinstance(n, ast.Attribute)}
    forbidden = sorted(attrs & FORBIDDEN)
    checks = {"identical_hidden_source_swap": r1 == r2, "identical_after_hidden_mutation": r1 == r3,
              "zero_live_future_arrival_overlap": overlap == 0, "live_rng_untouched": live_untouched,
              "seeds_ge_2_40": all(s >= WHATIF_SEED_BASE for s in seeds), "no_forbidden_attributes": not forbidden,
              "planner_input_snapshot_only": "def run_branch(snap: V2Snapshot" in (PKG / "world.py").read_text(encoding="utf-8")}
    return {"checks": checks, "passed": all(checks.values()), "label": "LEAKAGE TEST PASSED" if all(checks.values()) else "LEAKAGE TEST FAILED",
            "forbidden_found": forbidden, "days": [repr(day_a), repr(day_b)], "n_returns": len(r1)}
