"""V2 headroom stage pipeline with linked freeze records (spec §11a). NOTHING here runs without authorization.

Strict sequence:
  0. protocol freeze (code commit + spec hash; artifacts/v2_headroom/protocol_freeze.json, committed)
  1. calibrate (seeds 4080-4089 only)  -> dev/f_ref.json, dev/calibration_traces.json, dev/calibration_freeze.json
     -> the operator commits dev/calibration_freeze.json, dev/f_ref.json, dev/calibration_traces.json
  2. collect (seeds 4000-4079 only, two-pass replay check) -> dev/states.pkl, dev/k64_state_ids.json,
     dev/pre_branch_freeze.json -> the operator commits dev/pre_branch_freeze.json and dev/k64_state_ids.json
  3. branches (separate later authorization) -> dev/branches/*.jsonl, dev/branches_manifest.json
     -> the operator commits dev/branches_manifest.json (branch files are protected by its hashes)
  4. analyze -> dev/headroom_results.json
Every stage verifies the protocol freeze (spec hash, code commit, protected paths), the stage authorization in
dev_authorization.json, and every upstream freeze record / artifact hash; any mismatch or missing file refuses.
"""

from __future__ import annotations

import datetime as dt
import json
import pickle
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from callverse_workforce.v2_postfail import guards as G
from callverse_workforce.v2_postfail import protocol as P

STAGES = ("calibrate", "collect", "branches", "analyze")
F_REF_PROCEDURE = ("A1 (spec §7.0a): INITIAL RulePolicy, POST B1; calibration seeds 4080-4089 only; first decision at or "
                   "after each 15-min boundary; window [t-30, t]; n = AI_APPARENT_RESOLUTION + AI_VISIBLE_FAILURE; numerator = "
                   "AI_VISIBLE_FAILURE + FALSE_AI_DISCOVERED; windows with n < 10 (incl. 0) excluded; median of retained rates; "
                   "fail closed if none")


@dataclass
class Ctx:
    root: Path
    rule_params: dict[str, Any]
    base_path: str
    out_rel: str = "artifacts/v2_headroom"
    spec_rel: str = "docs/v2_headroom_spec.md"
    calibration_trace_fn: Callable[..., Any] = P.calibration_trace
    collect_day_fn: Callable[..., Any] = P.collect_day
    state_returns_fn: Callable[..., Any] = P.state_returns
    pmap: Callable[..., list] = field(default=None)  # type: ignore[assignment]

    def rel(self, *parts: str) -> str:
        return "/".join((self.out_rel,) + parts)

    def path(self, *parts: str) -> Path:
        return Path(self.root) / self.rel(*parts)

    def map(self, fn, jobs, workers: int) -> list:
        if self.pmap is not None:
            return self.pmap(_star, [(fn, j) for j in jobs], workers)
        return [fn(*j) for j in jobs]


def _star(args):
    """Picklable (fn, args) unpacker for process pools."""
    fn, a = args
    return fn(*a)


def _read(p: Path) -> dict:
    if not p.exists():
        raise G.GuardError(f"required record missing: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _write(p: Path, data: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=1, sort_keys=True, default=str), encoding="utf-8")


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


# ------------------------------------------------------------------------------------------------ guards
def verify_protocol_freeze(ctx: Ctx) -> dict:
    freeze = _read(ctx.path("protocol_freeze.json"))
    commit = freeze.get("code_commit") or freeze.get("git_commit")
    if G.sha256_file(Path(ctx.root) / ctx.spec_rel) != freeze.get("spec_sha256"):
        raise G.GuardError("specification changed after the protocol freeze")
    G.assert_protected_paths_frozen(ctx.root, commit)
    return freeze | {"code_commit": commit}


def require_authorization(ctx: Ctx, stage: str) -> dict:
    """Valid dev_authorization.json = names this protocol's code commit AND spec hash AND the stage."""
    if stage not in STAGES:
        raise G.GuardError(f"unknown stage {stage}")
    freeze = verify_protocol_freeze(ctx)
    auth_p = ctx.path("dev_authorization.json")
    if not auth_p.exists():
        raise G.GuardError("DEV run NOT authorised (artifacts/v2_headroom/dev_authorization.json missing)")
    auth = json.loads(auth_p.read_text(encoding="utf-8"))
    if auth.get("protocol_code_commit") != freeze["code_commit"] or auth.get("spec_sha256") != freeze["spec_sha256"]:
        raise G.GuardError("dev_authorization.json does not match the frozen code commit / specification")
    if stage not in auth.get("stages", []):
        raise G.GuardError(f"stage '{stage}' is not authorised")
    P.assert_disjoint()
    return freeze


def verify_calibration(ctx: Ctx, freeze: dict) -> dict:
    rel = ctx.rel("dev", "calibration_freeze.json")
    cal = _read(Path(ctx.root) / rel)
    G.assert_committed_unchanged(ctx.root, [rel])
    if cal.get("code_commit") != freeze["code_commit"] or cal.get("spec_sha256") != freeze["spec_sha256"]:
        raise G.GuardError("calibration freeze does not match the protocol freeze")
    if sorted(cal.get("calibration_seeds", [])) != list(P.CAL_SEEDS):
        raise G.GuardError("calibration freeze seeds differ from the frozen calibration set")
    G.verify_hashes(ctx.root, cal["artifacts"])
    f = json.loads((Path(ctx.root) / ctx.rel("dev", "f_ref.json")).read_text(encoding="utf-8"))["f_ref"]
    if f != cal["f_ref"]:
        raise G.GuardError("f_ref.json value differs from the calibration freeze")
    return cal


def verify_pre_branch(ctx: Ctx, freeze: dict, cal: dict) -> tuple[dict, dict, list[str]]:
    rel = ctx.rel("dev", "pre_branch_freeze.json")
    pre = _read(Path(ctx.root) / rel)
    G.assert_committed_unchanged(ctx.root, [rel, ctx.rel("dev", "k64_state_ids.json")])
    if pre.get("code_commit") != freeze["code_commit"] or pre.get("spec_sha256") != freeze["spec_sha256"]:
        raise G.GuardError("pre-branch freeze does not match the protocol freeze")
    G.verify_hashes(ctx.root, {ctx.rel("dev", "calibration_freeze.json"): pre["calibration_freeze_sha256"],
                               ctx.rel("dev", "f_ref.json"): pre["f_ref_json_sha256"],
                               ctx.rel("dev", "states.pkl"): pre["states_pkl_sha256"],
                               ctx.rel("dev", "k64_state_ids.json"): pre["k64_state_ids_sha256"]})
    if sorted(pre.get("evaluation_seeds", [])) != list(P.EVAL_SEEDS):
        raise G.GuardError("pre-branch freeze evaluation seeds differ from the frozen evaluation set")
    with open(ctx.path("dev", "states.pkl"), "rb") as fh:
        states = pickle.load(fh)
    k64 = json.loads(ctx.path("dev", "k64_state_ids.json").read_text(encoding="utf-8"))["state_ids"]
    collected = {f"{seed}:{s['state_index']}" for seed, v in states.items() for s in v["snapshots"]}
    if not set(k64) <= collected or len(collected) != pre["n_states"]:
        raise G.GuardError("K=64 state list or collected-state count inconsistent with the frozen states")
    return pre, states, k64


# ------------------------------------------------------------------------------------------------ stages
def stage_calibrate(ctx: Ctx, workers: int = 1) -> dict:
    freeze = require_authorization(ctx, "calibrate")
    if ctx.path("dev", "calibration_freeze.json").exists():
        raise G.GuardError("calibration already frozen")
    days = P.cal_days()
    traces = ctx.map(ctx.calibration_trace_fn, [(d, ctx.base_path, ctx.rule_params) for d in days], workers)
    res = P.f_ref_from_traces(traces)  # FRefError: fail closed
    _write(ctx.path("dev", "calibration_traces.json"),
           {str(d.seed): {"family": d.family, "cell": d.cell.cell_id, "windows": tr} for d, tr in zip(days, traces)})
    _write(ctx.path("dev", "f_ref.json"), {"f_ref": res["f_ref"], "procedure": F_REF_PROCEDURE,
                                           "used_by": "hypothetical-world AI multiplier only (agent.whatif.ai_multiplier); "
                                                      "not a B2 threshold, not a decision-state feature",
                                           "single_value_for": ["primary_B1", "primary_B2", "robustness_B1", "robustness_B2",
                                                                "all sensitivities"]})
    cal = {"code_commit": freeze["code_commit"], "spec_sha256": freeze["spec_sha256"], "frozen_at_utc": _now(),
           "calibration_seeds": [d.seed for d in days], "mapping": [d.to_dict() for d in days], "procedure": F_REF_PROCEDURE,
           "f_ref": res["f_ref"], "n_windows": res["n_windows"], "n_eligible_windows": res["n_eligible"],
           "n_excluded_windows": res["n_excluded"],
           "provenance": {str(d.seed): {"windows": len(tr), "eligible": sum(1 for w in tr if w[2] >= P.F_REF_MIN_AI)}
                          for d, tr in zip(days, traces)},
           "artifacts": {ctx.rel("dev", "f_ref.json"): G.sha256_file(ctx.path("dev", "f_ref.json")),
                         ctx.rel("dev", "calibration_traces.json"): G.sha256_file(ctx.path("dev", "calibration_traces.json"))},
           "next_step": "commit dev/calibration_freeze.json, dev/f_ref.json, dev/calibration_traces.json before 'collect'"}
    _write(ctx.path("dev", "calibration_freeze.json"), cal)
    return cal


def stage_collect(ctx: Ctx, workers: int = 1) -> dict:
    freeze = require_authorization(ctx, "collect")
    cal = verify_calibration(ctx, freeze)
    if ctx.path("dev", "pre_branch_freeze.json").exists() or ctx.path("dev", "states.pkl").exists():
        raise G.GuardError("evaluation states already collected")
    days = P.eval_days()
    res = ctx.map(ctx.collect_day_fn, [(d, ctx.base_path, ctx.rule_params) for d in days], workers)  # ReplayDivergence: fail closed
    states = {r["day"][2]: r for r in res}
    with open(ctx.path("dev", "states.pkl"), "wb") as fh:
        pickle.dump(states, fh)
    ids = sorted(f"{seed}:{s['state_index']}" for seed, v in states.items() for s in v["snapshots"])
    k64 = P.stability_subsample(ids)
    _write(ctx.path("dev", "k64_state_ids.json"), {"state_ids": k64, "rule": "25 % of the sorted COLLECTED state ids, seed 20261027, "
                                                   "without replacement; drawn once at collection, never recomputed"})
    fam = Counter(states[s]["day"][0] for s in states for _ in states[s]["snapshots"])
    cell = Counter(f"{states[s]['day'][0]}|{states[s]['day'][1]}" for s in states for _ in states[s]["snapshots"])
    pre = {"code_commit": freeze["code_commit"], "spec_sha256": freeze["spec_sha256"], "frozen_at_utc": _now(),
           "calibration_freeze_sha256": G.sha256_file(ctx.path("dev", "calibration_freeze.json")),
           "f_ref_json_sha256": G.sha256_file(ctx.path("dev", "f_ref.json")),
           "states_pkl_sha256": G.sha256_file(ctx.path("dev", "states.pkl")),
           "k64_state_ids_sha256": G.sha256_file(ctx.path("dev", "k64_state_ids.json")),
           "k64_state_ids_file": ctx.rel("dev", "k64_state_ids.json"), "n_k64": len(k64),
           "n_states": len(ids), "evaluation_seeds": [d.seed for d in days],
           "per_day": {str(s): {"day": list(v["day"]), "n_eligible": v["n_post"], "selected_indices": v["indices"]}
                       for s, v in sorted(states.items())},
           "states_by_family": dict(fam), "states_by_family_cell": dict(cell),
           "next_step": "commit dev/pre_branch_freeze.json and dev/k64_state_ids.json; branches require a later explicit authorization"}
    _write(ctx.path("dev", "pre_branch_freeze.json"), pre)
    return pre


def stage_branches(ctx: Ctx, sets: dict, a_ages: dict, workers: int = 1) -> dict:
    """sets: name -> (setting, comparison, world_model, age_key, k, subsample_only). The K = 64 list is LOADED."""
    freeze = require_authorization(ctx, "branches")
    cal = verify_calibration(ctx, freeze)
    pre, states, k64 = verify_pre_branch(ctx, freeze, cal)
    f_ref = cal["f_ref"]
    sub = set(k64)
    from callverse_workforce.config import load_config
    from callverse_workforce.phase_d.grid import MIXTURE_CELLS
    from callverse_workforce.phase_e.scenarios import DaySpec

    base = load_config(ctx.base_path)
    cells = {c.cell_id: c for c in MIXTURE_CELLS}
    shas = {}
    for name, (setting, comp, world, age, k, sub_only) in sets.items():
        rel = ctx.rel("dev", "branches", f"{name}.jsonl")
        jobs = [(s, DaySpec(v["day"][0], cells[v["day"][1]], v["day"][2]), base, P.ROWS[(setting, comp)], ctx.rule_params, k, world,
                 a_ages[age], f_ref)
                for seed, v in sorted(states.items()) for s in v["snapshots"] if not sub_only or f"{seed}:{s['state_index']}" in sub]
        rows = ctx.map(ctx.state_returns_fn, jobs, workers)
        p = Path(ctx.root) / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({k_: v_ for k_, v_ in r.items() if k_ != "classes"}, sort_keys=True) + "\n")
        shas[rel] = G.sha256_file(p)
    manifest = {"pre_branch_freeze_sha256": G.sha256_file(ctx.path("dev", "pre_branch_freeze.json")), "branch_files": shas,
                "f_ref": f_ref, "created_at_utc": _now()}
    _write(ctx.path("dev", "branches_manifest.json"), manifest)
    return manifest


def stage_analyze(ctx: Ctx, sets: dict) -> dict:
    from callverse_workforce.v2_postfail import stats as S

    freeze = require_authorization(ctx, "analyze")
    cal = verify_calibration(ctx, freeze)
    pre, states, k64 = verify_pre_branch(ctx, freeze, cal)
    man = _read(ctx.path("dev", "branches_manifest.json"))
    G.assert_committed_unchanged(ctx.root, [ctx.rel("dev", "branches_manifest.json")])
    if man["pre_branch_freeze_sha256"] != G.sha256_file(ctx.path("dev", "pre_branch_freeze.json")):
        raise G.GuardError("branch manifest does not match the pre-branch freeze")
    G.verify_hashes(ctx.root, man["branch_files"])
    load = lambda n: [json.loads(l) for l in ctx.path("dev", "branches", f"{n}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    res = {}
    for name in sets:
        rows = load(name)
        ok = [r for r in rows if r["status"] == "OK"]
        res[name] = S.analyze(ok, name) | {"aging_violations": sum(r["aging_violations"] for r in ok), "n_failed": len(rows) - len(ok)}
    sub = set(k64)
    mc = {}
    for c in ("B1", "B2"):
        k64_rows = res[f"k64_{c}"]["rows"]
        if {r["state_id"] for r in k64_rows} != sub:  # the prescribed subset could not be verified -> UNSTABLE
            mc[c] = {"status": "UNSTABLE", "mc_unstable": True, "reason": "K=64 rows do not match the frozen state list"}
        else:
            mc[c] = S.mc_stability(res[f"primary_{c}"]["rows"], k64_rows, f"mc_{c}")
    gate = S.decision_gate(res["primary_B1"]["classification"], res["primary_B2"]["classification"], mc["B1"], mc["B2"],
                           sum(v["aging_violations"] for v in res.values()))
    out = {"sets": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in res.items()}, "mc_stability": mc, "gate": gate,
           "L6": "The headroom results apply to POST_AI_FAILURE states generated by the live B1 policy. Results must not be "
                 "presented as an exhaustive measure of B2 on its own state distribution."}
    _write(ctx.path("dev", "headroom_results.json"), out)
    return out
