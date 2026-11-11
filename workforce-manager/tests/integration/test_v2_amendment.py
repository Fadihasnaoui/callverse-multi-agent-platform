"""Stage 1B amendment: calibration/evaluation separation, f_ref A1, two-pass collection, freeze sequence, git/hash guards.

All pipeline and guard tests run in TEMPORARY synthetic git repositories with synthetic simulation functions. No real
DEV seed is simulated, no real f_ref is computed, and no dev_authorization.json is created in the real repository.
"""

from __future__ import annotations

import ast
import json
import pickle
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.phase_e.scenarios import FAMILIES
from callverse_workforce.seeds import (
    PHASE_F_TEST_SEEDS,
    V2_HEADROOM_CAL_SEEDS,
    V2_HEADROOM_EVAL_SEEDS,
    V2_RESERVED_SEEDS,
    SeedPolicyError,
)
from callverse_workforce.v2_postfail import guards as G
from callverse_workforce.v2_postfail import pipeline as PL
from callverse_workforce.v2_postfail import protocol as P

REAL_AUTH = PROJECT_ROOT / "artifacts" / "v2_headroom" / "dev_authorization.json"
BASE_PATH = str(PROJECT_ROOT / "configs" / "normal_day.yaml")
SETS = {"primary_B1": ("primary", "B1", "persistence", "primary", 4, False), "primary_B2": ("primary", "B2", "persistence", "primary", 4, False),
        "k64_B1": ("primary", "B1", "persistence", "primary", 8, True), "k64_B2": ("primary", "B2", "persistence", "primary", 8, True)}
AGES = {"primary": 8.387374, "sensitivity": 3.0}


# ------------------------------------------------------------------------------------------- seed sets & mapping
def test_seed_sets_registered_and_disjoint():
    assert V2_HEADROOM_EVAL_SEEDS == tuple(range(4000, 4080)) and V2_HEADROOM_CAL_SEEDS == tuple(range(4080, 4090))
    assert P.EVAL_SEEDS == V2_HEADROOM_EVAL_SEEDS and P.CAL_SEEDS == V2_HEADROOM_CAL_SEEDS
    assert not set(P.EVAL_SEEDS) & set(P.CAL_SEEDS)
    protected = set(V2_RESERVED_SEEDS) | set(PHASE_F_TEST_SEEDS)
    assert not (set(P.EVAL_SEEDS) | set(P.CAL_SEEDS)) & protected
    with pytest.raises(SeedPolicyError):
        P.assert_disjoint([1, 2, 3], [3, 4])


def test_calibration_mapping_ten_normal_days():
    days = P.cal_days()
    assert [d.seed for d in days] == list(range(4080, 4090)) and {d.family for d in days} == {"NORMAL"}
    assert [d.cell.cell_id for d in days] == ["h0_m1_p0", "h1_m0.8_p0", "h2_m0.6_p0.05", "h4_m0.4_p0.05"] * 2 + ["h0_m1_p0", "h1_m0.8_p0"]
    assert Counter(d.cell.cell_id for d in days) == {"h0_m1_p0": 3, "h1_m0.8_p0": 3, "h2_m0.6_p0.05": 2, "h4_m0.4_p0.05": 2}
    normal_eval = [d for d in P.eval_days() if d.family == "NORMAL"]
    assert Counter(d.cell.cell_id for d in normal_eval) == Counter(d.cell.cell_id for d in days)


def test_evaluation_design_unchanged():
    days = P.eval_days()
    assert len(days) == 80 and [d.seed for d in days] == list(range(4000, 4080))
    assert Counter(d.family for d in days) == {f: 10 for f in FAMILIES}
    for f in FAMILIES:
        assert Counter(d.cell.cell_id for d in days if d.family == f) == {"h0_m1_p0": 3, "h1_m0.8_p0": 3, "h2_m0.6_p0.05": 2, "h4_m0.4_p0.05": 2}


def test_runtime_seed_role_enforcement():
    cal, ev = P.cal_days()[0], P.eval_days()[0]
    with pytest.raises(SeedPolicyError):
        P.calibration_trace(ev, BASE_PATH, {})  # an evaluation day can never calibrate (raises before any simulation)
    with pytest.raises(SeedPolicyError):
        P.live_b1_day(cal, BASE_PATH, {})  # a calibration day can never contribute evaluation states


def test_f_ref_used_only_in_world_model():
    pkg = PROJECT_ROOT / "src" / "callverse_workforce" / "v2_postfail"
    for name in ("policies.py", "encoding.py", "actions.py", "snapshot.py", "queue.py"):
        names = {n.id for n in ast.walk(ast.parse((pkg / name).read_text(encoding="utf-8"))) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(ast.parse((pkg / name).read_text(encoding="utf-8"))) if isinstance(n, ast.Attribute)}
        assert "f_ref" not in names | attrs, name
    src = (PROJECT_ROOT / "src/callverse_workforce/agent/whatif.py").read_text(encoding="utf-8")
    assert "m = (1.0 - snapshot.ai_failure_rate_30) / max(1e-9, 1.0 - plan.f_ref)" in src


# ------------------------------------------------------------------------------------------- f_ref A1
def test_f_ref_a1_synthetic():
    traces = [[(15.0, 0.9, 0), (30.0, 0.5, 9), (45.0, 0.2, 10), (60.0, 0.4, 12)], [(15.0, 0.3, 30), (30.0, 1.0, 3)]]
    r = P.f_ref_from_traces(traces)
    assert r == {"f_ref": float(np.median([0.2, 0.4, 0.3])), "n_windows": 6, "n_eligible": 3, "n_excluded": 3}
    with pytest.raises(P.FRefError):
        P.f_ref_from_traces([[(15.0, 0.5, 9), (30.0, 0.0, 0)]])  # no eligible window -> fail closed
    with pytest.raises(P.FRefError):
        P.f_ref_from_traces([[(15.0, 1.0, 20)]])  # f_ref = 1 makes the world-model multiplier undefined -> fail closed


# ------------------------------------------------------------------------------------------- two-pass collection
def _fake_live(n_post, diverge=None):
    def fn(day, base_path, rule, capture=None):
        n = n_post + (1 if diverge == "count" and capture is not None else 0)
        contacts = [f"C{i}" for i in range(n)]
        if diverge == "contacts" and capture is not None:
            contacts[0] = "X"
        snaps = []

        class S:
            def __init__(self, cid):
                self.target_contact_id = cid
        for i in sorted(capture or []):
            if diverge == "missing" and i == sorted(capture)[0]:
                continue
            snaps.append({"state_index": i, "contact_id": contacts[i], "snapshot": S(contacts[i] if diverge != "target" else "Z")})
        return {"n_post": n, "snapshots": snaps, "eligible_contacts": contacts}
    return fn


def test_two_pass_consistency():
    day = P.eval_days()[3]
    ok = P.collect_day(day, BASE_PATH, {}, live_fn=_fake_live(57))
    assert ok["n_post"] == 57 and ok["indices"] == P.sample_indices(day.seed, 57) and len(ok["indices"]) == 20
    assert [s["state_index"] for s in ok["snapshots"]] == ok["indices"]
    small = P.collect_day(day, BASE_PATH, {}, live_fn=_fake_live(7))
    assert small["indices"] == list(range(7))  # <= 20 eligible: all of them
    for mode in ("count", "contacts", "missing", "target"):
        with pytest.raises(P.ReplayDivergence):
            P.collect_day(day, BASE_PATH, {}, live_fn=_fake_live(57, mode))
    a, b = P.sample_indices(4003, 57), P.sample_indices(4003, 57)
    assert a == b and len(set(a)) == 20 and all(0 <= i < 57 for i in a)  # deterministic, without replacement


# ------------------------------------------------------------------------------------------- synthetic git repository
def _git(root, *args):
    r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    for d in G.PROTECTED_PATHS:
        (root / d).mkdir(parents=True)
        (root / d / "f.txt").write_text(f"{d} v1\n", encoding="utf-8")
    (root / "docs" / "v2_headroom_spec.md").write_text("# spec v1\n", encoding="utf-8")
    (root / ".gitignore").write_text("*.tmp\nartifacts/v2_headroom/dev/*.pkl\n", encoding="utf-8")
    (root / "artifacts" / "v2_headroom").mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "code")
    code = _git(root, "rev-parse", "HEAD")
    freeze = {"code_commit": code, "spec_sha256": G.sha256_file(root / "docs" / "v2_headroom_spec.md")}
    (root / "artifacts/v2_headroom/protocol_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "freeze")
    return root, code


def _auth(root, code, stages, spec=None):
    a = {"protocol_code_commit": code, "spec_sha256": spec or G.sha256_file(root / "docs" / "v2_headroom_spec.md"), "stages": stages}
    (root / "artifacts/v2_headroom/dev_authorization.json").write_text(json.dumps(a), encoding="utf-8")


def _ctx(root):
    def trace(day, base, rule):
        r = 0.2 + 0.01 * (day.seed - 4080)
        return [(15.0 * (i + 1), r + 0.001 * i, 12 if i % 3 else 4) for i in range(30)]

    def collect(day, base, rule):
        n = 25 + (day.seed % 7)
        idx = P.sample_indices(day.seed, n)
        return {"day": (day.family, day.cell.cell_id, day.seed), "n_post": n, "indices": idx,
                "snapshots": [{"state_index": i, "contact_id": f"C{i}", "snapshot": None, "valid": ["HUMAN_NORMAL", "HUMAN_HIGH"],
                               "b2_action": "HUMAN_NORMAL", "urgency": "LOW"} for i in idx]}

    def returns(snap_rec, day, base, row, rule, k, world, a_age, f_ref):
        rng = np.random.default_rng([day.seed, snap_rec["state_index"], k])
        return {"status": "OK", "state_id": f"{day.seed}:{snap_rec['state_index']}", "day": day.seed, "family": day.family,
                "urgency": "LOW", "baseline_action": "HUMAN_NORMAL", "aging_violations": 0, "aging_overrides": 0,
                "returns": {"HUMAN_NORMAL": rng.normal(0, 1, k).tolist(), "HUMAN_HIGH": rng.normal(0, 1, k).tolist()}}

    return PL.Ctx(root=root, rule_params={}, base_path=BASE_PATH, calibration_trace_fn=trace, collect_day_fn=collect,
                  state_returns_fn=returns)


def _commit(root, msg):
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", msg)


def test_artifact_only_commit_passes_source_guard(repo):
    root, code = repo
    (root / "artifacts/v2_headroom/note.json").write_text("{}", encoding="utf-8")
    _commit(root, "artifact only")
    G.assert_protected_paths_frozen(root, code)  # no exception
    (root / "docs" / "ignored.tmp").write_text("x", encoding="utf-8")  # ignored file inside a protected path
    G.assert_protected_paths_frozen(root, code)


@pytest.mark.parametrize("d", G.PROTECTED_PATHS)
def test_protected_modification_rejected(repo, d):
    root, code = repo
    (root / d / "f.txt").write_text(f"{d} v2\n", encoding="utf-8")
    with pytest.raises(G.GuardError):
        G.assert_protected_paths_frozen(root, code)
    _commit(root, "committed code change")  # committing it does not help: compared against the FROZEN commit
    with pytest.raises(G.GuardError):
        G.assert_protected_paths_frozen(root, code)


def test_staged_and_untracked_rejected(repo):
    root, code = repo
    f = root / "src" / "f.txt"
    f.write_text("src staged\n", encoding="utf-8")
    _git(root, "add", "src/f.txt")
    f.write_text("src v1\n", encoding="utf-8")  # working tree == commit, index differs: only the staged check catches it
    with pytest.raises(G.GuardError, match="staged"):
        G.assert_protected_paths_frozen(root, code)
    _git(root, "reset", "-q", "HEAD", "--", "src/f.txt")
    G.assert_protected_paths_frozen(root, code)
    (root / "scripts" / "new.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(G.GuardError, match="untracked"):
        G.assert_protected_paths_frozen(root, code)


def test_missing_and_mismatched_artifacts_rejected(tmp_path):
    (tmp_path / "a.json").write_text("1", encoding="utf-8")
    good = {"a.json": G.sha256_file(tmp_path / "a.json")}
    G.verify_hashes(tmp_path, good)
    with pytest.raises(G.GuardError, match="missing"):
        G.verify_hashes(tmp_path, {"b.json": "0" * 64})
    (tmp_path / "a.json").write_text("2", encoding="utf-8")
    with pytest.raises(G.GuardError, match="mismatch"):
        G.verify_hashes(tmp_path, good)


def test_spec_hash_and_code_commit_changes_rejected(repo):
    root, code = repo
    c = _ctx(root)
    freeze = json.loads((root / "artifacts/v2_headroom/protocol_freeze.json").read_text(encoding="utf-8"))
    for bad in ({"spec_sha256": "0" * 64}, {"code_commit": "0" * 40}, {"code_commit": code[:12]}):
        (root / "artifacts/v2_headroom/protocol_freeze.json").write_text(json.dumps(freeze | bad), encoding="utf-8")
        with pytest.raises(G.GuardError):
            PL.verify_protocol_freeze(c)
    (root / "artifacts/v2_headroom/protocol_freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    PL.verify_protocol_freeze(c)


def test_calibrate_requires_valid_authorization(repo):
    root, code = repo
    c = _ctx(root)
    with pytest.raises(G.GuardError, match="NOT authorised"):
        PL.stage_calibrate(c)
    _auth(root, "f" * 40, ["calibrate"])
    with pytest.raises(G.GuardError, match="does not match"):
        PL.stage_calibrate(c)
    _auth(root, code, ["collect"])
    with pytest.raises(G.GuardError, match="not authorised"):
        PL.stage_calibrate(c)
    assert not (root / "artifacts/v2_headroom/dev/f_ref.json").exists()


def test_full_synthetic_sequence_and_freeze_records(repo, monkeypatch):
    root, code = repo
    c = _ctx(root)
    _auth(root, code, ["calibrate", "collect", "branches", "analyze"])
    with pytest.raises(G.GuardError):  # collect refuses without a calibration freeze
        PL.stage_collect(c)
    cal = PL.stage_calibrate(c)
    assert cal["calibration_seeds"] == list(range(4080, 4090)) and cal["code_commit"] == code
    assert cal["n_eligible_windows"] + cal["n_excluded_windows"] == cal["n_windows"] == 300
    with pytest.raises(G.GuardError, match="not committed"):  # calibration records must be committed first
        PL.stage_collect(c)
    _commit(root, "calibration records")
    pre = PL.stage_collect(c)
    assert pre["evaluation_seeds"] == list(range(4000, 4080)) and pre["n_states"] == 80 * 20
    assert set(pre["per_day"]) == {str(s) for s in range(4000, 4080)}
    assert all(v["n_eligible"] >= 20 and len(v["selected_indices"]) == 20 for v in pre["per_day"].values())
    assert pre["states_by_family"] == {f: 200 for f in FAMILIES}
    assert pre["n_k64"] == 400 and pre["calibration_freeze_sha256"] == G.sha256_file(root / "artifacts/v2_headroom/dev/calibration_freeze.json")
    with pytest.raises(G.GuardError, match="not committed"):
        PL.stage_branches(c, SETS, AGES)
    _commit(root, "pre-branch records")
    # the K=64 list is LOADED from the frozen file, never recomputed
    monkeypatch.setattr(P, "stability_subsample", lambda ids: (_ for _ in ()).throw(AssertionError("recomputed")))
    man = PL.stage_branches(c, SETS, AGES)
    k64 = set(json.loads((root / "artifacts/v2_headroom/dev/k64_state_ids.json").read_text(encoding="utf-8"))["state_ids"])
    rows = [json.loads(l) for l in (root / "artifacts/v2_headroom/dev/branches/k64_B1.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["state_id"] for r in rows} == k64 and len(man["branch_files"]) == 4
    with pytest.raises(G.GuardError, match="not committed"):
        PL.stage_analyze(c, SETS)
    _commit(root, "branch manifest")
    out = PL.stage_analyze(c, SETS)
    assert set(out["mc_stability"]) == {"B1", "B2"} and "L6" in out
    # tampering with any frozen evaluation artifact -> refusal
    for rel in ("dev/states.pkl", "dev/k64_state_ids.json", "dev/f_ref.json"):
        p = root / "artifacts/v2_headroom" / rel
        orig = p.read_bytes()
        p.write_bytes(orig + b" ")
        with pytest.raises(G.GuardError):
            PL.stage_branches(c, SETS, AGES)
        p.write_bytes(orig)
    bm = root / "artifacts/v2_headroom/dev/branches/primary_B1.jsonl"
    bm.write_text(bm.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(G.GuardError):
        PL.stage_analyze(c, SETS)


def test_k64_mismatch_is_unstable(repo, monkeypatch):
    root, code = repo
    c = _ctx(root)
    _auth(root, code, ["calibrate", "collect", "branches", "analyze"])
    PL.stage_calibrate(c)
    _commit(root, "cal")
    PL.stage_collect(c)
    _commit(root, "pre")
    PL.stage_branches(c, SETS, AGES)
    _commit(root, "manifest")
    p = root / "artifacts/v2_headroom/dev/branches/k64_B2.jsonl"
    rows = p.read_text(encoding="utf-8").splitlines()
    p.write_text("\n".join(rows[1:]) + "\n", encoding="utf-8")  # one prescribed K=64 state missing
    man_p = root / "artifacts/v2_headroom/dev/branches_manifest.json"
    man = json.loads(man_p.read_text(encoding="utf-8"))
    man["branch_files"]["artifacts/v2_headroom/dev/branches/k64_B2.jsonl"] = G.sha256_file(p)
    man_p.write_text(json.dumps(man), encoding="utf-8")
    _commit(root, "manifest (with the incomplete K=64 file)")
    out = PL.stage_analyze(c, SETS)
    assert out["mc_stability"]["B2"]["status"] == "UNSTABLE" and out["gate"]["verdict"].startswith("UNSTABLE")


def test_no_real_dev_authorization_exists():
    assert not REAL_AUTH.exists()


def test_runner_cli_guarded_in_real_repo():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rv2b", PROJECT_ROOT / "scripts" / "run_v2_headroom.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for stage in ("calibrate", "collect", "branches", "analyze"):
        with pytest.raises(SystemExit):
            m.require_authorization(stage)
    assert set(m.SETS) >= {"primary_B1", "primary_B2", "k64_B1", "k64_B2"} and m.SETS["k64_B1"][4] == 64 and m.SETS["primary_B1"][4] == 32
    assert not REAL_AUTH.exists()
