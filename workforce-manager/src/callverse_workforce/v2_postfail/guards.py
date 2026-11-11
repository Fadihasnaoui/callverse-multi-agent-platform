"""Runtime integrity guards for the V2 headroom protocol (spec §11a).

Source integrity is checked on EXACTLY the five protected code paths (src, tests, scripts, configs, docs) against the
frozen code commit: working tree vs commit, index (staged) vs commit, and untracked non-ignored files. Artifact-only
commits therefore never trip the source guard. Artifacts are checked against SHA-256 values held in SEPARATE freeze
records (never self-referential); freeze records themselves must be committed and unchanged since commit.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

PROTECTED_PATHS = ("src", "tests", "scripts", "configs", "docs")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


class GuardError(SystemExit):
    """Any integrity / authorization failure: the stage refuses to run."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)


def assert_commit(root: Path, commit: str) -> None:
    if not isinstance(commit, str) or not FULL_SHA.match(commit):
        raise GuardError(f"frozen code commit must be a full 40-character SHA, got {commit!r}")
    if _git(root, "cat-file", "-e", f"{commit}^{{commit}}").returncode != 0:
        raise GuardError(f"frozen code commit {commit} does not exist in this repository")


def assert_protected_paths_frozen(root: Path, commit: str, paths: Iterable[str] = PROTECTED_PATHS) -> None:
    assert_commit(root, commit)
    paths = list(paths)
    wt = _git(root, "diff", "--exit-code", "--name-only", commit, "--", *paths)
    if wt.returncode != 0:
        raise GuardError(f"protected code differs from frozen commit (working tree): {wt.stdout.split()[:5]}")
    idx = _git(root, "diff", "--cached", "--exit-code", "--name-only", commit, "--", *paths)
    if idx.returncode != 0:
        raise GuardError(f"protected code differs from frozen commit (staged): {idx.stdout.split()[:5]}")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "--", *paths).stdout.split()
    if untracked:
        raise GuardError(f"untracked non-ignored files in protected paths: {untracked[:5]}")


def verify_hashes(root: Path, expected: Mapping[str, str]) -> None:
    """expected: {repo-relative path: sha256}. Missing file or mismatch -> GuardError."""
    for rel, sha in expected.items():
        p = Path(root) / rel
        if not p.exists():
            raise GuardError(f"frozen artifact missing: {rel}")
        got = sha256_file(p)
        if got != sha:
            raise GuardError(f"frozen artifact hash mismatch: {rel} ({got[:12]} != {sha[:12]})")


def assert_committed_unchanged(root: Path, rels: Iterable[str]) -> None:
    """A freeze record must be tracked by git and identical to its committed version (protected by the commit id)."""
    for rel in rels:
        if _git(root, "ls-files", "--error-unmatch", "--", rel).returncode != 0:
            raise GuardError(f"freeze record not committed: {rel}")
        if _git(root, "diff", "--exit-code", "HEAD", "--", rel).returncode != 0 or \
                _git(root, "diff", "--cached", "--exit-code", "HEAD", "--", rel).returncode != 0:
            raise GuardError(f"freeze record modified after commit: {rel}")
