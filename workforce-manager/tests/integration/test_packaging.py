"""Package validation: declared dependencies cover all imports; optional isolated editable install."""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tomllib
import venv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
IMPORT_TO_DIST = {"yaml": "pyyaml", "stable_baselines3": "stable-baselines3", "sb3_contrib": "sb3-contrib"}


def _declared() -> set[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    deps = list(project["dependencies"]) + [d for extra in project.get("optional-dependencies", {}).values() for d in extra]
    return {re.split(r"[<>=!~\[ ]", d, maxsplit=1)[0].lower().replace("_", "-") for d in deps}


def _third_party_imports() -> set[str]:
    found: set[str] = set()
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module.split(".")[0])
    return {m for m in found if m not in sys.stdlib_module_names and m != "callverse_workforce"}


def test_every_runtime_import_is_a_declared_dependency():
    missing = {IMPORT_TO_DIST.get(m, m).lower() for m in _third_party_imports()} - _declared()
    assert not missing, f"undeclared runtime dependencies: {missing}"


@pytest.mark.skipif(os.environ.get("CALLVERSE_PACKAGING_TEST") != "1",
                    reason="slow/network: set CALLVERSE_PACKAGING_TEST=1 to build an isolated venv")
def test_editable_install_in_isolated_environment(tmp_path):  # B1-18
    env_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(env_dir)
    py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    clean_env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    subprocess.run([str(py), "-m", "pip", "install", "-q", "-e", str(ROOT)], check=True, env=clean_env)
    out = subprocess.run(
        [str(py), "-c", "import callverse_workforce, simpy, numpy, yaml, pydantic; print(callverse_workforce.__file__)"],
        check=True, capture_output=True, text=True, env=clean_env, cwd=tmp_path,
    )
    assert str(ROOT / "src") in out.stdout
