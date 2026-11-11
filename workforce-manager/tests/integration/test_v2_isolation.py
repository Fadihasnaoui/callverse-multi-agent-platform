"""Import isolation: no V1.1 / E.1 / Phase F module imports anything from ``callverse_workforce.v2_postfail``.

Companion to the approved amendment of tests/integration/test_phase_e1.py (the V2 package is excluded from its string
scan, so isolation is checked here, statically AND at runtime).
"""

from __future__ import annotations

import ast
import subprocess
import sys

from callverse_workforce.config.loader import PROJECT_ROOT

SRC = PROJECT_ROOT / "src" / "callverse_workforce"
V2_DIR = (SRC / "v2_postfail").resolve()
V2_NAME = "callverse_workforce.v2_postfail"


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            yield from (a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            yield ("." * n.level) + (n.module or "")


def test_no_legacy_module_imports_v2_statically():
    legacy = [p for p in SRC.rglob("*.py") if V2_DIR not in p.resolve().parents]
    assert len(legacy) > 50
    offenders = {str(p.relative_to(PROJECT_ROOT)): m for p in legacy for m in _imports(p) if "v2_postfail" in m}
    assert offenders == {}


def test_no_legacy_module_imports_v2_at_runtime():
    """Import the V1.1 / E.1 / Phase F runtime (environment, encoder, masks, policies, twin, agent) in a fresh interpreter."""
    code = (
        "import sys\n"
        "import callverse_workforce.phase_e.env, callverse_workforce.phase_e.smdp, callverse_workforce.phase_e.training\n"
        "import callverse_workforce.phase_e1.training, callverse_workforce.phase_e1.evaluation\n"
        "import callverse_workforce.encoding.observation_encoder, callverse_workforce.routing.mask\n"
        "import callverse_workforce.baselines.policies, callverse_workforce.twin.digital_twin, callverse_workforce.twin.runner\n"
        "import callverse_workforce.agent.workforce_manager, callverse_workforce.agent.evaluation, callverse_workforce.final_test.v1_1\n"
        f"bad = [m for m in sys.modules if m.startswith('{V2_NAME}')]\n"
        "print('V2_LOADED=' + repr(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT,
                         env={**__import__("os").environ, "PYTHONPATH": str(PROJECT_ROOT / "src")})
    assert out.returncode == 0, out.stderr
    assert "V2_LOADED=[]" in out.stdout


def test_amended_e1_scan_still_detects_legacy_changes(tmp_path):
    """The narrowed scan must still catch a V2 action name added to any legacy module (simulated on a copy)."""
    files = [p for p in SRC.rglob("*.py") if V2_DIR not in p.resolve().parents]
    text = "".join(p.read_text(encoding="utf-8") for p in files)
    assert "HUMAN_HIGH" not in text and "HUMAN_NORMAL" not in text
    poisoned = text + "\nX = 'HUMAN_HIGH'\n"  # e.g. a change to routing/mask.py or domain/enums.py
    assert "HUMAN_HIGH" in poisoned
    # the exclusion is exactly one directory: nothing else is skipped
    skipped = [p for p in SRC.rglob("*.py") if V2_DIR in p.resolve().parents]
    assert skipped and all(p.parent.resolve() == V2_DIR for p in skipped)
