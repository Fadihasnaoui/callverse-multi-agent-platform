"""DEVELOPMENT_LOAD_CHECK_ONLY stressed paths and calibration-audit reproducibility."""

from __future__ import annotations

import json

import pytest

from callverse_workforce.calibration import build_assumption_audit
from callverse_workforce.config import load_config
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.routing import DevelopmentPolicy
from callverse_workforce.twin import run_simulation
from callverse_workforce.utils.serialization import to_jsonable

LOAD_CHECK = PROJECT_ROOT / "configs" / "load_check.yaml"


@pytest.fixture(scope="module")
def stressed():
    return run_simulation(load_config(LOAD_CHECK), DevelopmentPolicy())


@pytest.fixture(scope="module")
def normal():
    return run_simulation(load_config(), DevelopmentPolicy())


def test_load_check_is_labelled_development_only():
    cfg = load_config(LOAD_CHECK)
    assert cfg.scenario.purpose == "DEVELOPMENT_LOAD_CHECK_ONLY"
    assert "DEVELOPMENT_LOAD_CHECK_ONLY" in LOAD_CHECK.read_text(encoding="utf-8").splitlines()[1]
    # the historical forecast reference is NOT copied from the stressed (hidden) profile
    assert cfg.forecasting.reference_hourly_arrival_rates == load_config().forecasting.reference_hourly_arrival_rates
    assert cfg.forecasting.reference_hourly_arrival_rates != cfg.scenario.hourly_arrival_rates


def test_load_check_exercises_stressed_paths(stressed, normal):  # B1-9
    m, n = stressed.metrics, normal.metrics
    s = m["stress_indicators"]
    assert s["max_human_queue_size"] >= 10
    assert s["max_human_queue_size"] > 2 * n["stress_indicators"]["max_human_queue_size"]
    assert m["online_wait_minutes"]["p90"] > 2 * n["online_wait_minutes"]["p90"]
    assert m["abandonments"] > 0 and m["missed_deadlines"] > 0 and m["unfinished_contacts"] > 0
    assert s["max_callbacks_pending"] > 0
    assert "CALLBACK_QUEUE" in s["unfinished_by_state"]  # callbacks still pending at end of day
    assert m["reconciliation"]["balanced"]


def test_calibration_audit_is_reproducible():  # B1-10
    cfg = load_config()
    a = json.dumps(to_jsonable(build_assumption_audit(cfg, seeds=(1001, 1002))), sort_keys=True)
    b = json.dumps(to_jsonable(build_assumption_audit(cfg, seeds=(1001, 1002))), sort_keys=True)
    assert a == b
    audit = json.loads(a)
    for section in ("patience", "ai_behavior", "callback", "human_service", "structural_assumptions"):
        assert section in audit
    assert audit["label"] == "SYNTHETIC_ASSUMPTION_V1"


def test_audit_reports_without_changing_config():
    cfg = load_config()
    before = cfg.model_dump()
    audit = build_assumption_audit(cfg, seeds=(1001,))
    assert cfg.model_dump() == before
    assert any(f["code"] == "AI_RATES_ARE_PLACEHOLDERS" for f in audit["ai_behavior"]["flags"])
    assert any(f["code"] == "PATIENCE_PARAMETERS_UNCALIBRATED" for f in audit["patience"]["flags"])
