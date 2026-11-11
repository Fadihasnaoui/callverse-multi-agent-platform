"""10-seed calibration audit and paired B1 vs B1.5 comparison."""

from __future__ import annotations

import json

import pytest

from callverse_workforce.calibration import (
    CALIBRATION_SEEDS,
    build_assumption_audit,
    build_paired_comparison,
    render_markdown,
    render_paired_markdown,
)
from callverse_workforce.config import load_config
from callverse_workforce.utils.serialization import to_jsonable


@pytest.fixture(scope="module")
def paired():
    return build_paired_comparison(load_config(), CALIBRATION_SEEDS)


def test_calibration_seed_list_is_fixed():
    assert CALIBRATION_SEEDS == tuple(range(1001, 1011))


def test_paired_comparison_uses_the_same_ten_seeds(paired):  # B1.5-28
    assert paired["seeds"] == list(CALIBRATION_SEEDS)
    assert paired["seeds_by_variant"]["b1"] == paired["seeds_by_variant"]["b1_5"] == list(CALIBRATION_SEEDS)
    assert list(paired["per_seed"]["b1"]) == list(paired["per_seed"]["b1_5"]) == [str(s) for s in CALIBRATION_SEEDS]


def test_variants_differ_only_in_patience_and_callback_model(paired):
    b1, b15 = paired["configs"]["b1"], paired["configs"]["b1_5"]
    assert (b1["patience"]["model"], b1["callback"]["model"]) == ("exponential", "b1")
    assert (b15["patience"]["model"], b15["callback"]["model"]) == ("weibull", "b1_5")
    for cfg in (b1, b15):
        cfg["patience"].pop("model")
        cfg["callback"].pop("model")
    assert b1 == b15  # reward, AI, recontact, everything else identical


def test_paired_effects_point_in_the_expected_direction(paired):
    s = paired["summary"]
    assert s["patience_below_1min"]["b1_5"]["mean"] < 0.03 < s["patience_below_1min"]["b1"]["mean"]
    assert s["callback_per_dial_answer_rate"]["b1_5"]["mean"] < s["callback_per_dial_answer_rate"]["b1"]["mean"]
    assert s["callback_success_rate"]["b1_5"]["mean"] < s["callback_success_rate"]["b1"]["mean"]
    assert paired["reconciled"] == {"b1": True, "b1_5": True}  # B1.5-30
    assert "-1 penalty for MAX_ATTEMPTS / WINDOW_EXPIRED may underprice" in paired["warnings"][0]
    assert "B1 vs B1.5" in render_paired_markdown(paired)


def test_ten_seed_audit_is_reproducible():  # B1.5-29
    cfg = load_config()
    a = json.dumps(to_jsonable(build_assumption_audit(cfg, CALIBRATION_SEEDS)), sort_keys=True)
    b = json.dumps(to_jsonable(build_assumption_audit(cfg, CALIBRATION_SEEDS)), sort_keys=True)
    assert a == b
    audit = json.loads(a)
    assert audit["run"]["seeds"] == list(CALIBRATION_SEEDS)
    assert audit["patience"]["observed_across_seeds"]["abandonment_rate"]["n"] == 10
    assert {"mean", "std", "p10", "median", "p90"} <= set(audit["patience"]["observed_across_seeds"]["abandonment_rate"])
    assert not any(f["code"] == "PATIENCE_MASS_NEAR_ZERO" for f in audit["patience"]["flags"])
    assert any(f["code"] == "CALLBACK_PENALTY_MAY_UNDERPRICE_LOST_CUSTOMER" for f in audit["callback"]["flags"])
    assert render_markdown(audit).startswith("# Calibration")
