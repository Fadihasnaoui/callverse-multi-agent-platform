"""Calibration seeds (re-exported from the frozen registry in ``callverse_workforce.seeds``)."""

from __future__ import annotations

from callverse_workforce.seeds import (
    B2_LOAD_CHECK_EVAL_SEEDS,
    B2_NORMAL_EVAL_SEEDS,
    B2_VALIDATION_SEEDS,
    CALIBRATION_SEEDS,
    FINAL_TEST_RESERVED,
)

RESERVED_RANGES = {
    "calibration": (CALIBRATION_SEEDS[0], CALIBRATION_SEEDS[-1]),
    "b2_validation": (B2_VALIDATION_SEEDS[0], B2_VALIDATION_SEEDS[-1]),
    "b2_normal_eval": (B2_NORMAL_EVAL_SEEDS[0], B2_NORMAL_EVAL_SEEDS[-1]),
    "b2_load_check_eval": (B2_LOAD_CHECK_EVAL_SEEDS[0], B2_LOAD_CHECK_EVAL_SEEDS[-1]),
    "final_test_reserved": (FINAL_TEST_RESERVED.start, FINAL_TEST_RESERVED.stop - 1),
}

__all__ = ["CALIBRATION_SEEDS", "RESERVED_RANGES"]
