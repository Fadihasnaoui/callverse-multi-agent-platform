"""Frozen seed registry (single source of truth for every experiment phase).

* ``CALIBRATION_SEEDS``        1001-1010  already used by the B1.5 calibration audit; never reused
* ``B2_VALIDATION_SEEDS``      1011-1020  baseline tuning / selection (NORMAL only)
* ``B2_NORMAL_EVAL_SEEDS``     1101-1120  baseline evaluation, NORMAL
* ``B2_LOAD_CHECK_EVAL_SEEDS`` 1201-1220  baseline evaluation, LOAD_CHECK (development only)
* ``HEADROOM_PRIMARY_SEEDS``   1301-1310  Phase C headroom days
* ``HEADROOM_EXTENSION_SEEDS`` 1311-1320  Phase C extra days, used ONLY if confidence intervals stay too wide
* ``PHASE_D_AUDIT_SEEDS``      1401-1410  Phase D development / V1.1 audit
* ``PHASE_D_VALIDATION_SEEDS`` 1411-1420  Phase D baseline validation (tuning)
* ``PHASE_D_HEADROOM_SEEDS``   1421-1430  Phase D headroom and baseline evaluation
* ``PHASE_D_EXTENSION_SEEDS``  1431-1440  Phase D extra days, only under the pre-registered rule
* ``PPO_VALIDATION_SEEDS``     1501-1580  Phase E validation (8 families x 10): tuning / checkpoint selection
* ``PPO_DEV_EVAL_SEEDS``       1601-1680  Phase E development evaluation (8 x 10): opened only after freezing
* ``PPO_E1_DEV_SEEDS``         1681-1760  Phase E.1 development evaluation (new; opened once)
* ``PPO_TRAINING_SEEDS``       >= 100000  procedural training days
* ``FINAL_TEST_RESERVED``      2000-2999  untouched until the final test phase
"""

from __future__ import annotations

from typing import Iterable

CALIBRATION_SEEDS: tuple[int, ...] = tuple(range(1001, 1011))
B2_VALIDATION_SEEDS: tuple[int, ...] = tuple(range(1011, 1021))
B2_NORMAL_EVAL_SEEDS: tuple[int, ...] = tuple(range(1101, 1121))
B2_LOAD_CHECK_EVAL_SEEDS: tuple[int, ...] = tuple(range(1201, 1221))
HEADROOM_PRIMARY_SEEDS: tuple[int, ...] = tuple(range(1301, 1311))
HEADROOM_EXTENSION_SEEDS: tuple[int, ...] = tuple(range(1311, 1321))
PHASE_D_AUDIT_SEEDS: tuple[int, ...] = tuple(range(1401, 1411))
PHASE_D_VALIDATION_SEEDS: tuple[int, ...] = tuple(range(1411, 1421))
PHASE_D_HEADROOM_SEEDS: tuple[int, ...] = tuple(range(1421, 1431))
PHASE_D_EXTENSION_SEEDS: tuple[int, ...] = tuple(range(1431, 1441))
PPO_VALIDATION_SEEDS: tuple[int, ...] = tuple(range(1501, 1581))
PPO_DEV_EVAL_SEEDS: tuple[int, ...] = tuple(range(1601, 1681))
PPO_E1_DEV_SEEDS: tuple[int, ...] = tuple(range(1681, 1761))
PPO_TRAINING_SEED_MIN = 100_000
FINAL_TEST_RESERVED: range = range(2000, 3000)
# Permanent split of the reserved range (docs/final_test_v1_1_prereg.md)
FINAL_V1_1_SEEDS: tuple[int, ...] = tuple(range(2000, 2240))  # V1.1 final test, opened once
V2_RESERVED_SEEDS: tuple[int, ...] = tuple(range(2240, 3000))  # closed for future V2


# Phase F (agentic runtime): fresh band 3000-99999 never used before (collision-audited before use)
PHASE_F_DEV_SEEDS: tuple[int, ...] = tuple(range(3000, 3160))  # 8 families x 20
PHASE_F_TEST_SEEDS: tuple[int, ...] = tuple(range(3500, 3770))  # 8 x 30 + 30 extra AI_DEGRADATION (3740-3769)


# V2 post-AI-failure headroom (docs/v2_headroom_spec.md): disjoint evaluation and calibration sets
V2_HEADROOM_EVAL_SEEDS: tuple[int, ...] = tuple(range(4000, 4080))  # 8 families x 10 (evaluation states only)
V2_HEADROOM_CAL_SEEDS: tuple[int, ...] = tuple(range(4080, 4090))  # 10 NORMAL days (f_ref calibration only)
assert not set(V2_HEADROOM_EVAL_SEEDS) & set(V2_HEADROOM_CAL_SEEDS)


def assert_v2_reserved_untouched(seeds) -> None:
    bad = sorted(set(seeds) & set(V2_RESERVED_SEEDS))
    if bad:
        raise SeedPolicyError(f"seeds {bad[:5]}... belong to the V2 reserve 2240-2999 and are closed")

SEED_SETS: dict[str, tuple[int, ...]] = {
    "calibration": CALIBRATION_SEEDS,
    "b2_validation": B2_VALIDATION_SEEDS,
    "b2_normal_eval": B2_NORMAL_EVAL_SEEDS,
    "b2_load_check_eval": B2_LOAD_CHECK_EVAL_SEEDS,
    "headroom_primary": HEADROOM_PRIMARY_SEEDS,
    "headroom_extension": HEADROOM_EXTENSION_SEEDS,
    "phase_d_audit": PHASE_D_AUDIT_SEEDS,
    "phase_d_validation": PHASE_D_VALIDATION_SEEDS,
    "phase_d_headroom": PHASE_D_HEADROOM_SEEDS,
    "phase_d_extension": PHASE_D_EXTENSION_SEEDS,
    "ppo_validation": PPO_VALIDATION_SEEDS,
    "ppo_dev_eval": PPO_DEV_EVAL_SEEDS,
    "ppo_e1_dev_eval": PPO_E1_DEV_SEEDS,
    "final_test_reserved": tuple(FINAL_TEST_RESERVED),
}


class SeedPolicyError(ValueError):
    """Raised when a phase tries to use seeds outside its allowed set."""


def require_seeds(seeds: Iterable[int], allowed: tuple[int, ...], purpose: str) -> tuple[int, ...]:
    seeds = tuple(seeds)
    if not seeds:
        raise SeedPolicyError(f"{purpose}: empty seed list")
    outside = sorted(set(seeds) - set(allowed))
    if outside:
        raise SeedPolicyError(f"{purpose}: seeds {outside} are not in the allowed set {allowed[0]}-{allowed[-1]}")
    return seeds


def registry_dict() -> dict[str, object]:
    return {
        "calibration": list(CALIBRATION_SEEDS),
        "b2_validation": list(B2_VALIDATION_SEEDS),
        "b2_normal_eval": list(B2_NORMAL_EVAL_SEEDS),
        "b2_load_check_eval": list(B2_LOAD_CHECK_EVAL_SEEDS),
        "headroom_primary": list(HEADROOM_PRIMARY_SEEDS),
        "headroom_extension_reserved": list(HEADROOM_EXTENSION_SEEDS),
        "phase_d_audit": list(PHASE_D_AUDIT_SEEDS),
        "phase_d_validation": list(PHASE_D_VALIDATION_SEEDS),
        "phase_d_headroom": list(PHASE_D_HEADROOM_SEEDS),
        "phase_d_extension_reserved": list(PHASE_D_EXTENSION_SEEDS),
        "ppo_validation": list(PPO_VALIDATION_SEEDS),
        "ppo_dev_eval": list(PPO_DEV_EVAL_SEEDS),
        "ppo_e1_dev_eval": list(PPO_E1_DEV_SEEDS),
        "ppo_training_seed_min": PPO_TRAINING_SEED_MIN,
        "final_test_reserved": [FINAL_TEST_RESERVED.start, FINAL_TEST_RESERVED.stop - 1],
        "rules": [
            "calibration seeds (1001-1010) are never reused for validation or evaluation",
            "final-test seeds (2000-2999) are untouched until the final test phase",
            "headroom extension seeds (1311-1320) are used only if Phase C intervals remain too wide",
            "all sets are pairwise disjoint",
        ],
    }
