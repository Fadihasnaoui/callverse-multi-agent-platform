"""Selectable patience models (B1 exponential kept; B1.5 Weibull with floor)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from pydantic import ValidationError

from callverse_workforce.config import apply_overrides, load_config
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.twin.patience import PatienceDistribution, sample_patience
from callverse_workforce.utils.rng import RngManager

BASE = load_config()
EXP = apply_overrides(BASE, {"patience.model": "exponential"})
LOGN = apply_overrides(BASE, {"patience.model": "lognormal"})


def _day(cfg, seed=7):
    return ContactGenerator(cfg, RngManager(seed)).generate_day()


def test_default_b1_5_model_is_weibull_with_floor():  # B1.5-2
    assert BASE.patience.model == "weibull"
    assert BASE.patience.weibull.shape == 1.5 and BASE.patience.floor_minutes == 0.5


def test_old_exponential_model_still_selectable():  # B1.5-1
    assert EXP.patience.model == "exponential" and EXP.patience.floor_minutes == 0.0
    rng_a, rng_b = np.random.default_rng(3), np.random.default_rng(3)
    assert sample_patience(rng_a, 6.0, EXP.patience) == float(rng_b.exponential(6.0))


@pytest.mark.parametrize("cfg", [BASE, EXP, LOGN], ids=["weibull", "exponential", "lognormal"])
def test_models_preserve_the_configured_mean(cfg):
    rng = np.random.default_rng(11)
    samples = np.array([sample_patience(rng, 8.0, cfg.patience) for _ in range(40_000)])
    assert samples.mean() == pytest.approx(8.0, rel=0.03)
    d = PatienceDistribution.from_config(8.0, cfg.patience)
    if d.model == "weibull":
        assert d.floor + d.scale * math.gamma(1 + 1 / d.shape) == pytest.approx(8.0)


def test_patience_floor_respected():  # B1.5-3
    day = _day(BASE)
    assert min(c.patience_minutes for c in day) >= BASE.patience.floor_minutes
    assert min(c.patience_minutes for c in _day(LOGN)) >= LOGN.patience.floor_minutes
    assert PatienceDistribution.from_config(6.0, BASE.patience).cdf(0.5) == 0.0


def test_patience_sampling_reproducible():  # B1.5-4
    assert [c.patience_minutes for c in _day(BASE, 4)] == [c.patience_minutes for c in _day(BASE, 4)]
    assert [c.patience_minutes for c in _day(BASE, 4)] != [c.patience_minutes for c in _day(BASE, 5)]


def test_patience_model_does_not_shift_arrivals_or_other_attributes():
    a, b = _day(BASE), _day(EXP)
    strip = lambda c: (c.arrival_time, c.request_type, c.channel, c.urgency, c.human_service_minutes, c.request_value)
    assert [strip(c) for c in a] == [strip(c) for c in b]


def test_new_model_removes_near_zero_mass():  # B1.5-5
    frac = lambda day, x: np.mean([c.patience_minutes < x for c in day])
    new, old = _day(BASE), _day(EXP)
    assert frac(new, 1.0) < 0.03 < 0.07 < frac(old, 1.0)
    assert frac(new, 2.0) < frac(old, 2.0) / 2
    # analytic: the worst configured cell (FAQ/PHONE, mean 6) has P(<1 min) ~ 2.3% vs 15.4% before
    assert PatienceDistribution.from_config(6.0, BASE.patience).cdf(1.0) == pytest.approx(0.0232, abs=0.001)
    assert PatienceDistribution.from_config(6.0, EXP.patience).cdf(1.0) == pytest.approx(1 - math.exp(-1 / 6))


def test_quantiles_match_cdf():
    for cfg in (BASE, EXP, LOGN):
        d = PatienceDistribution.from_config(7.0, cfg.patience)
        for p in (0.1, 0.5, 0.9):
            assert d.cdf(d.quantile(p)) == pytest.approx(p, abs=1e-6)


def test_floor_must_be_below_every_mean():
    with pytest.raises(ValidationError):
        apply_overrides(BASE, {"patience.weibull.floor_minutes": 5.0})  # FAQ/PHONE recontact mean is 4.8
