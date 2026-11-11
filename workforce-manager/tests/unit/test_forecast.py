"""Observable-only forecaster: arithmetic and leakage contract (unit level)."""

from __future__ import annotations

import inspect
import typing

import pytest

from callverse_workforce.config.models import ForecastingConfig, ScenarioConfig, SimulationConfig
from callverse_workforce.forecasting.forecaster import ObservedCorrectionForecaster, ReferenceProfile, build_forecaster


def _cfg(**kw) -> ForecastingConfig:
    base = dict(reference_hourly_arrival_rates=[60.0, 120.0], lookback_minutes=60.0, smoothing_pseudo_count=0.0,
                min_correction=0.5, max_correction=2.0)
    base.update(kw)
    return ForecastingConfig(**base)


def test_reference_profile_integration():
    p = ReferenceProfile([60.0, 120.0])
    assert p.expected(0, 30) == pytest.approx(30.0)
    assert p.expected(45, 75) == pytest.approx(15.0 + 30.0)
    assert p.expected(110, 140) == pytest.approx(20.0)  # profile ends at 120
    assert p.expected(-30, 0) == 0.0 and p.expected(500, 530) == 0.0


def test_forecast_is_reference_times_observed_correction():
    f = ObservedCorrectionForecaster(_cfg(), horizon_minutes=30)
    observed = [i * 0.8 for i in range(75)]  # 75 arrivals in [0, 60) vs 60 expected
    assert f.correction(60.0, observed) == pytest.approx(75 / 60)
    assert f.forecast(60.0, observed) == pytest.approx(60.0 * 75 / 60)  # H(60, 90) = 60


def test_correction_is_clipped_and_smoothed():
    f = ObservedCorrectionForecaster(_cfg(), 30)
    assert f.correction(60.0, []) == 0.5  # nothing observed -> clipped at min
    assert f.correction(60.0, [0.1] * 1000) == 2.0  # clipped at max
    smooth = ObservedCorrectionForecaster(_cfg(smoothing_pseudo_count=10.0), 30)
    assert smooth.correction(1.0, []) == pytest.approx(10 / 11)  # early day: shrunk towards 1


def test_forecast_never_reads_arrivals_after_now():
    f = ObservedCorrectionForecaster(_cfg(smoothing_pseudo_count=5.0), 30)
    past = [5.0, 20.0, 31.0, 44.0]
    with_future = past + [50.0 + i * 0.1 for i in range(200)]  # 'future' entries must be ignored
    assert f.forecast(45.0, past) == f.forecast(45.0, with_future)


def test_forecaster_factory_cannot_receive_scenario_information():
    hints = typing.get_type_hints(build_forecaster)
    assert hints["config"] is ForecastingConfig
    assert set(inspect.signature(build_forecaster).parameters) == {"config", "horizon_minutes"}
    assert ScenarioConfig not in hints.values() and SimulationConfig not in hints.values()
    assert not set(ForecastingConfig.model_fields) & {"hourly_arrival_rates", "name", "purpose", "day_type"}
