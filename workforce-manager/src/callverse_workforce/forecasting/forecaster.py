"""Short-term demand forecast (``forecast_30min``), observable information only.

Leakage contract
----------------
A forecaster may use ONLY:
  * a historical/reference arrival profile known before the day starts;
  * the current clock time;
  * arrival times already observed (``<= now``).
It never receives the scenario config, today's true arrival profile, future
arrivals, hidden multipliers or scenario labels. ``build_forecaster`` takes the
``ForecastingConfig`` section only, so the hidden ``scenario`` section cannot
reach it.

V1 algorithm (``ObservedCorrectionForecaster``)
------------------------------------------------
    H(a, b)    = expected arrivals in [a, b) under the reference profile
    n_obs      = observed arrivals in [now - L, now]
    n_exp      = H(now - L, now)
    correction = clip((n_obs + k) / (n_exp + k), c_min, c_max)
    forecast   = H(now, now + horizon) * correction

``k`` (pseudo-count) shrinks the correction towards 1 when little has been
observed, e.g. at the start of the day.
"""

from __future__ import annotations

import bisect
from typing import Protocol, Sequence

from callverse_workforce.config.models import ForecastingConfig


class DemandForecaster(Protocol):
    def forecast(self, now: float, observed_arrival_times: Sequence[float]) -> float:
        """Expected arrivals in ``[now, now + horizon)`` given arrivals observed so far (sorted)."""
        ...


class ReferenceProfile:
    """Piecewise-constant hourly reference arrival profile, starting at t=0."""

    def __init__(self, hourly_rates: Sequence[float]) -> None:
        self.hourly_rates = tuple(hourly_rates)
        self.end = 60.0 * len(self.hourly_rates)

    def expected(self, start: float, end: float) -> float:
        start, end = max(0.0, start), min(end, self.end)
        total, t = 0.0, start
        while t < end:
            hour = int(t // 60)
            seg_end = min(end, (hour + 1) * 60.0)
            total += self.hourly_rates[hour] / 60.0 * (seg_end - t)
            t = seg_end
        return total


class ObservedCorrectionForecaster:
    """Reference profile scaled by a smoothed, clipped correction from observed arrivals."""

    def __init__(self, config: ForecastingConfig, horizon_minutes: float) -> None:
        self.config = config
        self.profile = ReferenceProfile(config.reference_hourly_arrival_rates)
        self.horizon_minutes = horizon_minutes

    def correction(self, now: float, observed_arrival_times: Sequence[float]) -> float:
        c = self.config
        lo = now - c.lookback_minutes
        # observed_arrival_times is sorted; count arrivals in [lo, now] without looking past now
        n_obs = bisect.bisect_right(observed_arrival_times, now) - bisect.bisect_left(observed_arrival_times, lo)
        n_exp = self.profile.expected(lo, now)
        k = c.smoothing_pseudo_count
        if n_exp + k <= 0:
            return 1.0
        return min(c.max_correction, max(c.min_correction, (n_obs + k) / (n_exp + k)))

    def forecast(self, now: float, observed_arrival_times: Sequence[float]) -> float:
        base = self.profile.expected(now, now + self.horizon_minutes)
        return base * self.correction(now, observed_arrival_times)


def build_forecaster(config: ForecastingConfig, horizon_minutes: float) -> DemandForecaster:
    """The only factory used by the twin: receives the forecasting section, never the scenario."""
    return ObservedCorrectionForecaster(config, horizon_minutes)
