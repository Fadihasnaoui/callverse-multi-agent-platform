"""MANDATORY leakage test: two worlds with identical observable history up to t
and different hidden futures must produce identical forecasts (and observations) at t."""

from __future__ import annotations

from callverse_workforce.domain.enums import RoutingAction
from helpers import fixed, make_config, make_contact, run_scripted

T = 100.0
SHARED = [(i, 10.0 * i) for i in range(11)]  # arrivals at 0, 10, ..., 100 (= T)


def _world(hidden_rates, extra_arrivals):
    cfg = make_config(scenario__hourly_arrival_rates=hidden_rates)
    contacts = [make_contact(i, t, request_value=0.1 * (i % 5)) for i, t in SHARED]
    contacts += [make_contact(100 + j, t) for j, t in enumerate(extra_arrivals)]
    _, result, decisions = run_scripted(cfg, contacts, fixed(RoutingAction.HUMAN))
    return {round(d.timestamp, 6): d.observation for d in decisions}


def test_forecast_identical_until_observable_history_diverges():
    # World A: calm hidden future, one later arrival. World B: hidden 3x profile + surge right after T.
    calm = _world([40.0] * 12, [120.0])
    surge = _world([120.0] * 12, [T + 1 + 0.3 * k for k in range(60)] + [120.0])

    for t in sorted(calm):
        if t <= T:
            assert calm[t].forecast_30min == surge[t].forecast_30min, t
            assert calm[t] == surge[t], t  # the whole observation is identical, not only the forecast

    # After B's 60 extra arrivals are OBSERVED (101..119), forecasts diverge at t=120
    # (the calm world's correction sits at the clip floor, the surge world's does not).
    assert surge[120.0].forecast_30min > calm[120.0].forecast_30min
    assert surge[120.0].recent_arrival_rate > calm[120.0].recent_arrival_rate


def test_hidden_profile_alone_never_changes_any_observation():
    contacts = [make_contact(i, t) for i, t in SHARED]
    obs = []
    for rates in ([10.0] * 12, [400.0] * 12):
        cfg = make_config(scenario__hourly_arrival_rates=rates)
        _, _, decisions = run_scripted(cfg, contacts, fixed(RoutingAction.HUMAN))
        obs.append([d.observation for d in decisions])
    assert obs[0] == obs[1]
