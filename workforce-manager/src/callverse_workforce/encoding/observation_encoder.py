"""Canonical NumPy encoding of ``RoutingObservation`` (version ``obs_encoder_v1``).

Frozen for all future policies and RL algorithms:

* fixed feature order = ``RoutingObservation`` field order;
* categoricals are one-hot over the full enum, in enum declaration order;
* numerics are clipped to fixed, documented bounds and scaled to [0, 1]
  (``hour_sin``/``hour_cos`` are kept in [-1, 1]); nothing is fitted on data;
* output is ``float32`` with a fixed dimension (``OBSERVATION_DIM`` = 43).

The bounds are part of the encoder version, NOT of a scenario config, so every
scenario (normal day, load check, later stress days) is encoded identically.
Values outside the bounds saturate. Changing anything here requires a new
encoder version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from enum import Enum
from typing import Callable

import numpy as np

from callverse_workforce.domain.enums import Channel, DayType, LanguageDialect, RequestType, Sensitivity, Urgency
from callverse_workforce.domain.observation import RoutingObservation

ENCODER_VERSION = "obs_encoder_v1"

CATEGORICAL_FEATURES: dict[str, type[Enum]] = {
    "request_type": RequestType,
    "urgency_pred": Urgency,
    "sensitivity_pred": Sensitivity,
    "channel": Channel,
    "language_dialect": LanguageDialect,
    "day_type": DayType,
}


@dataclass(frozen=True)
class NumericSpec:
    low: float
    high: float
    transform: str = "linear"  # "linear": (x-lo)/(hi-lo); "log1p": log1p(x-lo)/log1p(hi-lo); "identity": clip only
    unit: str = ""


# Fixed V1 bounds (SYNTHETIC_ASSUMPTION_V1 for the capacity-like ones).
NUMERIC_FEATURES: dict[str, NumericSpec] = {
    "request_value": NumericSpec(0.0, 1.0, unit="normalised value"),
    "waiting_time_so_far": NumericSpec(0.0, 60.0, unit="min"),
    "deadline_remaining_pred": NumericSpec(0.0, 4320.0, "log1p", unit="min (negative -> 0)"),
    "callback_allowed": NumericSpec(0.0, 1.0, unit="bool"),
    "human_requested": NumericSpec(0.0, 1.0, unit="bool"),
    "ai_resolution_score": NumericSpec(0.0, 1.0, unit="probability-like score"),
    "routing_attempt_count": NumericSpec(0.0, 3.0, unit="decisions"),
    "humans_available": NumericSpec(0.0, 20.0, unit="advisors"),
    "human_utilization": NumericSpec(0.0, 1.0, unit="fraction"),
    "human_queue_size": NumericSpec(0.0, 50.0, unit="contacts"),
    "oldest_human_wait": NumericSpec(0.0, 60.0, unit="min"),
    "ai_load": NumericSpec(0.0, 2.0, unit="sessions / capacity"),
    "recent_ai_failure_rate": NumericSpec(0.0, 1.0, unit="fraction"),
    "wait_queue_size": NumericSpec(0.0, 50.0, unit="contacts"),
    "callbacks_pending": NumericSpec(0.0, 200.0, unit="contacts"),
    "oldest_callback_age": NumericSpec(0.0, 240.0, unit="min"),
    "recent_arrival_rate": NumericSpec(0.0, 200.0, unit="contacts/hour"),
    "forecast_30min": NumericSpec(0.0, 100.0, unit="contacts"),
    "recent_abandonment_rate": NumericSpec(0.0, 1.0, unit="fraction"),
    "recent_service_level": NumericSpec(0.0, 1.0, unit="fraction"),
    "hour_sin": NumericSpec(-1.0, 1.0, "identity"),
    "hour_cos": NumericSpec(-1.0, 1.0, "identity"),
}

_OBS_FIELDS = tuple(f.name for f in fields(RoutingObservation))
assert set(_OBS_FIELDS) == set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES), "encoder must cover all 28 features"


def _scale(spec: NumericSpec) -> Callable[[float], float]:
    span = spec.high - spec.low
    if spec.transform == "identity":
        return lambda x: min(spec.high, max(spec.low, x))
    if spec.transform == "log1p":
        denom = math.log1p(span)
        return lambda x: math.log1p(min(span, max(0.0, x - spec.low))) / denom
    return lambda x: (min(spec.high, max(spec.low, x)) - spec.low) / span


def _build_layout() -> tuple[tuple[str, ...], list[tuple[str, Callable[[object], list[float]]]]]:
    names: list[str] = []
    plan: list[tuple[str, Callable[[object], list[float]]]] = []
    for field_name in _OBS_FIELDS:
        if field_name in CATEGORICAL_FEATURES:
            members = list(CATEGORICAL_FEATURES[field_name])
            names += [f"{field_name}={m.value}" for m in members]
            plan.append((field_name, lambda v, members=members: [1.0 if v is m else 0.0 for m in members]))
        else:
            scale = _scale(NUMERIC_FEATURES[field_name])
            names.append(field_name)
            plan.append((field_name, lambda v, scale=scale: [scale(float(v))]))
    return tuple(names), plan


FEATURE_NAMES, _PLAN = _build_layout()
OBSERVATION_DIM = len(FEATURE_NAMES)


class ObservationEncoder:
    """Deterministic, stateless ``RoutingObservation -> np.ndarray[float32]`` encoder."""

    version = ENCODER_VERSION
    feature_names = FEATURE_NAMES
    dim = OBSERVATION_DIM

    def encode(self, observation: RoutingObservation) -> np.ndarray:
        if not isinstance(observation, RoutingObservation):
            raise TypeError(f"ObservationEncoder only accepts RoutingObservation, got {type(observation).__name__}")
        values: list[float] = []
        for field_name, encode_field in _PLAN:
            value = getattr(observation, field_name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"non-finite value for {field_name}: {value}")
            values.extend(encode_field(value))
        return np.asarray(values, dtype=np.float32)

    def encode_batch(self, observations: list[RoutingObservation]) -> np.ndarray:
        if not observations:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.encode(o) for o in observations])

    def describe(self) -> list[dict[str, object]]:
        """Human-readable layout for debugging and documentation."""
        rows = []
        for i, name in enumerate(self.feature_names):
            base = name.split("=")[0]
            spec = NUMERIC_FEATURES.get(base)
            rows.append({
                "index": i,
                "name": name,
                "kind": "one_hot" if spec is None else spec.transform,
                "bounds": None if spec is None else [spec.low, spec.high],
                "unit": "" if spec is None else spec.unit,
            })
        return rows
