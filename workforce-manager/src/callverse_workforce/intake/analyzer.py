"""Simulated intake / request-analysis layer.

In production ``analyze_request`` will be an LLM or classifier. Here it is a
noisy function of the hidden truth whose error rates come from
``IntakeNoiseConfig``. The output, ``ObservedContactFeatures``, is the only
per-contact information routing policies ever see.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Protocol, TypeVar

import numpy as np

from callverse_workforce.config.models import IntakeNoiseConfig
from callverse_workforce.domain.contact import GroundTruthContact, ObservedContactFeatures
from callverse_workforce.domain.enums import RequestType, Sensitivity, Urgency

E = TypeVar("E", bound=Enum)


class RequestAnalyzer(Protocol):
    def analyze_request(
        self, truth: GroundTruthContact, rng: np.random.Generator, now: float
    ) -> ObservedContactFeatures: ...


class SimulatedIntakeAnalyzer:
    """Noisy analyzer: truth -> predicted features."""

    def __init__(self, noise: IntakeNoiseConfig) -> None:
        self.noise = noise

    def analyze_request(
        self, truth: GroundTruthContact, rng: np.random.Generator, now: float
    ) -> ObservedContactFeatures:
        n = self.noise
        # Fixed draw order keeps predictions reproducible per contact.
        u_type, u_urg, u_sens = rng.random(3)
        z_deadline, z_score = rng.standard_normal(2)
        shift_urg, shift_sens = rng.random(2)
        type_pick = rng.random()

        request_type = truth.request_type
        if u_type < n.request_type_error_prob:
            others = [t for t in RequestType if t is not truth.request_type]
            request_type = others[int(type_pick * len(others))]

        urgency = truth.urgency
        if u_urg < n.urgency_error_prob:
            urgency = _adjacent(Urgency, truth.urgency, shift_urg)

        sensitivity = truth.sensitivity
        if u_sens < n.sensitivity_error_prob:
            sensitivity = _adjacent(Sensitivity, truth.sensitivity, shift_sens)

        predicted_offset = truth.deadline_offset_minutes * math.exp(n.deadline_log_sigma * z_deadline)
        score = float(np.clip(truth.ai_correct_prob + n.ai_score_noise_std * z_score, 0.0, 1.0))

        return ObservedContactFeatures(
            contact_id=truth.contact_id,
            observed_at=now,
            request_type=request_type,
            urgency_pred=urgency,
            sensitivity_pred=sensitivity,
            channel=truth.channel,
            language_dialect=truth.language_dialect,
            request_value=truth.request_value,
            predicted_deadline_at=truth.arrival_time + predicted_offset,
            callback_allowed=truth.callback_allowed,
            human_requested=truth.human_requested,
            ai_resolution_score=score,
        )


def _adjacent(enum_cls: type[E], value: E, u: float) -> E:
    """Move one ordinal level up or down (direction chosen by ``u``; clamped at the ends)."""
    levels = list(enum_cls)
    i = levels.index(value)
    if i == 0:
        return levels[1]
    if i == len(levels) - 1:
        return levels[-2]
    return levels[i - 1] if u < 0.5 else levels[i + 1]
