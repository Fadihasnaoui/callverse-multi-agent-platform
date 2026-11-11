"""Typed configuration models.

All numeric synthetic assumptions live in YAML files (see
``configs/normal_day.yaml``), not in code. These models only define structure
and validation. Values are SYNTHETIC_ASSUMPTION_V1 unless stated otherwise.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from callverse_workforce.domain.enums import (
    Channel,
    DayType,
    LanguageDialect,
    RequestType,
    Sensitivity,
    Urgency,
)

_PROB_TOL = 1e-6


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _check_distribution(probs: dict, label: str) -> None:
    if any(p < 0 for p in probs.values()):
        raise ValueError(f"{label}: probabilities must be non-negative")
    if abs(sum(probs.values()) - 1.0) > _PROB_TOL:
        raise ValueError(f"{label}: probabilities must sum to 1 (got {sum(probs.values())})")


class MeanCv(_Frozen):
    """Lognormal distribution parameterised by mean and coefficient of variation."""

    mean: float = Field(gt=0)
    cv: float = Field(ge=0)


class BetaParams(_Frozen):
    a: float = Field(gt=0)
    b: float = Field(gt=0)


class Range(_Frozen):
    min: float = Field(ge=0)
    max: float = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Range:
        if self.max < self.min:
            raise ValueError("range max must be >= min")
        return self


ScenarioPurpose = Literal[
    "SYNTHETIC_NORMAL_DAY_V1",
    "DEVELOPMENT_LOAD_CHECK_ONLY",
    "HEADROOM_DIAGNOSTIC_ONLY",
    "TWIN_V1_1_SWEEP",
    "PREDEFINED_V1_SCENARIO",
    "PPO_V1_1_ENVIRONMENT",
]


class ScenarioConfig(_Frozen):
    """Hidden day-generation settings. Never visible to policies or the forecaster."""

    name: str
    purpose: ScenarioPurpose
    day_type: DayType
    start_hour: float = Field(ge=0, lt=24)
    arrival_window_minutes: float = Field(ge=0)
    drain_minutes: float = Field(ge=0)
    hourly_arrival_rates: list[float]  # contacts/hour, one entry per hour of the arrival window
    # Nested extra demand: layer l adds an independent Poisson stream with rate fraction_l x base profile,
    # generated from dedicated streams, so base contacts are untouched (NORMAL subset X1.3 subset X1.5).
    extra_load_layers: list[float] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rates_cover_window(self) -> ScenarioConfig:
        needed = math.ceil(self.arrival_window_minutes / 60.0)
        if len(self.hourly_arrival_rates) < needed:
            raise ValueError(f"need {needed} hourly rates, got {len(self.hourly_arrival_rates)}")
        if any(r < 0 for r in self.hourly_arrival_rates):
            raise ValueError("arrival rates must be non-negative")
        if any(f < 0 for f in self.extra_load_layers):
            raise ValueError("extra load layers must be non-negative")
        return self

    @property
    def horizon_minutes(self) -> float:
        return self.arrival_window_minutes + self.drain_minutes


class RequestTypeConfig(_Frozen):
    share: float = Field(ge=0, le=1)
    urgency_probs: dict[Urgency, float]
    sensitivity_probs: dict[Sensitivity, float]
    request_value_beta: BetaParams
    human_service_minutes: MeanCv
    ai_service_minutes: MeanCv
    patience_mean_minutes: float = Field(gt=0)
    ai_correct_prob: float = Field(ge=0, le=1)
    ai_false_resolution_prob: float = Field(ge=0, le=1)
    callback_allowed_prob: float = Field(ge=0, le=1)
    human_requested_prob: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _valid(self) -> RequestTypeConfig:
        _check_distribution(self.urgency_probs, "urgency_probs")
        _check_distribution(self.sensitivity_probs, "sensitivity_probs")
        if self.ai_correct_prob + self.ai_false_resolution_prob > 1.0 + _PROB_TOL:
            raise ValueError("ai_correct_prob + ai_false_resolution_prob must be <= 1")
        return self


class ChannelConfig(_Frozen):
    share: float = Field(ge=0, le=1)
    patience_multiplier: float = Field(gt=0)


class LanguageConfig(_Frozen):
    share: float = Field(ge=0, le=1)
    ai_correct_multiplier: float = Field(ge=0)


class HumanPoolConfig(_Frozen):
    n_advisors: int = Field(ge=1)


class AIConfig(_Frozen):
    available: bool
    max_concurrency: int = Field(ge=1)


class WeibullPatienceParams(_Frozen):
    shape: float = Field(gt=0)
    floor_minutes: float = Field(ge=0)


class LognormalPatienceParams(_Frozen):
    cv: float = Field(gt=0)
    floor_minutes: float = Field(ge=0)


class PatienceConfig(_Frozen):
    """Customer patience model (cumulative online waiting tolerated).

    The configured MEAN per request type x channel (``patience_mean_minutes`` x
    ``patience_multiplier``) is preserved by every model:

    * ``exponential`` (B1, kept for reproducible comparison): Exp(mean), no floor.
    * ``weibull``: floor + scale * Weibull(shape), scale = (mean - floor) / Gamma(1 + 1/shape).
    * ``lognormal``: floor + LogNormal with mean (mean - floor) and coefficient of variation ``cv``.
    """

    model: Literal["exponential", "weibull", "lognormal"]
    weibull: WeibullPatienceParams
    lognormal: LognormalPatienceParams

    @property
    def floor_minutes(self) -> float:
        if self.model == "weibull":
            return self.weibull.floor_minutes
        if self.model == "lognormal":
            return self.lognormal.floor_minutes
        return 0.0


class CallbackB1Params(_Frozen):
    """B1 callback model: retries until max attempts, no window, no dropout (kept for comparison)."""

    reach_probability: float = Field(ge=0, le=1)
    max_attempts: int = Field(ge=1)


class CallbackB15Params(_Frozen):
    """B1.5 callback model: completion window and pending-customer dropout."""

    answer_probability: float = Field(ge=0, le=1)
    max_attempts: int = Field(ge=1)
    max_delay_minutes: float = Field(gt=0)
    dropout_probability_per_hour: float = Field(ge=0, lt=1)


class CallbackConfig(_Frozen):
    model: Literal["b1", "b1_5"]
    available: bool
    max_pending: int = Field(ge=0)
    retry_delay_minutes: float = Field(ge=0)
    attempt_minutes: float = Field(gt=0)
    b1: CallbackB1Params
    b1_5: CallbackB15Params

    @property
    def answer_probability(self) -> float:
        return self.b1.reach_probability if self.model == "b1" else self.b1_5.answer_probability

    @property
    def max_attempts(self) -> int:
        return self.b1.max_attempts if self.model == "b1" else self.b1_5.max_attempts

    @property
    def max_delay_minutes(self) -> float | None:
        """Completion window after creation; ``None`` = no window (B1)."""
        return None if self.model == "b1" else self.b1_5.max_delay_minutes

    @property
    def dropout_hazard_per_minute(self) -> float:
        """Constant hazard lambda with P(dropout within 60 min pending) = dropout_probability_per_hour."""
        if self.model == "b1" or self.b1_5.dropout_probability_per_hour == 0:
            return 0.0
        return -math.log(1.0 - self.b1_5.dropout_probability_per_hour) / 60.0


class WaitConfig(_Frozen):
    reevaluation_minutes: float = Field(gt=0)
    max_waits: int = Field(ge=0)


class MaskConfig(_Frozen):
    urgent_deadline_threshold_minutes: float = Field(ge=0)


class FalseAIDiscoveryConfig(_Frozen):
    min_delay_minutes: float = Field(ge=0)
    mean_extra_delay_minutes: float = Field(ge=0)
    settle_at_horizon: bool


class IntakeNoiseConfig(_Frozen):
    request_type_error_prob: float = Field(ge=0, le=1)
    urgency_error_prob: float = Field(ge=0, le=1)
    sensitivity_error_prob: float = Field(ge=0, le=1)
    deadline_log_sigma: float = Field(ge=0)
    ai_score_noise_std: float = Field(ge=0)

    @classmethod
    def noiseless(cls) -> IntakeNoiseConfig:
        return cls(
            request_type_error_prob=0.0,
            urgency_error_prob=0.0,
            sensitivity_error_prob=0.0,
            deadline_log_sigma=0.0,
            ai_score_noise_std=0.0,
        )


class ObservationConfig(_Frozen):
    arrival_rate_window_minutes: float = Field(gt=0)
    ai_failure_window_minutes: float = Field(gt=0)
    abandonment_window_minutes: float = Field(gt=0)
    service_level_window_minutes: float = Field(gt=0)
    service_level_threshold_minutes: float = Field(ge=0)
    forecast_horizon_minutes: float = Field(gt=0)


class ForecastingConfig(_Frozen):
    """Observable-only forecast settings.

    ``reference_hourly_arrival_rates`` is the HISTORICAL reference profile known
    before the day starts. It is deliberately separate from
    ``scenario.hourly_arrival_rates`` (today's hidden truth).
    """

    reference_hourly_arrival_rates: list[float]
    lookback_minutes: float = Field(gt=0)
    smoothing_pseudo_count: float = Field(ge=0)
    min_correction: float = Field(gt=0)
    max_correction: float = Field(gt=0)

    @model_validator(mode="after")
    def _valid(self) -> ForecastingConfig:
        if self.max_correction < self.min_correction:
            raise ValueError("max_correction must be >= min_correction")
        if any(r < 0 for r in self.reference_hourly_arrival_rates):
            raise ValueError("reference rates must be non-negative")
        return self


class RecontactConfig(_Frozen):
    """Delayed recontact after a discovered false AI resolution."""

    enabled: bool
    max_depth: int = Field(ge=0, le=1)  # frozen design decision: at most one recontact per root
    probability: float = Field(ge=0, le=1)
    min_delay_minutes: float = Field(ge=0)
    mean_extra_delay_minutes: float = Field(ge=0)
    patience_multiplier: float = Field(gt=0)
    human_service_multiplier: float = Field(gt=0)
    human_requested_prob: float = Field(ge=0, le=1)
    min_remaining_deadline_minutes: float = Field(ge=0)


class TwinV11Config(_Frozen):
    """Twin V1.1 post-AI-failure consequences (SYNTHETIC_ASSUMPTION_V1_1). Defaults == exact V1 behaviour."""

    handoff_overhead_minutes: float = Field(default=0.0, ge=0)
    frustration_patience_multiplier: float = Field(default=1.0, gt=0, le=1)
    immediate_hangup_prob_after_ai_failure: float = Field(default=0.0, ge=0, le=1)

    @property
    def is_v1(self) -> bool:
        return (self.handoff_overhead_minutes == 0 and self.frustration_patience_multiplier == 1.0
                and self.immediate_hangup_prob_after_ai_failure == 0)


class AIDegradationConfig(_Frozen):
    """Scenario 5: partial AI degradation during the day (start drawn per day, hidden)."""

    start_min: float = Field(ge=0)
    start_max: float = Field(ge=0)
    duration_minutes: float = Field(gt=0)
    correct_multiplier: float = Field(ge=0, le=1)  # P(correct) x this; visible failure absorbs the rest
    service_time_multiplier: float = Field(ge=1)


class DemandShockConfig(_Frozen):
    """Scenario 7: sudden extra demand for one request type (start drawn per day, hidden)."""

    request_type: RequestType
    intensity_increase: float = Field(gt=0)  # extra rate = increase x type share x hourly profile
    start_min: float = Field(ge=0)
    start_max: float = Field(ge=0)
    duration_minutes: float = Field(gt=0)
    urgency_probs: dict[Urgency, float]

    @model_validator(mode="after")
    def _valid(self) -> DemandShockConfig:
        _check_distribution(self.urgency_probs, "shock urgency_probs")
        return self


class HumanShortageConfig(_Frozen):
    """Scenario 6: some advisors absent during a hidden window (non-preemptive: they leave after their call)."""

    start_min: float = Field(ge=0)
    start_max: float = Field(ge=0)
    duration_minutes: float = Field(gt=0)
    absent_advisors: int = Field(ge=1)


class ScenarioEventsConfig(_Frozen):
    """Predefined V1 scenario events (all absent / neutral by default)."""

    patience_scale: float = Field(default=1.0, gt=0)
    ai_degradation: AIDegradationConfig | None = None
    demand_shock: DemandShockConfig | None = None
    human_shortage: HumanShortageConfig | None = None


class RewardConfig(_Frozen):
    """V1 reward coefficients (frozen design decision, see README)."""

    resolution_multiplier: float
    human_busy_per_minute: float
    ai_busy_per_minute: float
    online_waiting_per_minute: float
    callback_pending_per_minute: float
    abandonment_multiplier: float
    missed_deadline_multiplier: float
    false_ai_extra_penalty: float
    callback_failure_penalty: float
    unfinished_multiplier: float


class SMDPConfig(_Frozen):
    gamma_per_minute: float = Field(gt=0, le=1)


class SimulationConfig(_Frozen):
    seed: int
    scenario: ScenarioConfig
    request_types: dict[RequestType, RequestTypeConfig]
    deadlines: dict[Urgency, Range]
    channels: dict[Channel, ChannelConfig]
    languages: dict[LanguageDialect, LanguageConfig]
    patience: PatienceConfig
    humans: HumanPoolConfig
    ai: AIConfig
    callback: CallbackConfig
    wait: WaitConfig
    mask: MaskConfig
    false_ai_discovery: FalseAIDiscoveryConfig
    recontact: RecontactConfig
    forecasting: ForecastingConfig
    intake_noise: IntakeNoiseConfig
    observation: ObservationConfig
    reward: RewardConfig
    smdp: SMDPConfig
    v1_1: TwinV11Config = Field(default_factory=TwinV11Config)
    events: ScenarioEventsConfig = Field(default_factory=ScenarioEventsConfig)

    @model_validator(mode="after")
    def _valid(self) -> SimulationConfig:
        _check_distribution({k: v.share for k, v in self.request_types.items()}, "request type shares")
        _check_distribution({k: v.share for k, v in self.channels.items()}, "channel shares")
        _check_distribution({k: v.share for k, v in self.languages.items()}, "language shares")
        floor = self.patience.floor_minutes
        for rtype, rcfg in self.request_types.items():
            for ch, chcfg in self.channels.items():
                mean = (rcfg.patience_mean_minutes * chcfg.patience_multiplier * min(1.0, self.recontact.patience_multiplier)
                        * self.events.patience_scale)
                if mean <= floor:
                    raise ValueError(f"patience mean {mean:.3f} for {rtype.value}/{ch.value} must exceed the floor {floor}")
        missing = set(Urgency) - set(self.deadlines)
        if missing:
            raise ValueError(f"deadlines missing for urgencies: {sorted(m.value for m in missing)}")
        return self
