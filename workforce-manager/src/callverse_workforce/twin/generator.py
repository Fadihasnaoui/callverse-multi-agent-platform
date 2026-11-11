"""Synthetic generation of a day's ground-truth contacts.

The whole day is generated up front from dedicated RNG streams, so the arrival
sequence and every contact's intrinsic attributes are independent of routing.
"""

from __future__ import annotations

import numpy as np

from callverse_workforce.config.models import ScenarioConfig, SimulationConfig
from callverse_workforce.domain.contact import GroundTruthContact
from callverse_workforce.twin.patience import sample_patience
from callverse_workforce.utils.distributions import categorical, lognormal_from_mean_cv
from callverse_workforce.utils.rng import RngManager, Stream


# Index ranges keep every contact's per-contact RNG key unique:
#   base day          0 .. 99_999
#   extra load layer l  LAYER_INDEX_BASE * (l + 1) + j
#   future variant k    FUTURE_INDEX_BASE * (k + 1) + j   (headroom counterfactual futures)
#   demand-shock contacts SHOCK_INDEX_BASE + j  (future variant k: FUTURE_INDEX_BASE * (k + 1) + SHOCK_INDEX_BASE + j)
LAYER_INDEX_BASE = 100_000
SHOCK_INDEX_BASE = 5_000_000
FUTURE_INDEX_BASE = 10_000_000


def scenario_window(rng: RngManager, stream: str, start_min: float, start_max: float, duration: float) -> tuple[float, float]:
    """Hidden day-level event window: start ~ U[start_min, start_max] from a dedicated stream."""
    start = start_min + (start_max - start_min) * float(rng.stream(stream).random())
    return start, start + duration


def contact_id_for(index: int) -> str:
    return f"C{index:05d}"


def generate_arrival_times(
    scenario: ScenarioConfig, rng: np.random.Generator, rate_multiplier: float = 1.0
) -> list[float]:
    """Non-homogeneous Poisson arrivals over the arrival window (thinning)."""
    rates = [r * rate_multiplier for r in scenario.hourly_arrival_rates]
    window = scenario.arrival_window_minutes
    if window <= 0:
        return []
    lam_max = max(rates[: int(np.ceil(window / 60.0))], default=0.0) / 60.0
    if lam_max <= 0:
        return []
    times: list[float] = []
    t = 0.0
    while True:
        t += float(rng.exponential(1.0 / lam_max))
        if t >= window:
            return times
        if rng.random() < (rates[int(t // 60)] / 60.0) / lam_max:
            times.append(t)


class ContactGenerator:
    def __init__(self, config: SimulationConfig, rng: RngManager) -> None:
        self.config = config
        self.rng = rng

    def generate_day(self) -> list[GroundTruthContact]:
        """Base day plus the nested extra-load layers (each from its own dedicated streams)."""
        contacts = self._poisson_contacts(Stream.ARRIVALS, Stream.REQUEST_TYPES, 1.0, 0, contact_id_for)
        for layer, fraction in enumerate(self.config.scenario.extra_load_layers):
            if fraction <= 0:
                continue
            contacts += self._poisson_contacts(
                f"{Stream.ARRIVALS}_load_layer_{layer}", f"{Stream.REQUEST_TYPES}_load_layer_{layer}",
                fraction, LAYER_INDEX_BASE * (layer + 1), contact_id_for,
            )
        contacts += self._shock_contacts("shock_arrivals", SHOCK_INDEX_BASE, lambda j: f"S{j:05d}")
        return contacts

    def shock_window(self) -> tuple[float, float] | None:
        shock = self.config.events.demand_shock
        if shock is None:
            return None
        return scenario_window(self.rng, Stream.DEMAND_SHOCK_TIMING, shock.start_min, shock.start_max,
                               shock.duration_minutes)

    def _shock_contacts(self, stream: str, index_base: int, make_id) -> list[GroundTruthContact]:
        """Scenario 7 extra arrivals of one request type inside the hidden shock window."""
        shock = self.config.events.demand_shock
        if shock is None:
            return []
        start, end = self.shock_window()  # type: ignore[misc]
        share = self.config.request_types[shock.request_type].share
        times = generate_arrival_times(self.config.scenario, self.rng.stream(stream), shock.intensity_increase * share)
        times = [t for t in times if start <= t < end]
        return [self._make_contact(index_base + j, t, shock.request_type, make_id(j), shock.urgency_probs)
                for j, t in enumerate(times)]

    def generate_future_variant(self, variant: int, after_time: float) -> list[GroundTruthContact]:
        """Alternative future arrivals (strictly after ``after_time``) for counterfactual future ``variant``.

        A whole variant day is drawn at the scenario's TOTAL rate (base + layers) from variant-keyed streams
        and only its part after ``after_time`` is kept, so every target state of a day shares the same
        variant-k future process, and all forced actions of a state share it too.
        """
        total = 1.0 + sum(self.config.scenario.extra_load_layers)
        contacts = self._poisson_contacts(
            f"future_arrivals_v{variant}", f"future_request_types_v{variant}", total,
            FUTURE_INDEX_BASE * (variant + 1), lambda idx: f"V{variant}C{idx % FUTURE_INDEX_BASE:05d}",
        )
        contacts += self._shock_contacts(f"future_shock_arrivals_v{variant}",
                                         FUTURE_INDEX_BASE * (variant + 1) + SHOCK_INDEX_BASE,
                                         lambda j: f"V{variant}S{j:05d}")
        return [c for c in contacts if c.arrival_time > after_time]

    def _poisson_contacts(self, arrival_stream, type_stream, multiplier, index_base, make_id) -> list[GroundTruthContact]:
        times = generate_arrival_times(self.config.scenario, self.rng.stream(arrival_stream), multiplier)
        type_rng = self.rng.stream(type_stream)
        shares = {k: v.share for k, v in self.config.request_types.items()}
        return [
            self._make_contact(index_base + i, t, categorical(type_rng, shares), make_id(index_base + i))
            for i, t in enumerate(times)
        ]

    def _make_contact(
        self, index: int, arrival_time: float, request_type, contact_id: str, urgency_probs=None
    ) -> GroundTruthContact:
        cfg = self.config
        rt = cfg.request_types[request_type]
        attr = self.rng.contact_stream(Stream.CONTACT_ATTRIBUTES, index)
        patience_rng = self.rng.contact_stream(Stream.PATIENCE, index)
        service_rng = self.rng.contact_stream(Stream.SERVICE_TIMES, index)
        false_rng = self.rng.contact_stream(Stream.FALSE_AI_DISCOVERY, index)

        urgency = categorical(attr, urgency_probs if urgency_probs is not None else rt.urgency_probs)
        sensitivity = categorical(attr, rt.sensitivity_probs)
        channel = categorical(attr, {k: v.share for k, v in cfg.channels.items()})
        language = categorical(attr, {k: v.share for k, v in cfg.languages.items()})
        request_value = float(attr.beta(rt.request_value_beta.a, rt.request_value_beta.b))
        deadline_range = cfg.deadlines[urgency]
        deadline_offset = float(attr.uniform(deadline_range.min, deadline_range.max))
        callback_allowed = bool(attr.random() < rt.callback_allowed_prob)
        human_requested = bool(attr.random() < rt.human_requested_prob)

        patience_mean = rt.patience_mean_minutes * cfg.channels[channel].patience_multiplier
        if cfg.events.patience_scale != 1.0:  # scenario 8 "more impatient customers"; V1 path untouched
            patience_mean *= cfg.events.patience_scale
        patience = sample_patience(patience_rng, patience_mean, cfg.patience)

        human_service = lognormal_from_mean_cv(service_rng, rt.human_service_minutes.mean, rt.human_service_minutes.cv)
        ai_service = lognormal_from_mean_cv(service_rng, rt.ai_service_minutes.mean, rt.ai_service_minutes.cv)

        ai_false = rt.ai_false_resolution_prob
        ai_correct = min(rt.ai_correct_prob * cfg.languages[language].ai_correct_multiplier, 1.0 - ai_false)

        discovery = cfg.false_ai_discovery
        discovery_delay = discovery.min_delay_minutes
        if discovery.mean_extra_delay_minutes > 0:
            discovery_delay += float(false_rng.exponential(discovery.mean_extra_delay_minutes))

        return GroundTruthContact(
            contact_id=contact_id,
            index=index,
            arrival_time=arrival_time,
            request_type=request_type,
            urgency=urgency,
            sensitivity=sensitivity,
            channel=channel,
            language_dialect=language,
            request_value=request_value,
            deadline_offset_minutes=deadline_offset,
            patience_minutes=patience,
            human_service_minutes=human_service,
            ai_service_minutes=ai_service,
            ai_correct_prob=ai_correct,
            ai_false_resolution_prob=ai_false,
            false_discovery_delay_minutes=discovery_delay,
            callback_allowed=callback_allowed,
            human_requested=human_requested,
        )


def recontact_id_for(root_contact_id: str, depth: int) -> str:
    return f"{root_contact_id}-R{depth}"


def draw_recontact(
    parent: GroundTruthContact,
    discovery_time: float,
    rng: np.random.Generator,
    config: SimulationConfig,
) -> tuple[bool, GroundTruthContact]:
    """Decide whether the customer returns and build the recontact's hidden truth.

    ``rng`` must be the parent's dedicated ``recontact`` stream. Every draw is
    made in a fixed order whether or not the recontact happens, so the result
    depends only on (seed, parent identity, discovery time) — never on the
    policy or on other contacts. The same customer problem is kept (type,
    channel, language, value, sensitivity, urgency, callback eligibility, AI
    capability); fresh latent draws are made for patience, service times and the
    discovery delay. SYNTHETIC_ASSUMPTION_V1 for all multipliers.
    """
    rc = config.recontact
    rt = config.request_types[parent.request_type]
    happens = bool(rng.random() < rc.probability)
    delay = rc.min_delay_minutes + (float(rng.exponential(rc.mean_extra_delay_minutes)) if rc.mean_extra_delay_minutes > 0 else 0.0)
    patience_mean = rt.patience_mean_minutes * config.channels[parent.channel].patience_multiplier * rc.patience_multiplier
    if config.events.patience_scale != 1.0:
        patience_mean *= config.events.patience_scale
    patience = sample_patience(rng, patience_mean, config.patience)
    human_service = lognormal_from_mean_cv(rng, rt.human_service_minutes.mean, rt.human_service_minutes.cv)
    ai_service = lognormal_from_mean_cv(rng, rt.ai_service_minutes.mean, rt.ai_service_minutes.cv)
    discovery = config.false_ai_discovery
    discovery_delay = discovery.min_delay_minutes + (
        float(rng.exponential(discovery.mean_extra_delay_minutes)) if discovery.mean_extra_delay_minutes > 0 else 0.0
    )
    human_requested = parent.human_requested or bool(rng.random() < rc.human_requested_prob)

    arrival = discovery_time + delay
    deadline_at = max(parent.deadline_at, arrival + rc.min_remaining_deadline_minutes)
    depth = parent.recontact_depth + 1
    recontact = GroundTruthContact(
        contact_id=recontact_id_for(parent.root_contact_id, depth),
        index=parent.index,
        arrival_time=arrival,
        request_type=parent.request_type,
        urgency=parent.urgency,
        sensitivity=parent.sensitivity,
        channel=parent.channel,
        language_dialect=parent.language_dialect,
        request_value=parent.request_value,
        deadline_offset_minutes=deadline_at - arrival,
        patience_minutes=patience,
        human_service_minutes=human_service * rc.human_service_multiplier,
        ai_service_minutes=ai_service,
        ai_correct_prob=parent.ai_correct_prob,
        ai_false_resolution_prob=parent.ai_false_resolution_prob,
        false_discovery_delay_minutes=discovery_delay,
        callback_allowed=parent.callback_allowed,
        human_requested=human_requested,
        root_contact_id=parent.root_contact_id,
        parent_contact_id=parent.contact_id,
        recontact_depth=depth,
    )
    return happens, recontact
