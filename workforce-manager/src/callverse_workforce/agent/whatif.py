"""Leakage-free Digital-Twin what-if planning (docs/phase_f_spec.md §8).

``run_twin_whatif`` receives ONLY an ``ObservableSnapshot`` (built from percepts, observable events and public
pre-day context). It builds FRESH hypothetical twins with the unchanged V1.1 Twin code, from NEW random streams
(seeds >= 2**40, disjoint from every day seed). Nothing of the live day's hidden state or future is reachable:
the live twin is never passed in, and the hypothetical world is driven only by observable quantities and
explicit, documented world-model assumptions:

* the scenario mixture cell is drawn from its PUBLIC prior (0.25 each), per hypothetical future;
* arrivals: the calendar reference profile scaled by forecast_30min / reference expectation (clip [0.5, 3]);
* request mix: historical prior blended with the request types observed over the last 60 minutes;
* AI condition: m_hat = clip((1 - f_obs) / (1 - f_ref), m_lo, 1) from the observed 30-min AI failure rate;
  PRIMARY world model "persistence": the observed condition persists over the whole horizon;
  DEV-only sensitivity "mean_reversion": it persists for ``mean_reversion_minutes`` only;
* contacts already in the system: re-created from their OBSERVED attributes, hidden attributes re-drawn from the
  frozen generator distributions (patience conditioned on the observed waiting time);
* occupied human slots (observable utilization x roster): contacts in service become holds with residual service
  drawn conditionally on the observed elapsed time; occupancy not explained by visible service is held for the
  whole horizon (persistence).

Branch A = current routing policy (PPO) for the whole horizon. Branch B = PPO until approval
(now + approval_delay), then RulePolicy for D minutes, then PPO. Both branches of future k share the SAME new
hypothetical exogenous future (common random numbers).
"""

from __future__ import annotations

import hashlib
import json
import math
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from callverse_workforce.agent.state_builder import StateBuilder
from callverse_workforce.agent.types import DecisionType, Percept, PublicDayContext
from callverse_workforce.config import SimulationConfig, apply_overrides
from callverse_workforce.domain.contact import GroundTruthContact
from callverse_workforce.domain.enums import Channel, LanguageDialect, Outcome, RequestType, RoutingAction, Sensitivity, Urgency
from callverse_workforce.forecasting.forecaster import ReferenceProfile
from callverse_workforce.phase_d.grid import MIXTURE_CELLS, cell_config
from callverse_workforce.phase_e.scenarios import MIXTURE_PROBABILITIES
from callverse_workforce.twin.digital_twin import DigitalTwin
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.twin.patience import sample_patience
from callverse_workforce.utils.distributions import lognormal_from_mean_cv
from callverse_workforce.utils.rng import RngManager

WHATIF_SEED_BASE = 2**40
SYNTHETIC_INDEX_BASE = 9_000_000
T90 = {1: 3.078, 3: 1.638, 7: 1.415, 15: 1.341, 31: 1.309, 63: 1.295}  # one-sided 0.90 Student-t quantiles
HOLD_EPS = 1e-6
FORCED_BY_STATE = {"AI_QUEUE": "AI", "AI_SERVICE": "AI", "HUMAN_QUEUE": "HUMAN", "WAITING": "WAIT", "CALLBACK_QUEUE": "CALLBACK"}
HUMAN_BUSY_STATES = ("HUMAN_SERVICE", "CALLBACK_ATTEMPT", "CALLBACK_SERVICE")


def whatif_seed(day_seed: int, cycle_index: int, k: int) -> int:
    """New hypothetical streams: >= 2**40, keyed by (day, supervision cycle, future k) — never a day seed."""
    return WHATIF_SEED_BASE + zlib.crc32(f"{day_seed}|{cycle_index}".encode()) * 64 + k


# ---------------------------------------------------------------------------------------------------------
# observable snapshot (the ONLY input of the planner)
# ---------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class InSystemContact:
    contact_id: str
    state: str
    observed: tuple[tuple[str, Any], ...]
    elapsed_in_state: float
    waited: float
    since_decision: float
    escalated: bool


@dataclass(frozen=True)
class ObservableSnapshot:
    now: float
    public: PublicDayContext
    center: tuple[tuple[str, Any], ...]  # last observed centre-state features
    in_system: tuple[InSystemContact, ...]
    ai_failure_rate_30: float
    n_ai_30: int
    request_mix_60: tuple[tuple[str, int], ...]
    history_60: tuple[tuple[float, str, float], ...]  # (t, observable event type, online_wait) for indicator warm-up
    arrival_times: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["public"] = self.public.to_dict()
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode()).hexdigest()

    def center_value(self, key: str) -> Any:
        return dict(self.center)[key]


@dataclass(frozen=True)
class WhatIfPlan:
    """Exactly the operational proposal: PPO until approval, candidate for D minutes, then PPO."""

    approval_delay: float
    duration: float
    horizon: float = 120.0
    world_model: str = "persistence"
    mean_reversion_minutes: float = 60.0
    f_ref: float = 0.15
    m_lo: float = 0.3
    n_min_ai: int = 10


@dataclass
class WhatIfResult:
    whatif_id: str
    status: str
    k: int
    seeds: list[int]
    plan: dict[str, Any]
    snapshot_sha256: str
    world: dict[str, Any]
    branch_a: dict[str, Any]
    branch_b: dict[str, Any]
    diffs: list[float]
    estimated_gain: float
    diff_sd: float
    t80_lower: float
    share_positive: float
    per_future: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------
# hypothetical world construction (observable snapshot + public base config only)
# ---------------------------------------------------------------------------------------------------------
def ai_multiplier(snapshot: ObservableSnapshot, plan: WhatIfPlan) -> float:
    if snapshot.n_ai_30 < plan.n_min_ai:
        return 1.0
    m = (1.0 - snapshot.ai_failure_rate_30) / max(1e-9, 1.0 - plan.f_ref)
    return float(min(1.0, max(plan.m_lo, m)))


def world_parameters(snapshot: ObservableSnapshot, base: SimulationConfig, plan: WhatIfPlan) -> dict[str, Any]:
    now = snapshot.now
    ref = ReferenceProfile(snapshot.public.reference_hourly_rates)
    ref30 = ref.expected(now, now + 30.0)
    f30 = float(snapshot.center_value("forecast_30min"))
    scale = float(min(3.0, max(0.5, f30 / ref30))) if ref30 > 0 else 1.0
    prior = dict(snapshot.public.request_type_prior)
    counts = dict(snapshot.request_mix_60)
    n = sum(counts.values())
    mix = {k: (counts.get(k, 0) + 20.0 * p) / (n + 20.0) for k, p in prior.items()}
    end = min(now + plan.horizon, base.scenario.horizon_minutes)
    return {"arrival_scale": scale, "mix": mix, "ai_multiplier": ai_multiplier(snapshot, plan), "end": end,
            "degradation_minutes": plan.horizon if plan.world_model == "persistence" else plan.mean_reversion_minutes}


def hypothetical_config(snapshot: ObservableSnapshot, base: SimulationConfig, params: dict[str, Any], cell) -> SimulationConfig:
    now, end = snapshot.now, params["end"]
    window = min(base.scenario.arrival_window_minutes, end)
    o: dict[str, Any] = {
        "scenario.hourly_arrival_rates": [r * params["arrival_scale"] for r in snapshot.public.reference_hourly_rates],
        "scenario.extra_load_layers": [],
        "scenario.day_type": snapshot.public.day_type,
        "scenario.arrival_window_minutes": window,
        "scenario.drain_minutes": max(0.0, end - window),
        "forecasting.reference_hourly_arrival_rates": list(snapshot.public.reference_hourly_rates),
        "humans.n_advisors": snapshot.public.rostered_advisors,
        "events.patience_scale": 1.0, "events.human_shortage": None, "events.demand_shock": None,
        "events.ai_degradation": None,
    }
    o |= {f"request_types.{k}.share": v for k, v in params["mix"].items()}
    m = params["ai_multiplier"]
    if m < 1.0:
        o["events.ai_degradation"] = {"start_min": now, "start_max": now, "duration_minutes": params["degradation_minutes"],
                                      "correct_multiplier": m, "service_time_multiplier": 1.0}
    return cell_config(apply_overrides(base, o), cell)


def _synthetic_contact(cfg: SimulationConfig, c: InSystemContact, j: int, now: float, rng: np.random.Generator) -> GroundTruthContact:
    obs = dict(c.observed)
    rtype = RequestType(obs["request_type"])
    rt = cfg.request_types[rtype]
    channel, lang = Channel(obs["channel"]), LanguageDialect(obs["language_dialect"])
    mean_p = rt.patience_mean_minutes * cfg.channels[channel].patience_multiplier
    residual = None
    for _ in range(50):  # patience conditioned on survival beyond the observed waiting time
        p = sample_patience(rng, mean_p, cfg.patience)
        if p > c.waited:
            residual = p - c.waited
            break
    patience = residual if residual is not None else 1.0
    human_service = lognormal_from_mean_cv(rng, rt.human_service_minutes.mean, rt.human_service_minutes.cv)
    ai_service = lognormal_from_mean_cv(rng, rt.ai_service_minutes.mean, rt.ai_service_minutes.cv)
    ai_false = rt.ai_false_resolution_prob
    ai_correct = min(rt.ai_correct_prob * cfg.languages[lang].ai_correct_multiplier, 1.0 - ai_false)
    disc = cfg.false_ai_discovery
    delay = disc.min_delay_minutes + (float(rng.exponential(disc.mean_extra_delay_minutes)) if disc.mean_extra_delay_minutes > 0 else 0.0)
    deadline = max(0.5, float(obs["deadline_remaining_pred"]) - c.since_decision)
    return GroundTruthContact(
        contact_id=f"H{j:05d}", index=SYNTHETIC_INDEX_BASE + j, arrival_time=now, request_type=rtype,
        urgency=Urgency(obs["urgency_pred"]), sensitivity=Sensitivity(obs["sensitivity_pred"]), channel=channel,
        language_dialect=lang, request_value=float(obs["request_value"]), deadline_offset_minutes=deadline,
        patience_minutes=patience, human_service_minutes=human_service, ai_service_minutes=ai_service,
        ai_correct_prob=ai_correct, ai_false_resolution_prob=ai_false, false_discovery_delay_minutes=delay,
        callback_allowed=bool(obs["callback_allowed"]), human_requested=bool(obs["human_requested"]))


def _residual_service(cfg: SimulationConfig, c: InSystemContact, rng: np.random.Generator) -> float:
    if c.state == "CALLBACK_ATTEMPT":
        return max(0.1, cfg.callback.attempt_minutes - c.elapsed_in_state)
    rt = cfg.request_types[RequestType(dict(c.observed)["request_type"])]
    for _ in range(50):
        s = lognormal_from_mean_cv(rng, rt.human_service_minutes.mean, rt.human_service_minutes.cv)
        if s > c.elapsed_in_state:
            return s - c.elapsed_in_state
    return max(0.1, 0.5 * rt.human_service_minutes.mean)


def build_world(snapshot: ObservableSnapshot, cfg: SimulationConfig, params: dict[str, Any], seed: int):
    """(contacts, holds, forced first actions) of one hypothetical future — identical for both branches."""
    now, end = snapshot.now, params["end"]
    future = [c for c in ContactGenerator(cfg, RngManager(seed)).generate_day() if now < c.arrival_time < end]
    contacts, forced, holds = [], {}, []
    busy = 0
    for j, c in enumerate(snapshot.in_system):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0xF2, j)))
        if c.state in HUMAN_BUSY_STATES:
            holds.append(_residual_service(cfg, c, rng))
            busy += 1
        elif c.state in FORCED_BY_STATE:
            gt = _synthetic_contact(cfg, c, j, now, rng)
            contacts.append(gt)
            forced[gt.contact_id] = FORCED_BY_STATE[c.state]
    occupied = int(round(float(snapshot.center_value("human_utilization")) * snapshot.public.rostered_advisors))
    holds += [params["end"] - now] * max(0, occupied - busy)  # unexplained occupancy persists over the horizon
    return contacts + future, holds, forced


# ---------------------------------------------------------------------------------------------------------
# branch simulation
# ---------------------------------------------------------------------------------------------------------
def _hold(twin: DigitalTwin, start: float, duration: float):
    yield twin.env.timeout(start)
    req = twin.humans.request(priority=-1)
    yield req
    yield twin.env.timeout(duration)
    twin._release_human(req)


def _warm_indicators(twin: DigitalTwin, snapshot: ObservableSnapshot) -> None:
    ind = twin.indicators
    for t, et, ow in snapshot.history_60:
        if et == "ARRIVAL":
            ind.record_arrival(t)
        elif et in ("AI_APPARENT_RESOLUTION", "AI_VISIBLE_FAILURE"):
            ind.record_ai_completion(t)
        if et in ("AI_VISIBLE_FAILURE", "FALSE_AI_DISCOVERED"):
            ind.record_ai_failure_observed(t)
        elif et == "ABANDONMENT":
            ind.record_abandonment(t)
        elif et in ("HUMAN_SERVICE_STARTED", "AI_ATTEMPT_STARTED"):
            ind.record_answered(t, ow)
    twin._observed_arrivals.extend(snapshot.arrival_times)


def simulate_branch(snapshot, cfg, contacts, holds, forced, seed, segments, default_policy, sb: StateBuilder) -> dict[str, Any]:
    now = snapshot.now
    twin = DigitalTwin(cfg, seed, contacts=contacts)
    _warm_indicators(twin, snapshot)
    for dur in holds:
        twin.env.process(_hold(twin, max(0.0, now - HOLD_EPS), dur))
    n_rule = n_dec = 0
    while True:
        rt = twin.advance_to_next_decision()
        if rt is None:
            break
        obs, mask = twin.observe(rt)
        cid = rt.contact_id
        if cid in forced and rt.routing_attempt_count == 0:
            a = forced[cid]
        else:
            t = twin.now
            pol = next((p for t0, t1, p in segments if t0 <= t < t1), default_policy)
            n_rule += pol is not default_policy
            n_dec += 1
            pct = Percept(f"W{n_dec}", cid, t, DecisionType.INITIAL, obs, mask)
            a = pol.predict(sb.build(pct), obs, mask)
        if not mask.valid[RoutingAction(a).index]:
            a = RoutingAction.HUMAN.value
        twin.apply_decision(twin.new_decision_id(), RoutingAction(a), mask)
    twin.finalize()
    rew = twin.reward.episode.to_dict()
    rts = list(twin.contacts.values())
    n = len(rts) or 1
    busy = sum(r.human_direct_minutes + r.callback_human_minutes for r in rts) + sum(holds)
    span = max(1e-9, (twin.horizon - now) * snapshot.public.rostered_advisors)
    q_end = sum(1 for e in twin.events.records
                if e.event_type.value == "UNFINISHED_AT_HORIZON" and getattr(e.data.get("state"), "value", "") == "HUMAN_QUEUE")
    return {"reward": rew["total"], "reward_breakdown": rew,
            "resolution_rate": sum(1 for r in rts if r.outcome in (Outcome.RESOLVED_AI, Outcome.RESOLVED_HUMAN, Outcome.RESOLVED_CALLBACK)) / n,
            "abandonment_rate": sum(1 for r in rts if r.outcome is Outcome.ABANDONED) / n,
            "human_utilization": min(1.0, busy / span), "wait_mean": float(np.mean([r.online_wait_accumulated for r in rts])) if rts else 0.0,
            "false_ai_cost": rew.get("false_ai", 0.0), "queue_at_horizon": q_end, "contacts": len(rts),
            "decisions_by_candidate": n_rule, "decisions": n_dec}


def _summ(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = ("reward", "resolution_rate", "abandonment_rate", "human_utilization", "wait_mean", "false_ai_cost", "queue_at_horizon")
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0}
            for k in keys}


def run_twin_whatif(snapshot: ObservableSnapshot, base: SimulationConfig, current_policy, candidate_policy,
                    plan: WhatIfPlan, seeds: Sequence[int], state_builder: StateBuilder | None = None,
                    whatif_id: str = "") -> WhatIfResult:
    """Sandboxed planning from an ObservableSnapshot only (no live twin, no hidden state, new RNG)."""
    if not isinstance(snapshot, ObservableSnapshot):
        raise TypeError("run_twin_whatif accepts an ObservableSnapshot only")
    if any(s < WHATIF_SEED_BASE for s in seeds):
        raise ValueError("what-if seeds must be >= 2**40 (disjoint from day seeds)")
    sb = state_builder or StateBuilder()
    params = world_parameters(snapshot, base, plan)
    t_switch = snapshot.now + plan.approval_delay
    segments_b = [(t_switch, t_switch + plan.duration, candidate_policy)]
    cfg_cache: dict[str, SimulationConfig] = {}
    rows_a, rows_b, per_future = [], [], []
    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0xF1,)))
        cell = MIXTURE_CELLS[int(rng.choice(len(MIXTURE_CELLS), p=np.array(MIXTURE_PROBABILITIES)))]
        if cell.cell_id not in cfg_cache:
            cfg_cache[cell.cell_id] = hypothetical_config(snapshot, base, params, cell)
        cfg = cfg_cache[cell.cell_id]
        contacts, holds, forced = build_world(snapshot, cfg, params, seed)
        a = simulate_branch(snapshot, cfg, contacts, holds, forced, seed, [], current_policy, sb)
        b = simulate_branch(snapshot, cfg, contacts, holds, forced, seed, segments_b, current_policy, sb)
        rows_a.append(a)
        rows_b.append(b)
        per_future.append({"k": k, "seed": seed, "cell": cell.cell_id, "J_A": a["reward"], "J_B": b["reward"],
                           "n_contacts": a["contacts"], "n_holds": len(holds), "rule_decisions": b["decisions_by_candidate"]})
    diffs = [b["reward"] - a["reward"] for a, b in zip(rows_a, rows_b)]
    d = np.asarray(diffs)
    sd = float(d.std(ddof=1)) if d.size > 1 else 0.0
    tq = T90.get(d.size - 1, 1.282)
    return WhatIfResult(
        whatif_id=whatif_id, status="OK", k=len(seeds), seeds=list(seeds), plan=asdict(plan), snapshot_sha256=snapshot.sha256(),
        world={k: v for k, v in params.items() if k != "end"} | {"horizon_end": params["end"]},
        branch_a=_summ(rows_a), branch_b=_summ(rows_b), diffs=diffs, estimated_gain=float(d.mean()), diff_sd=sd,
        t80_lower=float(d.mean() - tq * sd / math.sqrt(d.size)), share_positive=float((d > 0).mean()), per_future=per_future)
