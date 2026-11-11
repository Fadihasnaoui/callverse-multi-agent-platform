"""Leakage-free hypothetical end-of-day branches for the V2 headroom protocol (spec §7, §7.0a, §7.1, §7.1a, §7.1b).

One branch = (V2Snapshot, evaluated action, hypothetical future k, continuation row, world model, A_age):
build a FRESH V2 twin from the observable snapshot only (Phase F world-model functions reused unchanged), restore every
in-system contact at its exact pending point (V2Twin.seed_contact), apply the evaluated action exactly once at the
target contact's POST_AI_FAILURE decision, then follow the continuation row for every later decision until the end
of the day (780); return the SMDP-discounted return (gamma 0.995/min) with the frozen end-of-day settlement
(headroom.branch.DiscountedReturn, unchanged). Restoration is identical for every action of a future.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from callverse_workforce.agent.types import DecisionType
from callverse_workforce.agent.whatif import (
    WHATIF_SEED_BASE,
    WhatIfPlan,
    _hold,
    _warm_indicators,
    hypothetical_config,
    world_parameters,
)
from callverse_workforce.config import SimulationConfig
from callverse_workforce.domain.contact import GroundTruthContact, ObservedContactFeatures
from callverse_workforce.domain.enums import Channel, LanguageDialect, RequestType, Sensitivity, Urgency
from callverse_workforce.headroom.branch import DiscountedReturn
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e.scenarios import MIXTURE_PROBABILITIES
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.twin.patience import sample_patience
from callverse_workforce.utils.distributions import lognormal_from_mean_cv
from callverse_workforce.utils.rng import RngManager
from callverse_workforce.v2_postfail.snapshot import BUSY_STATES, V2InSystem, V2Snapshot
from callverse_workforce.v2_postfail.twin import RestoreSpec, V2Environment, V2Twin

SEED_TAG = "v2hr"
SEEDED_INDEX_BASE = 9_000_000
MEAN_REVERSION_MINUTES = 60.0  # the unchanged Phase F definition (agent/config.py)
WORLD_MODELS = ("persistence", "mean_reversion")
RESTORE_EPS = 1e-6  # restored contacts start at now - RESTORE_EPS (advisor holds before queued contacts)
MAX_REJECTION = 50
# creation order at the restoration instant: contacts already holding an advisor first, then AI in service, then the
# rest by live state-entry time (FIFO positions)
ORDER = {"HUMAN_SERVICE": 0, "CALLBACK_ATTEMPT": 0, "CALLBACK_SERVICE": 0, "AI_SERVICE": 2}


def branch_seed(day_seed: int, state_index: int, k: int) -> int:
    return WHATIF_SEED_BASE + zlib.crc32(f"{SEED_TAG}|{day_seed}|{state_index}".encode()) * 64 + k


def _observed(c: V2InSystem) -> ObservedContactFeatures:
    o = dict(c.observed)
    return ObservedContactFeatures(
        contact_id=c.contact_id, observed_at=float(o["observed_at"]), request_type=RequestType(o["request_type"]),
        urgency_pred=Urgency(o["urgency_pred"]), sensitivity_pred=Sensitivity(o["sensitivity_pred"]), channel=Channel(o["channel"]),
        language_dialect=LanguageDialect(o["language_dialect"]), request_value=float(o["request_value"]),
        predicted_deadline_at=float(o["predicted_deadline_at"]), callback_allowed=bool(o["callback_allowed"]),
        human_requested=bool(o["human_requested"]), ai_resolution_score=float(o["ai_resolution_score"]))


def _conditional(draw, exceed: float) -> float:
    """A total duration re-drawn from its frozen distribution conditioned on exceeding the observed elapsed time
    (rejection, MAX_REJECTION tries; documented fallback: elapsed + 0.1 min)."""
    for _ in range(MAX_REJECTION):
        x = draw()
        if x > exceed:
            return x
    return exceed + 0.1


def _seed_truth(cfg: SimulationConfig, c: V2InSystem, j: int, now: float, rng: np.random.Generator) -> GroundTruthContact:
    """Hidden attributes re-drawn from the frozen generator distributions given OBSERVED attributes only:
    total patience conditioned on surviving the observed online wait; in-progress human / AI service conditioned on
    exceeding the observed elapsed service time; deadline = predicted deadline (observable)."""
    o = dict(c.observed)
    rtype, channel, lang = RequestType(o["request_type"]), Channel(o["channel"]), LanguageDialect(o["language_dialect"])
    rt = cfg.request_types[rtype]
    mean_p = rt.patience_mean_minutes * cfg.channels[channel].patience_multiplier
    patience = _conditional(lambda: sample_patience(rng, mean_p, cfg.patience), c.online_wait)
    h = cfg.v1_1.handoff_overhead_minutes if c.handoff_applied else 0.0
    draw_h = lambda: lognormal_from_mean_cv(rng, rt.human_service_minutes.mean, rt.human_service_minutes.cv)
    draw_a = lambda: lognormal_from_mean_cv(rng, rt.ai_service_minutes.mean, rt.ai_service_minutes.cv)
    human_service = _conditional(draw_h, c.elapsed_in_state - h) if c.state in ("HUMAN_SERVICE", "CALLBACK_SERVICE") else draw_h()
    ai_service = _conditional(draw_a, c.elapsed_in_state) if c.state == "AI_SERVICE" else draw_a()
    ai_false = rt.ai_false_resolution_prob
    ai_correct = min(rt.ai_correct_prob * cfg.languages[lang].ai_correct_multiplier, 1.0 - ai_false)
    disc = cfg.false_ai_discovery
    delay = disc.min_delay_minutes + (float(rng.exponential(disc.mean_extra_delay_minutes)) if disc.mean_extra_delay_minutes > 0 else 0.0)
    return GroundTruthContact(
        contact_id=f"H{j:05d}", index=SEEDED_INDEX_BASE + j, arrival_time=max(0.0, now - RESTORE_EPS), request_type=rtype,
        urgency=Urgency(o["urgency_pred"]), sensitivity=Sensitivity(o["sensitivity_pred"]), channel=channel, language_dialect=lang,
        request_value=float(o["request_value"]), deadline_offset_minutes=max(0.5, float(o["predicted_deadline_at"]) - now + RESTORE_EPS),
        patience_minutes=patience, human_service_minutes=human_service, ai_service_minutes=ai_service, ai_correct_prob=ai_correct,
        ai_false_resolution_prob=ai_false, false_discovery_delay_minutes=delay, callback_allowed=bool(o["callback_allowed"]),
        human_requested=bool(o["human_requested"]))


def restore_spec(c: V2InSystem) -> RestoreSpec:
    return RestoreSpec(state=c.state, entered_at=c.entered_at, elapsed_in_state=c.elapsed_in_state, escalated=c.escalated,
                       wait_count=c.wait_count, hq_prev=c.hq_prev, queue_class=c.queue_class, routing_attempt_count=c.routing_attempt_count,
                       online_wait=c.online_wait, handoff_applied=c.handoff_applied, decision_type=DecisionType(c.decision_type),
                       callback_started_at=c.callback_started_at, callback_attempts=c.callback_attempts,
                       callback_attempt_times=tuple(c.callback_attempt_times))


def plan_for(snap: V2Snapshot, base: SimulationConfig, world_model: str, f_ref: float) -> WhatIfPlan:
    if world_model not in WORLD_MODELS:
        raise ValueError(world_model)
    return WhatIfPlan(approval_delay=0.0, duration=0.0, horizon=base.scenario.horizon_minutes - snap.now, world_model=world_model,
                      mean_reversion_minutes=MEAN_REVERSION_MINUTES, f_ref=f_ref)


def build_world(snap: V2Snapshot, cfg: SimulationConfig, params: dict[str, Any], seed: int):
    """Future arrivals (new streams) + restored in-system contacts + unexplained-occupancy holds. Identical for every
    action of a future."""
    now, end = snap.now, params["end"]
    future = [c for c in ContactGenerator(cfg, RngManager(seed)).generate_day() if now < c.arrival_time < end]
    restored = []
    for j, c in enumerate(snap.in_system):
        rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0xF3, j)))
        restored.append((c, _seed_truth(cfg, c, j, now, rng)))
    busy = sum(1 for c in snap.in_system if c.state in BUSY_STATES)
    occupied = int(round(float(snap.center_value("human_utilization")) * snap.public.rostered_advisors))
    holds = [end - now] * max(0, occupied - busy)  # occupancy not explained by a visible contact (R1: until closing)
    restored.sort(key=lambda ct: (ORDER.get(ct[0].state, 3), ct[0].entered_at, ct[0].contact_id))
    return future, restored, holds


def restore_world(snap: V2Snapshot, base: SimulationConfig, seed: int, world_model: str, a_age: float, f_ref: float,
                  cfg_cache: dict | None = None):
    """Fresh hypothetical V2 twin with every in-system contact restored; returns (twin, cfg, cell, live->restored id map)."""
    plan = plan_for(snap, base, world_model, f_ref)
    params = world_parameters(snap, base, plan)
    rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0xF1,)))
    cell = MIXTURE_CELLS[int(rng.choice(len(MIXTURE_CELLS), p=np.array(MIXTURE_PROBABILITIES)))]
    key = (snap.sha256(), world_model, f_ref, cell.cell_id)
    cache = cfg_cache if cfg_cache is not None else {}
    if key not in cache:
        cache[key] = hypothetical_config(snap, base, params, cell)
    cfg = cache[key]
    future, restored, holds = build_world(snap, cfg, params, seed)
    twin = V2Twin(cfg, seed, contacts=future, a_age=a_age)
    _warm_indicators(twin, snap)
    start = max(0.0, snap.now - RESTORE_EPS)
    busy = [(c, t) for c, t in restored if c.state in BUSY_STATES]
    rest = [(c, t) for c, t in restored if c.state not in BUSY_STATES]
    for c, t in busy:  # advisors already held by visible contacts
        twin.seed_contact(t, _observed(c), restore_spec(c), start)
    for dur in holds:
        twin.env.process(_hold(twin, start, dur))
    for c, t in rest:
        twin.seed_contact(t, _observed(c), restore_spec(c), start)
    twin.env.run(until=max(0.0, start - RESTORE_EPS))
    twin.accrue()
    return twin, cfg, cell, {c.contact_id: t.contact_id for c, t in restored}


def run_branch(snap: V2Snapshot, base: SimulationConfig, action: str, seed: int, continuation, world_model: str,
               a_age: float, f_ref: float, cfg_cache: dict | None = None) -> dict[str, Any]:
    twin, cfg, cell, id_map = restore_world(snap, base, seed, world_model, a_age, f_ref, cfg_cache)
    tracker = DiscountedReturn(lambda: twin.now, twin.now, cfg.smdp.gamma_per_minute)
    twin.reward.observer = tracker
    env = V2Environment(twin)
    target_id = id_map[snap.target_contact_id]
    dp = env.next_decision()
    forced, n_after = False, 0
    while dp is not None:
        if not forced and dp.contact_id == target_id and dp.decision_type is DecisionType.POST_AI_FAILURE:
            if not dp.mask.is_valid(action):
                return {"status": "INVALID_FORCED_ACTION", "action": action}
            a, forced = action, True
        else:
            a = continuation.act(dp)
            n_after += 1
        dp = env.step(a)
    if not forced:
        return {"status": "TARGET_NOT_REACHED", "action": action}
    s = env.summary()
    target_actions = [e.data["action"] for e in twin.events.records
                      if e.event_type.value == "ROUTING_DECISION" and e.contact_id == target_id]
    return {"status": "OK", "action": action, "target_actions": target_actions, "seed": seed, "cell": cell.cell_id,
            "continuation": list(continuation.spec), "world_model": world_model, "a_age": a_age, "discounted_return": tracker.total,
            "undiscounted_return": sum(tracker.undiscounted.values()), "decisions_after_target": n_after,
            "aging_violations": s["aging_violations"], "aging_overrides": s["aging_overrides"], "classes": s["classes"],
            "horizon": twin.horizon, "finalized": twin.finalized}
