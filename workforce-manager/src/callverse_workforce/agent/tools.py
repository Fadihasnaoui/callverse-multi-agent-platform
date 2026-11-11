"""ToolRegistry: real runtime tools whose outputs drive the supervision control flow (every call is logged)."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Callable

import numpy as np

from callverse_workforce.agent.memory import MemoryStore
from callverse_workforce.agent.types import DecisionType, PublicDayContext
from callverse_workforce.agent.whatif import InSystemContact, ObservableSnapshot, WhatIfPlan, run_twin_whatif
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.forecasting.forecaster import ReferenceProfile
from callverse_workforce.routing.mask import ServiceAvailability, compute_action_mask

CENTER_FIELDS = ("humans_available", "human_utilization", "human_queue_size", "oldest_human_wait", "ai_load",
                 "recent_ai_failure_rate", "wait_queue_size", "callbacks_pending", "oldest_callback_age",
                 "recent_arrival_rate", "forecast_30min", "recent_abandonment_rate", "recent_service_level", "hour_sin", "hour_cos")


def _h(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def queue_growth(samples, now: float, window: float) -> float:
    """OLS slope of human_queue_size over the window, in contacts per 15 minutes."""
    pts = [(t, q) for t, q in samples if t >= now - window]
    if len(pts) < 3:
        return 0.0
    t = np.array([p[0] for p in pts])
    q = np.array([p[1] for p in pts], dtype=float)
    if np.ptp(t) <= 0:
        return 0.0
    return float(np.polyfit(t, q, 1)[0] * 15.0)


class MaskProvider:
    """Version-aware masks. V1_1/INITIAL = exactly the frozen routing.mask.compute_action_mask."""

    def __init__(self, base_config) -> None:
        self.cfg = base_config

    def get(self, policy_version: str, decision_type: DecisionType, observation) -> ActionMask:
        if policy_version != "V1_1" or decision_type is not DecisionType.INITIAL:
            raise NotImplementedError("only V1_1 / INITIAL masks exist in Phase F")
        cb = self.cfg.callback
        avail = ServiceAvailability(ai_available=self.cfg.ai.available,
                                    callback_available=cb.available and observation.callbacks_pending < cb.max_pending)
        return compute_action_mask(observation, avail, self.cfg.mask, self.cfg.wait)


class ToolRegistry:
    def __init__(self, memory: MemoryStore, public: PublicDayContext, base_config, cfg, status_fn: Callable[[], dict],
                 masks: MaskProvider) -> None:
        self.memory = memory
        self.public = public
        self.base = base_config
        self.cfg = cfg
        self.status_fn = status_fn
        self.masks = masks
        self.calls: list[dict[str, Any]] = []

    def _log(self, name: str, t: float, inputs: Any, output: Any, t0: float) -> None:
        rec = {"tool": name, "t": t, "input_hash": _h(inputs), "output_hash": _h(output),
               "duration_ms": (time.perf_counter() - t0) * 1000}
        self.calls.append(rec)
        self.memory.record("ToolCall", t, **{k: v for k, v in rec.items() if k != "t"})

    # --- tools ----------------------------------------------------------------------------------------
    def get_kpis(self, now: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        m, w = self.memory, self.cfg.kpi_window_minutes
        since = now - w
        done = m.count({"AI_APPARENT_RESOLUTION", "AI_VISIBLE_FAILURE"}, since)
        fail = m.count({"AI_VISIBLE_FAILURE", "FALSE_AI_DISCOVERED"}, since)
        arr = m.count({"ARRIVAL"}, since)
        ab = m.count({"ABANDONMENT"}, since)
        obs = m.last_percept.observation if m.last_percept else None
        k = {"t": now, "observation_t": m.last_percept.timestamp if m.last_percept else None,
             "ai_failure_rate_30": min(1.0, fail / done) if done else 0.0, "n_ai_30": done,
             "abandonment_rate_30": ab / arr if arr else 0.0, "n_arrivals_30": arr,
             "human_queue_growth": queue_growth(m.queue_samples, now, w)}
        k |= {f: (float(getattr(obs, f)) if obs is not None else 0.0) for f in CENTER_FIELDS}
        k["human_queue_size"] = int(k["human_queue_size"])
        k["ppo_health"] = self.status_fn().get("ppo_health", {})
        m.kpi_snapshots.append(k)
        self._log("get_kpis", now, {"now": now}, k, t0)
        return k

    def forecast(self, now: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        obs = self.memory.last_percept.observation if self.memory.last_percept else None
        f30 = float(obs.forecast_30min) if obs else 0.0
        ref = ReferenceProfile(self.public.reference_hourly_rates)
        ref30 = ref.expected(now, now + 30.0)
        corr = f30 / ref30 if ref30 > 0 else 1.0
        out = {"forecast_30min": f30, "reference_30min": ref30, "correction": corr,
               "expected_next_120min": ref.expected(now, now + 120.0) * corr,
               "recent_arrival_rate": float(obs.recent_arrival_rate) if obs else 0.0}
        self._log("forecast", now, {"now": now}, out, t0)
        return out

    def get_recent_context(self, now: float, window: float = 30.0) -> dict[str, Any]:
        t0 = time.perf_counter()
        m = self.memory
        dec = [d for d in m.decisions if d["t"] >= now - window]
        shares: dict[str, float] = {}
        for d in dec:
            shares[d["action"]] = shares.get(d["action"], 0) + 1
        out = {"decisions": len(dec), "action_share": {a: c / len(dec) for a, c in shares.items()} if dec else {},
               "ai_visible_failures": m.count({"AI_VISIBLE_FAILURE"}, now - window),
               "false_ai_discovered": m.count({"FALSE_AI_DISCOVERED"}, now - window),
               "abandonments": m.count({"ABANDONMENT"}, now - window),
               "queue_trajectory": [(round(t, 2), q) for t, q in m.queue_samples if t >= now - window][-20:],
               "open_contacts": len(m.open),
               "last_mode_changes": [r["to"] for r in m.records("ModeTransition")[-3:]]}
        self._log("get_recent_context", now, {"now": now, "window": window}, out, t0)
        return out

    def get_action_mask(self, policy_version: str, decision_type: DecisionType, observation) -> ActionMask:
        return self.masks.get(policy_version, decision_type, observation)

    def get_agent_status(self) -> dict[str, Any]:
        return self.status_fn()

    def observable_snapshot(self, now: float, kpis: dict[str, Any]) -> ObservableSnapshot:
        """Build the planner input from memory + public context ONLY."""
        m = self.memory
        obs = m.last_percept.observation
        center = tuple((f, float(getattr(obs, f))) for f in CENTER_FIELDS)
        in_sys = tuple(
            InSystemContact(c.contact_id, c.state, tuple(sorted(c.observed.items())), now - c.since, c.waited(now),
                            now - c.decided_at, c.escalated)
            for c in sorted(m.open.values(), key=lambda c: c.contact_id))
        mix: dict[str, int] = {}
        for d in m.decisions:  # observed (predicted) request types routed over the last 60 minutes
            if d["t"] >= now - 60.0:
                mix[d["request_type"]] = mix.get(d["request_type"], 0) + 1
        hist = tuple((e.timestamp, e.event_type, float(e.data.get("online_wait", 0.0)))
                     for e in m.events if e.timestamp >= now - 60.0)
        return ObservableSnapshot(now, self.public, center, in_sys, kpis["ai_failure_rate_30"], kpis["n_ai_30"],
                                  tuple(sorted(mix.items())), hist, tuple(m.arrival_times))

    def run_twin_whatif(self, snapshot: ObservableSnapshot, current_policy, candidate_policy, plan: WhatIfPlan,
                        seeds, state_builder, whatif_id: str):
        t0 = time.perf_counter()
        res = run_twin_whatif(snapshot, self.base, current_policy, candidate_policy, plan, seeds, state_builder, whatif_id)
        self._log("run_twin_whatif", snapshot.now, {"snapshot": res.snapshot_sha256, "seeds": list(seeds)},
                  {"gain": res.estimated_gain}, t0)
        return res
