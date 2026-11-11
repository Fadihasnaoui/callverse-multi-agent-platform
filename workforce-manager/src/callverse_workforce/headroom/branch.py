"""Forced-action counterfactual branch replay.

One branch = (day seed, target decision index, forced action a, future variant k, continuation policy):

1. Replay the day from t=0 with the frozen pre-target policy (RulePolicy) up to the target decision.
2. Verify the reconstructed decision (id, contact, time, observation, mask) against the sampled record.
   Any mismatch FAILS the branch (it is never continued from a different state).
3. Branch point: ``DigitalTwin.activate_future_variant(k, variant-k future arrivals)`` -- the history is
   fixed, everything random after the target decision comes from variant-k streams that do not depend on
   the forced action (common random numbers across actions).
4. Apply the forced action exactly once, then follow the continuation policy until the end of the day.
5. Return the exact SMDP-discounted return from the target time: every event reward is discounted by
   ``gamma ** (t_event - t_target)`` and time-integrated costs are integrated analytically over each
   interval (``integral of gamma ** (s - t0) ds``).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from callverse_workforce.baselines.policies import make_policy
from callverse_workforce.config.models import SimulationConfig
from callverse_workforce.domain.enums import RoutingAction
from callverse_workforce.reward.breakdown import REWARD_COMPONENTS, RewardBreakdown
from callverse_workforce.twin.environment import RoutingEnvironment
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.utils.rng import RngManager

FLOAT_TOL = 1e-9  # documented reconstruction tolerance (replays are bit-deterministic; 0 mismatches expected)


class DiscountedReturn:
    """Reward observer computing the exact SMDP-discounted return from ``origin``."""

    def __init__(self, clock, origin: float, gamma: float) -> None:
        self.clock, self.origin, self.gamma = clock, origin, gamma
        self.log_inv = -math.log(gamma)
        self.components = {c: 0.0 for c in REWARD_COMPONENTS}
        self.undiscounted = {c: 0.0 for c in REWARD_COMPONENTS}

    def factor(self, t0: float, t1: float) -> float:
        """Average discount over [t0, t1] (point discount when t0 == t1)."""
        a = self.gamma ** (t0 - self.origin)
        if t1 - t0 <= 0:
            return a
        return (a - self.gamma ** (t1 - self.origin)) / ((t1 - t0) * self.log_inv)

    def __call__(self, delta: RewardBreakdown, dt: float) -> None:
        now = self.clock()
        t0 = now - dt
        if t0 < self.origin - FLOAT_TOL:
            raise AssertionError("reward interval starts before the branch point")
        f = self.factor(max(t0, self.origin), now)
        for c in REWARD_COMPONENTS:
            v = getattr(delta, c)
            if v:
                self.components[c] += v * f
                self.undiscounted[c] += v

    @property
    def total(self) -> float:
        return sum(self.components.values())


@dataclass(frozen=True)
class StateKey:
    """What a reconstructed target decision must match."""

    target_index: int
    decision_id: str
    contact_id: str
    timestamp: float
    observation: dict[str, Any]
    mask: tuple[bool, ...]


@dataclass(frozen=True)
class BranchJob:
    state_id: str
    config: SimulationConfig
    seed: int
    key: StateKey
    action: str
    variant: int
    pre_policy: tuple[str, tuple[tuple[str, Any], ...]]
    continuation: tuple[str, tuple[tuple[str, Any], ...]]


def _policy(spec: tuple[str, tuple[tuple[str, Any], ...]], seed: int):
    name, params = spec
    return make_policy(name, dict(params), seed)


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=FLOAT_TOL)
    return a == b


def reconstruction_mismatch(dp, key: StateKey) -> str | None:
    if dp is None:
        return "day ended before the target decision"
    if dp.decision_id != key.decision_id:
        return f"decision_id {dp.decision_id} != {key.decision_id}"
    if dp.contact_id != key.contact_id:
        return f"contact_id {dp.contact_id} != {key.contact_id}"
    if not _same(dp.timestamp, key.timestamp):
        return f"timestamp {dp.timestamp} != {key.timestamp}"
    if dp.action_mask.valid != key.mask:
        return f"mask {dp.action_mask.valid} != {key.mask}"
    obs = dp.observation.to_dict()
    bad = [k for k in key.observation if not _same(obs[k], key.observation[k])]
    return f"observation differs in {bad}" if bad else None


_CACHE: OrderedDict[tuple[str, int, int], tuple] = OrderedDict()
_CACHE_SIZE = 24


def _contacts(cfg: SimulationConfig, seed: int, variant: int) -> tuple:
    """Per-process LRU memo of generated (immutable) contacts: variant -1 = base day, k >= 0 = future k.

    Generation is a pure function of (config, seed, variant), so memoization cannot change any result.
    """
    key = (cfg.model_dump_json(), seed, variant)
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return _CACHE[key]
    gen = ContactGenerator(cfg, RngManager(seed))
    value = tuple(gen.generate_day() if variant < 0 else gen.generate_future_variant(variant, -1.0))
    _CACHE[key] = value
    if len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return value


def simulate_branch(job: BranchJob):
    """Run one branch; returns (status dict, env, tracker, details). env/tracker are None on failure."""
    cfg = job.config
    env = RoutingEnvironment(cfg, contacts=_contacts(cfg, job.seed, -1))
    pre = _policy(job.pre_policy, job.seed)
    pre.reset()
    dp = env.reset(job.seed)
    for _ in range(job.key.target_index):
        if dp is None:
            break
        dp = env.step(pre.predict(dp.observation, dp.action_mask)).next_decision
    out: dict[str, Any] = {"state_id": job.state_id, "action": job.action, "variant": job.variant,
                           "continuation": job.continuation[0]}
    mismatch = reconstruction_mismatch(dp, job.key)
    if mismatch:
        return out | {"status": "RECONSTRUCTION_FAILED", "reason": mismatch}, None, None, {}
    action = RoutingAction(job.action)
    if not dp.action_mask.is_valid(action):
        return out | {"status": "INVALID_FORCED_ACTION", "reason": f"{action.value} is masked"}, None, None, {}

    twin = env.twin
    t_target = twin.now
    future = [c for c in _contacts(cfg, job.seed, job.variant) if c.arrival_time > t_target]
    twin.activate_future_variant(job.variant, future)
    tracker = DiscountedReturn(lambda: twin.now, t_target, cfg.smdp.gamma_per_minute)
    twin.reward.observer = tracker
    events_before = len(twin.events)

    step = env.step(action)  # the forced action, applied exactly once
    cont = _policy(job.continuation, job.seed)
    cont.reset()
    dp = step.next_decision
    n_after = 0
    while dp is not None:
        dp = env.step(cont.predict(dp.observation, dp.action_mask)).next_decision
        n_after += 1
    details = {"t_target": t_target, "first": step.transition, "n_after": n_after, "future": future,
               "events_before": events_before}
    return out | {"status": "OK"}, env, tracker, details


def run_branch(job: BranchJob) -> dict[str, Any]:
    out, env, tracker, d = simulate_branch(job)
    if out["status"] != "OK":
        return out
    result = env.result()
    transitions_total = sum(t.reward for t in result.transitions) + result.pre_decision_reward.total
    target = [x for x in result.decisions if x.decision_id == job.key.decision_id]
    return out | {
        "t_target": d["t_target"],
        "elapsed_to_horizon": result.horizon_minutes - d["t_target"],
        "discounted_return": tracker.total,
        "discounted_components": dict(tracker.components),
        "undiscounted_return": sum(tracker.undiscounted.values()),
        "undiscounted_components": dict(tracker.undiscounted),
        "first_interval_reward": d["first"].reward,
        "first_interval_delta_t": d["first"].delta_t_minutes,
        "decisions_after_target": d["n_after"],
        "future_arrivals": len(d["future"]),
        "events_before_branch": d["events_before"],
        "forced_action_applied": len(target) == 1 and target[0].action is RoutingAction(job.action),
        "reward_reconciles": math.isclose(transitions_total, result.episode_reward.total, abs_tol=1e-6),
        "terminal_reconciles": all(c.outcome is not None for c in result.contacts)
        and len(result.contacts) == sum(1 for e in result.events if e.event_type.value == "ARRIVAL"),
    }


def policy_spec(name: str, params: dict[str, Any]) -> tuple[str, tuple[tuple[str, Any], ...]]:
    """Hashable/picklable frozen policy specification."""
    return name, tuple(sorted((k, v) for k, v in params.items() if not isinstance(v, (list, dict))))
