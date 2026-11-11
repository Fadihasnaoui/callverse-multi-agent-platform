"""Deterministic, time-stratified sampling of headroom states (actual RulePolicy decision states).

Strata follow the NORMAL arrival profile (t=0 is 08:00; hourly rates 40,60,80,90,80,65,65,80,85,70,55,40):

    calm       08:00-10:00   [0, 120)
    rising     10:00-11:00   [120, 180)
    peak       11:00-13:00 and 15:00-17:00   [180, 300) u [420, 540)
    post_peak  13:00-15:00   [300, 420)
    late       17:00-close   [540, horizon)

Per day quota (30): calm 4, rising 5, peak 12, post_peak 4, late 5 -- peak-weighted but covering the whole day.
Only decisions with >= 2 valid actions are eligible. The sample is NOT conditioned on Rule/AI-first
disagreement. Selection uses a dedicated RNG keyed by (day seed, load level) only.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from callverse_workforce.baselines.policies import AIFirstPolicy, make_policy
from callverse_workforce.config.models import SimulationConfig
from callverse_workforce.encoding import ObservationEncoder
from callverse_workforce.domain.enums import ACTION_ORDER
from callverse_workforce.twin.environment import RoutingEnvironment

STRATA: tuple[tuple[str, tuple[tuple[float, float], ...]], ...] = (
    ("calm", ((0.0, 120.0),)),
    ("rising", ((120.0, 180.0),)),
    ("peak", ((180.0, 300.0), (420.0, 540.0))),
    ("post_peak", ((300.0, 420.0),)),
    ("late", ((540.0, float("inf")),)),
)
QUOTA: dict[str, int] = {"calm": 4, "rising": 5, "peak": 12, "post_peak": 4, "late": 5}
_ENCODER = ObservationEncoder()


def stratum_of(t: float) -> str:
    for name, ranges in STRATA:
        if any(lo <= t < hi for lo, hi in ranges):
            return name
    raise ValueError(t)


def trajectory(
    config: SimulationConfig, seed: int, rule_params: dict[str, Any], threshold_theta: float = 0.30
) -> list[dict[str, Any]]:
    """All decision points of the RulePolicy day (the pre-target trajectory)."""
    env = RoutingEnvironment(config)
    rule = make_policy("RulePolicy", rule_params, seed)
    threshold = make_policy("ThresholdPolicy", {"theta": threshold_theta}, seed)
    ai_first = AIFirstPolicy()
    out = []
    dp = env.reset(seed)
    index = 0
    while dp is not None:
        obs, mask = dp.observation, dp.action_mask
        rule_action = rule.predict(obs, mask)
        out.append({
            "target_index": index,
            "decision_id": dp.decision_id,
            "contact_id": dp.contact_id,
            "timestamp": dp.timestamp,
            "stratum": stratum_of(dp.timestamp),
            "observation": obs.to_dict(),
            "encoded": [float(x) for x in _ENCODER.encode(obs)],
            "mask": {a.value: ok for a, ok in zip(ACTION_ORDER, mask.valid)},
            "mask_reasons": {k: list(v) for k, v in mask.reasons.items()},
            "n_valid": mask.n_valid,
            "rule_action": rule_action.value,
            "aifirst_action": ai_first.predict(obs, mask).value,
            "threshold_action": threshold.predict(obs, mask).value,
        })
        dp = env.step(rule_action).next_decision
        index += 1
    return out


def sample_states(
    config: SimulationConfig, load: str, seed: int, rule_params: dict[str, Any], quota: dict[str, int] = QUOTA,
    threshold_theta: float = 0.30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traj = trajectory(config, seed, rule_params, threshold_theta)
    eligible = [d for d in traj if d["n_valid"] >= 2]
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(
        seed, spawn_key=(zlib.crc32(b"headroom_sampling"), zlib.crc32(load.encode())))))
    chosen: list[dict[str, Any]] = []
    shortfall = {}
    for name, _ in STRATA:
        pool = [d for d in eligible if d["stratum"] == name]
        n = min(quota[name], len(pool))
        shortfall[name] = quota[name] - n
        picks = sorted(rng.choice(len(pool), size=n, replace=False).tolist()) if n else []
        chosen += [pool[i] for i in picks]
    chosen.sort(key=lambda d: d["target_index"])
    states = [_state_record(d, load, seed) for d in chosen]
    meta = {"seed": seed, "load": load, "decisions": len(traj), "eligible_decisions": len(eligible),
            "one_valid_decisions": len(traj) - len(eligible), "sampled": len(states), "quota_shortfall": shortfall}
    return states, meta


def _state_record(d: dict[str, Any], load: str, seed: int, noise: str = "MEDIUM") -> dict[str, Any]:
    o = d["observation"]
    return {
        "state_id": f"{load}|{noise}|{seed}|{d['target_index']}",
        "load": load,
        "noise": noise,
        "seed": seed,
        **d,
        "request_type": o["request_type"],
        "ai_resolution_score": o["ai_resolution_score"],
        "human_utilization": o["human_utilization"],
        "human_queue_size": o["human_queue_size"],
        "urgency_pred": o["urgency_pred"],
        "waiting_time_so_far": o["waiting_time_so_far"],
        "forecast_30min": o["forecast_30min"],
        "time_bucket": d["stratum"],
    }


def link_states(
    primary: list[dict[str, Any]], config: SimulationConfig, noise: str, rule_params: dict[str, Any]
) -> list[dict[str, Any]]:
    """Matched diagnostic states for another noise level.

    Changing score noise changes observed scores, hence Rule decisions, hence the trajectory, so the exact
    primary states cannot be reconstructed. For each primary state we take, in the same day/load under the
    new noise level, (1) the same contact at the same routing attempt if it is an eligible decision, else
    (2) the eligible decision of the same stratum closest in time. Link type and time gap are recorded.
    """
    by_seed: dict[int, list[dict[str, Any]]] = {}
    linked = []
    for p in primary:
        traj = by_seed.setdefault(p["seed"], [d for d in trajectory(config, p["seed"], rule_params) if d["n_valid"] >= 2])
        attempt = p["observation"]["routing_attempt_count"]
        same = [d for d in traj if d["contact_id"] == p["contact_id"]
                and d["observation"]["routing_attempt_count"] == attempt]
        if same:
            d, link = same[0], "same_contact"
        else:
            pool = [d for d in traj if d["stratum"] == p["stratum"]] or traj
            d, link = min(pool, key=lambda x: (abs(x["timestamp"] - p["timestamp"]), x["target_index"])), "nearest_time"
        rec = _state_record(d, p["load"], p["seed"], noise)
        rec |= {"linked_primary_state": p["state_id"], "link_type": link,
                "link_time_gap": abs(d["timestamp"] - p["timestamp"])}
        linked.append(rec)
    return linked
