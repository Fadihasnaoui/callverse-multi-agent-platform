"""Phase F DEV/TEST harness (simulation substrate). Not used by the runtime agent itself.

Hidden information (scenario family, windows, event truth) is used here ONLY after the fact, to label
evaluation windows (TRUE_OPERATIONAL_DEGRADATION); it is never passed to the agent.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from callverse_workforce.agent.actions import V1_1_REGISTRY
from callverse_workforce.agent.config import SupervisionConfig
from callverse_workforce.agent.emergency import evaluate_triggers
from callverse_workforce.agent.executor import RoutingExecutor, simulation_handlers
from callverse_workforce.agent.manager import SimulatedManager
from callverse_workforce.agent.perception import SimulationPerception, public_context_from
from callverse_workforce.agent.routing_policy import PPOPolicy, RulePolicyAdapter
from callverse_workforce.agent.workforce_manager import WorkforceManagerAgent
from callverse_workforce.baselines.metrics import policy_day_metrics
from callverse_workforce.config import load_config
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.phase_e.scenarios import FAMILIES, DaySpec
from callverse_workforce.tracing.summary import summarize
from callverse_workforce.twin.environment import RoutingEnvironment

WINDOW, STEP = 30.0, 15.0
MIN_ATTEMPTS, MIN_ARRIVALS = 10, 10
KPI_KEYS = ("ai_failure_rate_30", "n_ai_30", "abandonment_rate_30", "n_arrivals_30", "human_queue_growth", "human_queue_size")


@lru_cache(maxsize=4)
def _base(path: str):
    return load_config(path)


def phase_f_days(seeds: Sequence[int], per_family: int) -> list[DaySpec]:
    return [DaySpec(FAMILIES[i // per_family], MIXTURE_CELLS[(i % per_family) % 4], s) for i, s in enumerate(seeds)]


def test_days(seeds_main: Sequence[int], seeds_extra_ai: Sequence[int]) -> list[DaySpec]:
    """270-day TEST map: 8 x 30 (3500-3739) + 30 extra AI_DEGRADATION days (3740-3769, cells continue the rotation)."""
    days = phase_f_days(seeds_main, 30)
    days += [DaySpec("AI_DEGRADATION", MIXTURE_CELLS[(30 + i) % 4], s) for i, s in enumerate(seeds_extra_ai)]
    return days


def run_agent_day(day: DaySpec, model_path: str, cfg: SupervisionConfig, rule_params: dict[str, Any], base_path: str, *,
                  whatif_cache: dict | None = None, keep_audit: bool = False, policy=None) -> dict[str, Any]:
    live_cfg = EV._config(base_path, day)  # hidden live configuration (harness only)
    base = _base(base_path)
    env = RoutingEnvironment(live_cfg)
    dp = env.reset(day.seed)
    public = public_context_from(live_cfg, base)
    perception = SimulationPerception(env, public)
    agent = WorkforceManagerAgent(policy or PPOPolicy(model_path), RulePolicyAdapter(rule_params), cfg, public, base,
                                  SimulatedManager(cfg.manager, cfg.g_min, cfg.approval_delay_minutes), day_seed=day.seed,
                                  whatif_cache=whatif_cache, keep_decision_records=keep_audit)
    executor = RoutingExecutor(V1_1_REGISTRY, simulation_handlers(V1_1_REGISTRY, env))
    actions = []
    while dp is not None:
        agent.observe(perception.new_events())
        out = agent.route(perception.percept(dp))
        actions.append(out.action)
        dp = executor.execute(out.action).next_decision
    agent.observe(perception.new_events())
    result = env.result()
    result.metrics = summarize(result)
    m = policy_day_metrics(result)
    row = {"seed": day.seed, "family": day.family, "cell": day.cell.cell_id, "config_id": cfg.config_id(), "enabled": cfg.enabled,
           "manager": cfg.manager, "world_model": cfg.world_model, "k": cfg.k_futures,
           "total_reward": m["total_reward"], "reward": m["reward"], "action_share": m["action_share"],
           "resolution_rate_instance": m["resolution_rate_instance"], "abandonment_rate": m["abandonment_rate"],
           "human_utilization": m["human_utilization"], "online_wait_p90": m["online_wait_p90"],
           "false_ai_count": sum(1 for e in result.events if e.event_type.value == "AI_FALSE_RESOLUTION"),
           "agent": agent_summary(agent, live_cfg.scenario.horizon_minutes),
           "action_hash": hashlib.sha256("|".join(actions).encode()).hexdigest()}
    if keep_audit:
        row["_agent"] = agent
        row["_result"] = result
    return row


def agent_summary(agent, horizon: float) -> dict[str, Any]:
    mem = agent.memory
    rec = lambda k: mem.records(k)
    starts = [r for r in rec("PrudencePeriod") if r["phase"] == "start"]
    prudence = [(r["start"], min(r["scheduled_end"], horizon)) for r in starts]
    return {
        "emergencies": [(r["t"], r["trigger"]) for r in rec("EmergencyRecord")],
        "whatifs": [{"t": r["t"], "gain": r["estimated_gain"], "t80": r["t80_lower"], "share": r["share_positive"],
                     "ai_multiplier": r["world"]["ai_multiplier"]} for r in rec("WhatIfRecord")],
        "requests": [r["t"] for r in rec("ManagerRequest")],
        "approvals": sum(1 for r in rec("ManagerResponse") if r["decision"] == "APPROVE"),
        "rejections": sum(1 for r in rec("ManagerResponse") if r["decision"] == "REJECT"),
        "prudence_windows": prudence, "prudence_minutes": float(sum(e - s for s, e in prudence)),
        "technical_fallbacks": sum(1 for r in rec("TechnicalFallbackEvent") if r["phase"] == "start"),
        "routed_by": dict(agent.routed_by), "mask_mismatches": agent.guardrails.mask_mismatches,
        "kpi_trace": [[r["t"]] + [r["kpis"][k] if k in r["kpis"] else None for k in KPI_KEYS] for r in rec("SupervisionSnapshot")],
        "transitions": [(r["t"], r["from"], r["to"], r["reason"]) for r in rec("ModeTransition")],
    }


# -------------------------------------------------------------------------------------------------------
# evaluation-only ground truth (hindsight, from the PPO_ALONE run of the same day)
# -------------------------------------------------------------------------------------------------------
def window_stats(result) -> list[dict[str, float]]:
    horizon = result.horizon_minutes
    attempt_start: dict[str, float] = {}
    attempts, fails, arrivals, abandons = [], [], [], []
    for e in result.events:
        et = e.event_type.value
        if et == "AI_ATTEMPT_STARTED":
            attempt_start[e.contact_id] = e.time
            attempts.append(e.time)
        elif et in ("AI_VISIBLE_FAILURE", "AI_FALSE_RESOLUTION"):
            fails.append(attempt_start.get(e.contact_id, e.time))
        elif et == "ARRIVAL":
            arrivals.append(e.time)
        elif et == "ABANDONMENT":
            abandons.append(e.time)
    q = [(t.timestamp, t.state.human_queue_size) for t in result.transitions]
    cnt = lambda xs, s: sum(1 for x in xs if s <= x < s + WINDOW)
    out = []
    s = 0.0
    while s + WINDOW <= horizon:
        na, nf, narr, nab = cnt(attempts, s), cnt(fails, s), cnt(arrivals, s), cnt(abandons, s)
        qs = [v for t, v in q if s <= t < s + WINDOW]
        out.append({"start": s, "ai_fail": nf / na if na >= MIN_ATTEMPTS else None,
                    "abandon": nab / narr if narr >= MIN_ARRIVALS else None, "queue": float(np.mean(qs)) if qs else None})
        s += STEP
    return out


def thresholds_from_normal(windows_by_day: Sequence[list[dict[str, float]]]) -> dict[str, float]:
    vals = defaultdict(list)
    for w in windows_by_day:
        for x in w:
            for k in ("ai_fail", "abandon", "queue"):
                if x[k] is not None:
                    vals[k].append(x[k])
    return {f"q_{k}": float(np.percentile(v, 95)) for k, v in vals.items()}


def episodes(windows: list[dict[str, float]], q: dict[str, float]) -> list[tuple[float, float]]:
    deg = [w["start"] for w in windows
           if (w["ai_fail"] is not None and w["ai_fail"] >= q["q_ai_fail"]) or (w["abandon"] is not None and w["abandon"] >= q["q_abandon"])
           or (w["queue"] is not None and w["queue"] >= q["q_queue"])]
    eps: list[tuple[float, float]] = []
    for s in deg:
        if eps and s <= eps[-1][1]:
            eps[-1] = (eps[-1][0], s + WINDOW)
        else:
            eps.append((s, s + WINDOW))
    return eps


def alert_quality(alerts: Sequence[float], eps: Sequence[tuple[float, float]], lead: float = 15.0) -> dict[str, Any]:
    true = [a for a in alerts if any(s - lead <= a < e for s, e in eps)]
    detected = []
    for s, e in eps:
        hits = [a for a in alerts if s - lead <= a < e]
        if hits:
            detected.append(max(0.0, min(hits) - s))
    return {"alerts": len(alerts), "true_alerts": len(true), "false_alerts": len(alerts) - len(true), "episodes": len(eps),
            "detected": len(detected), "missed": len(eps) - len(detected), "delays": detected}


def replay_detector(trace: Sequence[Sequence[float]], cfg: SupervisionConfig) -> list[float]:
    """D1 screening: detector on the PPO_ALONE KPI trace; every alert assumed KEEP (cooldown, no intervention)."""
    alerts, streak, cooldown = [], 0, -1e9
    for row in trace:
        t = row[0]
        k = dict(zip(KPI_KEYS, row[1:]))
        if t < cooldown:
            streak = 0
            continue
        hits = evaluate_triggers(k, cfg)
        streak = streak + 1 if hits else 0
        if hits and streak >= cfg.debounce:
            alerts.append(t)
            streak = 0
            cooldown = t + cfg.cooldown_minutes
    return alerts


def mean_diff(a_rows: Sequence[dict], b_rows: Sequence[dict], families: Sequence[str]) -> tuple[float, float]:
    """(mean B-A over paired days of the families, mean A over those days)."""
    a = {r["seed"]: r for r in a_rows if r["family"] in families}
    b = {r["seed"]: r for r in b_rows if r["family"] in families}
    seeds = sorted(set(a) & set(b))
    if len(seeds) != len(a):
        raise ValueError("unpaired days")
    return (float(np.mean([b[s]["total_reward"] - a[s]["total_reward"] for s in seeds])),
            float(np.mean([a[s]["total_reward"] for s in seeds])))


def condition_check(a_rows, b_rows, families_ai=("AI_DEGRADATION",), b_key: int = 0) -> dict[str, Any]:
    """Primary TEST conditions A and B (day-paired bootstrap, frozen phase_e implementation)."""
    others = tuple(f for f in FAMILIES if f not in families_ai)
    pair = lambda fams: ({r["seed"]: r["total_reward"] for r in a_rows if r["family"] in fams},
                         {r["seed"]: r["total_reward"] for r in b_rows if r["family"] in fams})
    a1, b1 = pair(families_ai)
    a2, b2 = pair(others)
    ci1 = EV.paired_bootstrap(EV.paired_diff(b1, a1), key=500 + b_key)
    ci2 = EV.paired_bootstrap(EV.paired_diff(b2, a2), key=501 + b_key)
    margin = -0.01 * float(np.mean(list(a2.values())))
    return {"condition_A": ci1 | {"pass": ci1["lo"] > 0}, "condition_B": ci2 | {"margin": margin, "pass": ci2["lo"] > margin}}


def partial_day(day: DaySpec, model_path: str, cfg: SupervisionConfig, rule_params: dict[str, Any], base_path: str,
                until: float):
    """Run a live day up to the first routing event at or after ``until`` (leakage tests)."""
    live_cfg = EV._config(base_path, day)
    base = _base(base_path)
    env = RoutingEnvironment(live_cfg)
    dp = env.reset(day.seed)
    public = public_context_from(live_cfg, base)
    perception = SimulationPerception(env, public)
    agent = WorkforceManagerAgent(PPOPolicy(model_path), RulePolicyAdapter(rule_params), cfg, public, base,
                                  SimulatedManager(cfg.manager, cfg.g_min, cfg.approval_delay_minutes), day_seed=day.seed)
    executor = RoutingExecutor(V1_1_REGISTRY, simulation_handlers(V1_1_REGISTRY, env))
    while dp is not None and dp.timestamp < until:
        agent.observe(perception.new_events())
        dp = executor.execute(agent.route(perception.percept(dp)).action).next_decision
    agent.observe(perception.new_events())
    return agent, env, perception, executor, dp


# -------------------------------------------------------------------------------------------------------
# DEV worker jobs (module level => picklable)
# -------------------------------------------------------------------------------------------------------
def _day(t) -> DaySpec:
    family, cell_id, seed = t
    return DaySpec(family, {c.cell_id: c for c in MIXTURE_CELLS}[cell_id], seed)


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if not k.startswith("_")}


def base_job(job) -> dict[str, Any]:
    """A (monitor-only, checked bit-identical to supervision disabled) + C (RULE_ALONE) + ground-truth windows."""
    day_t, model_path, rule_params, base_path = job
    day = _day(day_t)
    off = run_agent_day(day, model_path, SupervisionConfig(enabled=False), rule_params, base_path)
    mon = run_agent_day(day, model_path, SupervisionConfig(t_ai=None, t_ab=None, t_q=None), rule_params, base_path, keep_audit=True)
    windows = window_stats(mon["_result"])
    rule = run_agent_day(day, model_path, SupervisionConfig(enabled=False), rule_params, base_path,
                         policy=RulePolicyAdapter(rule_params))
    a = _public_row(mon)
    a["bit_identical_to_disabled"] = (off["action_hash"] == mon["action_hash"] and off["total_reward"] == mon["total_reward"])
    return {"A": a, "C": _public_row(rule), "windows": windows}


def grid_job(job) -> list[dict[str, Any]]:
    """All configurations sharing (detector, D, K) on one day, with a shared what-if cache (identical snapshots)."""
    day_t, model_path, rule_params, base_path, cfgs = job
    day = _day(day_t)
    cache: dict = {}
    out = []
    for cfg_dict in cfgs:
        cfg = SupervisionConfig(**cfg_dict)
        row = _public_row(run_agent_day(day, model_path, cfg, rule_params, base_path, whatif_cache=cache))
        row["agent"].pop("kpi_trace", None)
        out.append(row)
    return out
