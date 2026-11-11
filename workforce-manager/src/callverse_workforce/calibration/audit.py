"""Calibration / synthetic-assumption audit (multi-seed) and paired B1 vs B1.5 comparison.

The audit REPORTS the current synthetic assumptions next to what the twin
produces over a fixed list of NORMAL-day seeds. It never modifies parameters.
Observed metrics come from the DevelopmentPolicy (deterministic round-robin),
which is not a realistic operating policy.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Sequence

import numpy as np

from callverse_workforce.calibration.seeds import CALIBRATION_SEEDS, RESERVED_RANGES
from callverse_workforce.calibration.variants import B1_5_OVERRIDES, B1_OVERRIDES, b1_5_config, b1_config
from callverse_workforce.config.models import MeanCv, SimulationConfig
from callverse_workforce.domain.enums import (
    CallbackReason,
    Channel,
    ContactState,
    EventType,
    LanguageDialect,
    Outcome,
    RequestType,
)
from callverse_workforce.routing.policy import DevelopmentPolicy
from callverse_workforce.twin.patience import PatienceDistribution
from callverse_workforce.twin.result import SimulationResult
from callverse_workforce.twin.runner import run_simulation

AUDIT_VERSION = "assumption_audit_v2_multiseed"
LABEL = "SYNTHETIC_ASSUMPTION_V1"
POLICY_NOTE = "DevelopmentPolicy (deterministic round-robin; NOT a realistic operating policy)"
Z90 = 1.2815515655446004
MIN_CELL_SAMPLE = 30
FLAG_PATIENCE_UNDER_1MIN_SHARE = 0.10
FLAG_ABANDONMENT_RATE = 0.05
FLAG_SHORT_P90_WAIT_MIN = 5.0
FLAG_CALLBACK_FINAL_FAILURE = 0.01
UNDERPRICING_WARNING = (
    "The current -1 penalty for MAX_ATTEMPTS / WINDOW_EXPIRED may underprice the business cost of an "
    "unresolved/lost customer. It is NOT modified in B1.5 and will be examined later in the reward "
    "sensitivity analysis."
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _r(x: Any, nd: int = 4) -> Any:
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _agg(values: Sequence[float]) -> dict[str, Any]:
    """Across-seed aggregate: mean, std (ddof=1), min, p10, median, p90, max."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": _r(a.mean()),
        "std": _r(a.std(ddof=1)) if a.size > 1 else 0.0,
        "min": _r(a.min()),
        "p10": _r(np.percentile(a, 10)),
        "median": _r(np.median(a)),
        "p90": _r(np.percentile(a, 90)),
        "max": _r(a.max()),
    }


def _dist(values: Sequence[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": _r(a.mean()),
        "median": _r(np.median(a)),
        "p10": _r(np.percentile(a, 10)),
        "p90": _r(np.percentile(a, 90)),
        "min": _r(a.min()),
        "max": _r(a.max()),
        "frac_below_1min": _r(np.mean(a < 1.0)),
        "frac_below_2min": _r(np.mean(a < 2.0)),
    }


def _flag(code: str, severity: str, message: str, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message, "evidence": evidence or {}}


def _lognormal_summary(spec: MeanCv, multiplier: float = 1.0) -> dict[str, Any]:
    mean = spec.mean * multiplier
    sigma = math.sqrt(math.log(1 + spec.cv**2))
    mu = math.log(mean) - sigma**2 / 2
    return {
        "distribution": "lognormal(mean, cv)", "mean": _r(mean), "cv": spec.cv, "median": _r(math.exp(mu)),
        "p10": _r(math.exp(mu - Z90 * sigma)), "p90": _r(math.exp(mu + Z90 * sigma)), "min": 0.0, "max": "unbounded",
    }


def _patience_samples(result: SimulationResult) -> list[tuple[str, float]]:
    """(channel, sampled patience) for every contact instance (hidden truth, simulator side)."""
    channel = {c.contact_id: c.channel.value for c in result.contacts}
    return [
        (channel[e.contact_id], float(e.data["truth"]["patience_minutes"]))
        for e in result.events
        if e.event_type is EventType.ARRIVAL
    ]


# ---------------------------------------------------------------------------
# per-day metrics (one simulated day)
# ---------------------------------------------------------------------------
def day_metrics(result: SimulationResult) -> dict[str, Any]:
    m = result.metrics
    contacts = result.contacts
    n = len(contacts)
    patience = [p for _, p in _patience_samples(result)]
    abandoned = [c for c in contacts if c.outcome is Outcome.ABANDONED]
    in_wait = sum(
        1 for e in result.events
        if e.event_type is EventType.ABANDONMENT and e.data["from_state"] is ContactState.WAITING
    )
    by_channel: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in contacts:
        by_channel[c.channel.value][1] += 1
        by_channel[c.channel.value][0] += c.outcome is Outcome.ABANDONED
    waits = np.array([c.online_wait_minutes for c in contacts], dtype=float)

    cb = m["callback"]
    rates = cb["reason_rates"]
    busy = m["human_busy_minutes"]
    failure = sum(rates[r.value] for r in (CallbackReason.CUSTOMER_DROPOUT, CallbackReason.MAX_ATTEMPTS,
                                           CallbackReason.WINDOW_EXPIRED, CallbackReason.MISSED_DEADLINE))
    return {
        "instances": n,
        "patience_below_1min": float(np.mean(np.array(patience) < 1.0)) if patience else 0.0,
        "patience_below_2min": float(np.mean(np.array(patience) < 2.0)) if patience else 0.0,
        "abandonment_rate": len(abandoned) / n if n else 0.0,
        "abandonment_rate_by_channel": {ch.value: (by_channel[ch.value][0] / by_channel[ch.value][1]
                                                   if by_channel[ch.value][1] else None) for ch in Channel},
        "share_of_abandonments_in_wait": in_wait / len(abandoned) if abandoned else 0.0,
        "online_wait_mean": float(waits.mean()) if n else 0.0,
        "online_wait_p90": float(np.percentile(waits, 90)) if n else 0.0,
        "callbacks_scheduled": cb["scheduled"],
        "callback_dial_attempts": cb["dial_attempts"],
        "callback_per_dial_answer_rate": cb["per_dial_answer_rate"],
        "callback_success_rate": rates[CallbackReason.CALLBACK_SUCCESS.value],
        "callback_failure_rate": failure,
        "callback_dropout_rate": rates[CallbackReason.CUSTOMER_DROPOUT.value],
        "callback_max_attempts_rate": rates[CallbackReason.MAX_ATTEMPTS.value],
        "callback_window_expired_rate": rates[CallbackReason.WINDOW_EXPIRED.value],
        "callback_missed_deadline_rate": rates[CallbackReason.MISSED_DEADLINE.value],
        "callback_unfinished_rate": rates[CallbackReason.UNFINISHED.value],
        "callback_reason_counts": cb["reasons"],
        "callback_pending_median": cb["pending_minutes_median"],
        "callback_pending_p90": cb["pending_minutes_p90"],
        "callback_human_minutes": cb["human_minutes"],
        "callback_share_of_human_busy": cb["human_minutes"] / busy if busy else 0.0,
        "hir_instance": m["human_intervention_rate"]["instance_level"],
        "hir_root": m["human_intervention_rate"]["root_level"],
        "human_routing_rate": m["human_routing_rate"],
        "human_busy_minutes": busy,
        "capacity_minutes": m["capacity"]["capacity_minutes"],
        "utilization": m["capacity"]["utilization"],
        "total_reward": m["total_reward"],
        "reconciliation_balanced": m["reconciliation"]["balanced"],
    }


SCALAR_KEYS = (
    "instances", "patience_below_1min", "patience_below_2min", "abandonment_rate", "share_of_abandonments_in_wait",
    "online_wait_mean", "online_wait_p90",
    "callbacks_scheduled", "callback_dial_attempts", "callback_per_dial_answer_rate", "callback_success_rate",
    "callback_failure_rate", "callback_dropout_rate", "callback_max_attempts_rate", "callback_window_expired_rate",
    "callback_missed_deadline_rate", "callback_unfinished_rate", "callback_pending_median", "callback_pending_p90",
    "callback_human_minutes", "callback_share_of_human_busy",
    "hir_instance", "hir_root", "human_routing_rate", "human_busy_minutes", "capacity_minutes", "utilization",
    "total_reward",
)


def _aggregate_days(days: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {k: _agg([d[k] for d in days]) for k in SCALAR_KEYS}
    out["abandonment_rate_by_channel"] = {
        ch.value: _agg([d["abandonment_rate_by_channel"][ch.value] for d in days]) for ch in Channel
    }
    out["callback_reason_counts_total"] = {
        r.value: sum(d["callback_reason_counts"][r.value] for d in days) for r in CallbackReason
    }
    out["all_days_reconciled"] = all(d["reconciliation_balanced"] for d in days)
    return out


def _run_days(config: SimulationConfig, seeds: Sequence[int]) -> list[SimulationResult]:
    return [run_simulation(config, DevelopmentPolicy(), seed=s) for s in seeds]


# ---------------------------------------------------------------------------
# configured (analytic) patience
# ---------------------------------------------------------------------------
def configured_patience(config: SimulationConfig) -> dict[str, Any]:
    cfg = config.patience
    by_type_channel = {}
    for rtype, rcfg in config.request_types.items():
        for ch, chcfg in config.channels.items():
            mean = rcfg.patience_mean_minutes * chcfg.patience_multiplier
            d = PatienceDistribution.from_config(mean, cfg)
            by_type_channel[f"{rtype.value}/{ch.value}"] = {
                "model": d.model,
                "configured_mean": _r(mean),
                "floor": d.floor,
                "median": _r(d.quantile(0.5)),
                "p10": _r(d.quantile(0.1)),
                "p90": _r(d.quantile(0.9)),
                "min": d.floor,
                "max": "unbounded",
                "p_below_1min": _r(d.cdf(1.0)),
                "p_below_2min": _r(d.cdf(2.0)),
            }
    params = {"model": cfg.model}
    if cfg.model != "exponential":
        params.update(getattr(cfg, cfg.model).model_dump())
    return {"parameters": params, "by_type_channel": by_type_channel}


def _simulated_patience_by_channel(results: Sequence[SimulationResult]) -> dict[str, Any]:
    pooled: dict[str, list[float]] = defaultdict(list)
    for r in results:
        for ch, p in _patience_samples(r):
            pooled[ch].append(p)
    out = {ch.value: _dist(pooled[ch.value]) for ch in Channel}
    out["ALL"] = _dist([p for v in pooled.values() for p in v])
    return out


# ---------------------------------------------------------------------------
# B1.5 multi-seed assumption audit
# ---------------------------------------------------------------------------
def build_assumption_audit(
    config: SimulationConfig,
    seeds: Sequence[int] = CALIBRATION_SEEDS,
    load_check_config: SimulationConfig | None = None,
) -> dict[str, Any]:
    results = _run_days(config, seeds)
    days = [day_metrics(r) for r in results]
    agg = _aggregate_days(days)
    audit: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "label": LABEL,
        "statement": (
            "The current Digital Twin is synthetic and is not yet calibrated to a real agency. Every value is "
            "SYNTHETIC_ASSUMPTION_V1. Nothing was retuned by this audit."
        ),
        "run": {
            "scenario": config.scenario.name,
            "purpose": config.scenario.purpose,
            "seeds": list(seeds),
            "seed_ranges": {k: list(v) for k, v in RESERVED_RANGES.items()},
            "policy": POLICY_NOTE,
            "patience_model": config.patience.model,
            "callback_model": config.callback.model,
        },
        "patience": _patience_section(config, results, agg),
        "ai_behavior": _ai_section(config, results),
        "callback": _callback_section(config, agg),
        "human_service": _human_section(config, results, agg),
        "capacity_check": capacity_check(config, seeds, load_check_config, [d["human_busy_minutes"] for d in days]),
        "per_seed": {str(s): {k: d[k] for k in SCALAR_KEYS} for s, d in zip(seeds, days)},
        "structural_assumptions": _structural_flags(config),
    }
    return audit


def _patience_section(config: SimulationConfig, results, agg) -> dict[str, Any]:
    configured = configured_patience(config)
    simulated = _simulated_patience_by_channel(results)
    flags = []
    high = {k: v["p_below_1min"] for k, v in configured["by_type_channel"].items()
            if v["p_below_1min"] > FLAG_PATIENCE_UNDER_1MIN_SHARE}
    if high:
        flags.append(_flag("PATIENCE_MASS_NEAR_ZERO", "high",
                           "A large share of customers have < 1 min patience.", {"p_below_1min": high}))
    ab, p90 = agg["abandonment_rate"]["mean"], agg["online_wait_p90"]["mean"]
    if ab > FLAG_ABANDONMENT_RATE and p90 < FLAG_SHORT_P90_WAIT_MIN:
        flags.append(_flag(
            "HIGH_ABANDONMENT_AT_SHORT_WAITS", "medium",
            "Abandonment stays above 5% while the p90 online wait is short. Patience means themselves "
            "(not only the shape) still need real queue-log calibration.",
            {"abandonment_rate_mean": ab, "online_wait_p90_mean": p90},
        ))
    wait_share = agg["share_of_abandonments_in_wait"]["mean"]
    if wait_share > 0.2:
        flags.append(_flag("WAIT_ACTION_DRIVES_ABANDONMENT", "medium",
                           "A large share of abandonments happens during deliberate WAIT.",
                           {"share_of_abandonments_in_wait_mean": wait_share}))
    flags.append(_flag("PATIENCE_PARAMETERS_UNCALIBRATED", "high",
                       "Patience shape, floor and means are synthetic; fit them on real abandonment-vs-wait data.",
                       configured["parameters"]))
    return {
        "label": LABEL,
        "configured": configured,
        "simulated_by_channel_pooled_over_seeds": simulated,
        "observed_across_seeds": {
            k: agg[k] for k in ("abandonment_rate", "share_of_abandonments_in_wait", "online_wait_mean",
                                "online_wait_p90", "patience_below_1min", "patience_below_2min")
        } | {"abandonment_rate_by_channel": agg["abandonment_rate_by_channel"]},
        "flags": flags,
    }


def _ai_section(config: SimulationConfig, results: Sequence[SimulationResult]) -> dict[str, Any]:
    configured = {}
    for rtype, rcfg in config.request_types.items():
        per_lang = {}
        for lang, lcfg in config.languages.items():
            correct = min(rcfg.ai_correct_prob * lcfg.ai_correct_multiplier, 1 - rcfg.ai_false_resolution_prob)
            per_lang[lang.value] = {"correct": _r(correct), "false_resolution": rcfg.ai_false_resolution_prob,
                                    "visible_failure": _r(1 - correct - rcfg.ai_false_resolution_prob)}
        configured[rtype.value] = {"base_correct": rcfg.ai_correct_prob,
                                   "false_resolution": rcfg.ai_false_resolution_prob, "by_language": per_lang}
    kinds = {EventType.AI_RESOLVED: "correct", EventType.AI_VISIBLE_FAILURE: "visible_failure",
             EventType.AI_FALSE_RESOLUTION: "false_resolution"}
    cells: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for r in results:
        info = {c.contact_id: c for c in r.contacts}
        for e in r.events:
            if e.event_type in kinds:
                c = info[e.contact_id]
                cells[(c.request_type.value, c.language_dialect.value)][kinds[e.event_type]] += 1

    def rates(cnt: Counter) -> dict[str, Any]:
        total = sum(cnt.values())
        out: dict[str, Any] = {"attempts_completed": total, "low_sample": total < MIN_CELL_SAMPLE}
        for k in ("correct", "visible_failure", "false_resolution"):
            out[f"{k}_rate"] = _r(cnt[k] / total) if total else None
        return out

    by_type, by_lang = defaultdict(Counter), defaultdict(Counter)
    for (t, lang), cnt in cells.items():
        by_type[t].update(cnt)
        by_lang[lang].update(cnt)
    low = sorted(f"{t}/{lang}" for (t, lang), cnt in cells.items() if sum(cnt.values()) < MIN_CELL_SAMPLE)
    flags = [_flag("AI_RATES_ARE_PLACEHOLDERS", "high",
                   "All AI correct / visible-failure / false-resolution rates are SYNTHETIC_ASSUMPTION_V1 and "
                   "unchanged in B1.5. Real values must come from the Customer Advisor team's observed results. "
                   "They must not be tuned to make PPO look better.")]
    if low:
        flags.append(_flag("AI_CELLS_LOW_SAMPLE", "info", "Cells with < 30 completed attempts pooled over all seeds.",
                           {"cells": low}))
    return {
        "label": LABEL,
        "configured": configured,
        "observed_pooled_over_seeds": {
            "by_request_type": {t.value: rates(by_type[t.value]) for t in RequestType},
            "by_language": {lang.value: rates(by_lang[lang.value]) for lang in LanguageDialect},
            "by_request_type_and_language": {f"{t}/{lang}": rates(c) for (t, lang), c in sorted(cells.items())},
        },
        "flags": flags,
    }


def _callback_section(config: SimulationConfig, agg) -> dict[str, Any]:
    cb = config.callback
    analytic_final_failure = (1 - cb.answer_probability) ** cb.max_attempts
    flags = [_flag("CALLBACK_PENALTY_MAY_UNDERPRICE_LOST_CUSTOMER", "high", UNDERPRICING_WARNING,
                   {"callback_failure_penalty": config.reward.callback_failure_penalty})]
    if analytic_final_failure < FLAG_CALLBACK_FINAL_FAILURE:
        flags.append(_flag("CALLBACK_UNREALISTICALLY_RELIABLE", "high",
                           "Callbacks almost never fail on unanswered dials.",
                           {"analytic_max_attempts_probability": _r(analytic_final_failure, 6)}))
    flags.append(_flag("CALLBACK_PARAMETERS_UNCALIBRATED", "high",
                       "Answer probability, attempts, window, dropout hazard and retry delay are synthetic; "
                       "calibrate on real callback logs.", cb.model_dump()))
    flags.append(_flag("CALLBACK_PENDING_COST_LOWER_THAN_ONLINE", "medium",
                       "A pending callback costs 0.02/min vs 0.05/min online (frozen reward coefficient).",
                       {"callback_pending_per_minute": config.reward.callback_pending_per_minute,
                        "online_waiting_per_minute": config.reward.online_waiting_per_minute}))
    return {
        "label": LABEL,
        "model": cb.model,
        "configured": cb.model_dump(),
        "effective": {"answer_probability": cb.answer_probability, "max_attempts": cb.max_attempts,
                      "max_delay_minutes": cb.max_delay_minutes,
                      "dropout_hazard_per_minute": _r(cb.dropout_hazard_per_minute, 8),
                      "analytic_all_attempts_unanswered": _r(analytic_final_failure, 6)},
        "reward_mapping": {
            CallbackReason.CALLBACK_SUCCESS.value: "+2V (resolution)",
            CallbackReason.CUSTOMER_DROPOUT.value: "-3V (abandonment component) only",
            CallbackReason.MAX_ATTEMPTS.value: "-1 (callback failure component)",
            CallbackReason.WINDOW_EXPIRED.value: "-1 (callback failure component)",
            CallbackReason.MISSED_DEADLINE.value: "-4V (missed deadline component)",
            CallbackReason.UNFINISHED.value: "-2V (unfinished component) at closing",
            "all": "plus callback-pending time cost accrued before the terminal event",
        },
        "observed_across_seeds": {k: agg[k] for k in SCALAR_KEYS if k.startswith("callback")}
        | {"reason_counts_total": agg["callback_reason_counts_total"]},
        "flags": flags,
    }


def _human_section(config: SimulationConfig, results, agg) -> dict[str, Any]:
    return {
        "label": LABEL,
        "configured_service_minutes": {t.value: _lognormal_summary(r.human_service_minutes)
                                       for t, r in config.request_types.items()},
        "configured_recontact_service_minutes": {
            t.value: _lognormal_summary(r.human_service_minutes, config.recontact.human_service_multiplier)
            for t, r in config.request_types.items()},
        "observed_across_seeds": {k: agg[k] for k in ("human_busy_minutes", "utilization", "capacity_minutes",
                                                      "hir_instance", "hir_root", "human_routing_rate")},
        "hir_definition": results[0].metrics["human_intervention_rate"]["definition"] if results else "",
        "flags": [
            _flag("HUMANS_ALWAYS_CORRECT", "medium", "Human service always resolves correctly (no rework model)."),
            _flag("NO_SHIFTS_OR_BREAKS", "medium", "Advisor count is constant over the horizon.",
                  {"n_advisors": config.humans.n_advisors}),
        ],
    }


def capacity_check(
    config: SimulationConfig,
    seeds: Sequence[int],
    load_check_config: SimulationConfig | None,
    normal_busy: Sequence[float] | None = None,
) -> dict[str, Any]:
    """capacity_minutes = advisors x available operating minutes (advisors present for the whole horizon)."""
    out: dict[str, Any] = {"formula": "capacity_minutes = n_advisors * operating_minutes; utilization = busy / capacity"}
    for name, cfg in (("normal_day", config), ("load_check", load_check_config)):
        if cfg is None:
            continue
        if name == "normal_day" and normal_busy is not None:
            busy = list(normal_busy)  # already simulated on the same seeds
        else:
            busy = [r.episode_reward.human_busy_minutes for r in _run_days(cfg, seeds)]
        capacity = cfg.humans.n_advisors * cfg.scenario.horizon_minutes
        out[name] = {
            "purpose": cfg.scenario.purpose,
            "n_advisors": cfg.humans.n_advisors,
            "operating_minutes": cfg.scenario.horizon_minutes,
            "arrival_window_minutes": cfg.scenario.arrival_window_minutes,
            "drain_minutes": cfg.scenario.drain_minutes,
            "capacity_minutes": capacity,
            "busy_minutes": _agg(busy),
            "utilization": _agg([b / capacity for b in busy]),
        }
    if "load_check" in out:
        n, lc = out["normal_day"], out["load_check"]
        out["explanation"] = (
            f"load_check has {lc['n_advisors']} advisors over {lc['operating_minutes']:.0f} min "
            f"(no drain) = {lc['capacity_minutes']:.0f} advisor-minutes, versus {n['n_advisors']} x "
            f"{n['operating_minutes']:.0f} = {n['capacity_minutes']:.0f} on the normal day. Its busy minutes are "
            "bounded by that smaller capacity, so it can be ~100% utilized while showing fewer busy minutes "
            "than the normal day. No configuration or utilization bug."
        )
    return out


def _structural_flags(config: SimulationConfig) -> list[dict[str, Any]]:
    return [
        _flag("RECONTACT_PARAMETERS_UNCALIBRATED", "high", "Recontact probability, delay and multipliers.",
              config.recontact.model_dump()),
        _flag("FALSE_AI_DISCOVERY_DELAY_UNCALIBRATED", "high", "Delay until a false AI answer is discovered.",
              config.false_ai_discovery.model_dump()),
        _flag("INTAKE_NOISE_UNCALIBRATED", "high", "Intake error rates of the future real analyzer.",
              config.intake_noise.model_dump()),
        _flag("ARRIVAL_PROFILE_SYNTHETIC", "high", "Hourly arrival profile and request mix are synthetic.",
              {"hourly": list(config.scenario.hourly_arrival_rates)}),
        _flag("DEADLINE_RANGES_SYNTHETIC", "medium", "Deadline ranges per urgency and the 3h mask threshold.",
              {k.value: v.model_dump() for k, v in config.deadlines.items()}),
    ]


# ---------------------------------------------------------------------------
# paired B1 vs B1.5 comparison
# ---------------------------------------------------------------------------
def build_paired_comparison(
    base_config: SimulationConfig,
    seeds: Sequence[int] = CALIBRATION_SEEDS,
    load_check_config: SimulationConfig | None = None,
) -> dict[str, Any]:
    variants = {"b1": b1_config(base_config), "b1_5": b1_5_config(base_config)}
    days = {name: [day_metrics(r) for r in _run_days(cfg, seeds)] for name, cfg in variants.items()}
    summary = {}
    for k in SCALAR_KEYS:
        b1 = [d[k] for d in days["b1"]]
        b15 = [d[k] for d in days["b1_5"]]
        summary[k] = {"b1": _agg(b1), "b1_5": _agg(b15), "paired_diff_b1_5_minus_b1": _agg(np.subtract(b15, b1))}
    summary["abandonment_rate_by_channel"] = {
        ch.value: {name: _agg([d["abandonment_rate_by_channel"][ch.value] for d in days[name]]) for name in days}
        for ch in Channel
    }
    summary["callback_reason_counts_total"] = {
        name: {r.value: sum(d["callback_reason_counts"][r.value] for d in days[name]) for r in CallbackReason}
        for name in days
    }
    return {
        "comparison": "B1 vs B1.5 (paired, identical seeds)",
        "label": LABEL,
        "policy": POLICY_NOTE,
        "seeds": list(seeds),
        "seeds_by_variant": {name: list(seeds) for name in variants},
        "variant_overrides": {"b1": B1_OVERRIDES, "b1_5": B1_5_OVERRIDES},
        "configs": {name: cfg.model_dump(mode="json") for name, cfg in variants.items()},
        "patience_configured": {name: configured_patience(cfg) for name, cfg in variants.items()},
        "summary": summary,
        "per_seed": {name: {str(s): {k: d[k] for k in SCALAR_KEYS} for s, d in zip(seeds, ds)}
                     for name, ds in days.items()},
        "capacity_check": capacity_check(variants["b1_5"], seeds, load_check_config,
                                         [d["human_busy_minutes"] for d in days["b1_5"]]),
        "reconciled": {name: all(d["reconciliation_balanced"] for d in ds) for name, ds in days.items()},
        "warnings": [UNDERPRICING_WARNING],
        "note": ("Under the round-robin DevelopmentPolicy, changing the twin also changes the decision sequence, so "
                 "differences include the policy's mechanical response. No baseline policy is involved."),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _fmt(a: dict[str, Any], key: str = "mean") -> str:
    if not a or a.get("n", 0) == 0:
        return "–"
    return f"{a[key]:.4g} ± {a['std']:.2g}" if key == "mean" else f"{a[key]:.4g}"


def render_markdown(audit: dict[str, Any]) -> str:
    run = audit["run"]
    L = [
        "# Calibration / synthetic-assumption audit (multi-seed)", "",
        f"> **{audit['label']}** — {audit['statement']}", "",
        f"Scenario `{run['scenario']}` ({run['purpose']}); patience model `{run['patience_model']}`, "
        f"callback model `{run['callback_model']}`; policy {run['policy']}.", "",
        f"Seeds ({len(run['seeds'])}): `{run['seeds']}`. Values are mean ± std across seeds unless stated.", "",
        "## 1. Patience", "",
        f"Parameters: `{audit['patience']['configured']['parameters']}`", "",
        "| type / channel | model | mean | floor | median | p10 | p90 | P(<1) | P(<2) |", "|---|---|---|---|---|---|---|---|---|",
    ]
    for k, v in audit["patience"]["configured"]["by_type_channel"].items():
        L.append(f"| {k} | {v['model']} | {v['configured_mean']} | {v['floor']} | {v['median']} | {v['p10']} | "
                 f"{v['p90']} | {v['p_below_1min']} | {v['p_below_2min']} |")
    L += ["", "Simulated patience pooled over seeds (all instances):", "",
          "| channel | n | mean | median | p10 | p90 | min | max | <1 min | <2 min |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ch, d in audit["patience"]["simulated_by_channel_pooled_over_seeds"].items():
        if d.get("n"):
            L.append(f"| {ch} | {d['n']} | {d['mean']} | {d['median']} | {d['p10']} | {d['p90']} | {d['min']} | "
                     f"{d['max']} | {d['frac_below_1min']} | {d['frac_below_2min']} |")
    o = audit["patience"]["observed_across_seeds"]
    L += ["", f"- abandonment rate {_fmt(o['abandonment_rate'])}; share of abandonments in WAIT "
          f"{_fmt(o['share_of_abandonments_in_wait'])}",
          f"- online wait mean {_fmt(o['online_wait_mean'])} min, p90 {_fmt(o['online_wait_p90'])} min",
          "- abandonment by channel: " + ", ".join(f"{ch} {_fmt(a)}" for ch, a in o["abandonment_rate_by_channel"].items()),
          ""]
    L += _md_flags(audit["patience"]["flags"])

    ai = audit["ai_behavior"]["observed_pooled_over_seeds"]
    L += ["## 2. AI behavior (SYNTHETIC_ASSUMPTION_V1, unchanged)", "",
          "Real values must later come from the Customer Advisor team's observed results.", "",
          "| request type | attempts | correct | visible | false |", "|---|---|---|---|---|"]
    for t, v in ai["by_request_type"].items():
        L.append(f"| {t} | {v['attempts_completed']} | {v['correct_rate']} | {v['visible_failure_rate']} | "
                 f"{v['false_resolution_rate']} |")
    L += ["", "| language | attempts | correct | visible | false |", "|---|---|---|---|---|"]
    for t, v in ai["by_language"].items():
        L.append(f"| {t} | {v['attempts_completed']} | {v['correct_rate']} | {v['visible_failure_rate']} | "
                 f"{v['false_resolution_rate']} |")
    L.append("")
    L += _md_flags(audit["ai_behavior"]["flags"])

    cb = audit["callback"]
    co = cb["observed_across_seeds"]
    L += ["## 3. Callback", "", f"Model `{cb['model']}`; effective `{cb['effective']}`", "",
          "| metric | mean ± std |", "|---|---|"]
    for k, v in co.items():
        if k != "reason_counts_total":
            L.append(f"| {k} | {_fmt(v)} |")
    L += ["", f"Terminal reasons (total over seeds): `{co['reason_counts_total']}`", "",
          "Reward mapping: " + "; ".join(f"{k}: {v}" for k, v in cb["reward_mapping"].items()), ""]
    L += _md_flags(cb["flags"])

    h = audit["human_service"]
    L += ["## 4. Human service", "", "| request type | mean | cv | median | p10 | p90 |", "|---|---|---|---|---|---|"]
    for t, s in h["configured_service_minutes"].items():
        L.append(f"| {t} | {s['mean']} | {s['cv']} | {s['median']} | {s['p10']} | {s['p90']} |")
    L += [""] + [f"- {k}: {_fmt(v)}" for k, v in h["observed_across_seeds"].items()] + ["", f"HIR: {h['hir_definition']}", ""]
    L += _md_flags(h["flags"])
    L += _md_capacity(audit["capacity_check"])
    L += ["## 6. Other structural assumptions", ""] + _md_flags(audit["structural_assumptions"])
    return "\n".join(L) + "\n"


def _md_capacity(cap: dict[str, Any]) -> list[str]:
    L = ["## 5. Advisor capacity check", "", f"`{cap['formula']}`", "",
         "| config | advisors | operating min | capacity (advisor-min) | busy min | utilization |",
         "|---|---|---|---|---|---|"]
    for name in ("normal_day", "load_check"):
        if name in cap:
            c = cap[name]
            L.append(f"| {name} | {c['n_advisors']} | {c['operating_minutes']:.0f} | {c['capacity_minutes']:.0f} | "
                     f"{_fmt(c['busy_minutes'])} | {_fmt(c['utilization'])} |")
    if "explanation" in cap:
        L += ["", cap["explanation"]]
    return L + [""]


def render_paired_markdown(cmp: dict[str, Any]) -> str:
    s = cmp["summary"]
    L = [
        "# B1 vs B1.5 — paired comparison on identical NORMAL seeds", "",
        f"> {cmp['label']}. Policy: {cmp['policy']}.", "",
        f"Seeds (both variants): `{cmp['seeds']}`", "",
        f"Variant overrides: B1 `{cmp['variant_overrides']['b1']}`, B1.5 `{cmp['variant_overrides']['b1_5']}`; "
        "all other parameters identical.", "",
        "| metric | B1 mean ± std | B1.5 mean ± std | paired diff (B1.5 − B1) mean ± std |", "|---|---|---|---|",
    ]
    for k in SCALAR_KEYS:
        L.append(f"| {k} | {_fmt(s[k]['b1'])} | {_fmt(s[k]['b1_5'])} | {_fmt(s[k]['paired_diff_b1_5_minus_b1'])} |")
    L += ["", "Abandonment rate by channel:", "", "| channel | B1 | B1.5 |", "|---|---|---|"]
    for ch, v in s["abandonment_rate_by_channel"].items():
        L.append(f"| {ch} | {_fmt(v['b1'])} | {_fmt(v['b1_5'])} |")
    L += ["", "Callback terminal reasons (total over the 10 seeds):", "", "| reason | B1 | B1.5 |", "|---|---|---|"]
    for r in s["callback_reason_counts_total"]["b1"]:
        L.append(f"| {r} | {s['callback_reason_counts_total']['b1'][r]} | {s['callback_reason_counts_total']['b1_5'][r]} |")
    L += ["", f"Reconciliation exact on every day: {cmp['reconciled']}", ""]
    L += _md_capacity(cmp["capacity_check"])
    L += ["## Warnings", ""] + [f"- {w}" for w in cmp["warnings"]] + ["", f"Note: {cmp['note']}", ""]
    return "\n".join(L)


def _md_flags(flags: list[dict[str, Any]]) -> list[str]:
    if not flags:
        return ["No flags.", ""]
    out = ["Flags:", ""]
    for f in flags:
        ev = f" Evidence: `{f['evidence']}`" if f["evidence"] else ""
        out.append(f"- **[{f['severity'].upper()}] {f['code']}** — {f['message']}{ev}")
    return out + [""]
