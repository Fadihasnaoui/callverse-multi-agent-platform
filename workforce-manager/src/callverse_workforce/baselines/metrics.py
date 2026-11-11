"""Per-day policy metrics (simulator side: may use truth for MEASUREMENT, never for decisions)."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np

from callverse_workforce.domain.enums import (
    ACTION_ORDER,
    CallbackReason,
    Channel,
    ContactState,
    EventType,
    LanguageDialect,
    Outcome,
)
from callverse_workforce.reward.breakdown import REWARD_COMPONENTS
from callverse_workforce.twin.result import SimulationResult

REWARD_KEYS = REWARD_COMPONENTS + ("total",)
_TERMINAL_REWARD_EVENTS = frozenset({
    EventType.CONTACT_RESOLVED, EventType.CALLBACK_DROPOUT, EventType.CALLBACK_FAILED,
    EventType.CALLBACK_WINDOW_EXPIRED, EventType.DEADLINE_MISSED, EventType.UNFINISHED_AT_HORIZON,
})


def _rate(num: float, den: float) -> float:
    return num / den if den else 0.0


def _group_metrics(contacts) -> dict[str, float]:
    waits = np.array([c.online_wait_minutes for c in contacts], dtype=float)
    n = len(contacts)
    return {
        "instances": n,
        "resolution_rate": _rate(sum(c.outcome.is_correct_resolution for c in contacts), n),
        "hir_instance": _rate(sum(c.human_touched for c in contacts), n),
        "wait_mean": float(waits.mean()) if n else 0.0,
        "wait_p90": float(np.percentile(waits, 90)) if n else 0.0,
        "abandonment_rate": _rate(sum(c.outcome is Outcome.ABANDONED for c in contacts), n),
    }


def policy_day_metrics(result: SimulationResult) -> dict[str, Any]:
    m = result.metrics
    contacts = result.contacts
    n = len(contacts)
    decisions = result.decisions
    n_dec = len(decisions)
    events = Counter(e.event_type for e in result.events)
    outcomes = Counter(c.outcome for c in contacts)

    roots: dict[str, bool] = {}
    for c in contacts:
        roots[c.root_contact_id] = roots.get(c.root_contact_id, False) or c.outcome.is_correct_resolution

    actions = Counter(d.action for d in decisions)
    action_share = {a.value: _rate(actions.get(a, 0), n_dec) for a in ACTION_ORDER}
    valid_hist = Counter(d.n_valid_actions for d in decisions)
    ai_completed = events[EventType.AI_RESOLVED] + events[EventType.AI_VISIBLE_FAILURE] + events[EventType.AI_FALSE_RESOLUTION]
    wait_limit = sum(1 for d in decisions if "max_waits_reached" in d.action_mask.reasons.get("WAIT", ()))
    ab_in_wait = sum(1 for e in result.events
                     if e.event_type is EventType.ABANDONMENT and e.data["from_state"] is ContactState.WAITING)
    cb = m["callback"]
    cb_ids = {c.contact_id for c in contacts if c.callback_created_at is not None}
    cb_event_reward = sum(float(e.data.get("reward", 0.0)) for e in result.events
                          if e.contact_id in cb_ids and e.event_type in _TERMINAL_REWARD_EVENTS)
    waits = np.array([c.online_wait_minutes for c in contacts], dtype=float)
    reward = result.episode_reward.to_dict()

    by_language = defaultdict(list)
    by_channel = defaultdict(list)
    for c in contacts:
        by_language[c.language_dialect.value].append(c)
        by_channel[c.channel.value].append(c)

    return {
        "seed": result.seed,
        "total_reward": result.episode_reward.total,
        "instances": n,
        "roots": len(roots),
        "resolution_rate_root": _rate(sum(roots.values()), len(roots)),
        "resolution_rate_instance": _rate(sum(c.outcome.is_correct_resolution for c in contacts), n),
        "hir_root": m["human_intervention_rate"]["root_level"],
        "hir_instance": m["human_intervention_rate"]["instance_level"],
        "human_routing_rate": m["human_routing_rate"],
        "human_busy_minutes": m["human_busy_minutes"],
        "human_utilization": m["capacity"]["utilization"],
        "decisions": n_dec,
        "action_share": action_share,
        "dominant_action": max(action_share, key=action_share.get) if n_dec else None,
        "dominant_action_share": max(action_share.values()) if n_dec else 0.0,
        "abandonment_rate": _rate(outcomes.get(Outcome.ABANDONED, 0), n),
        "missed_deadline_rate": _rate(outcomes.get(Outcome.MISSED_DEADLINE, 0), n),
        "unfinished_rate": _rate(outcomes.get(Outcome.UNFINISHED, 0), n),
        "online_wait_mean": float(waits.mean()) if n else 0.0,
        "online_wait_p90": float(np.percentile(waits, 90)) if n else 0.0,
        "ai_attempts_completed": ai_completed,
        "false_ai_rate": _rate(events[EventType.AI_FALSE_RESOLUTION], ai_completed),
        "ai_visible_failure_rate": _rate(events[EventType.AI_VISIBLE_FAILURE], ai_completed),
        "ai_escalations": events[EventType.AI_ESCALATION],
        "wait_limit_count": wait_limit,
        "abandonments_in_wait": ab_in_wait,
        "callbacks_scheduled": cb["scheduled"],
        "callback_reasons": cb["reasons"],
        "callback_success": cb["reasons"][CallbackReason.CALLBACK_SUCCESS.value],
        "callback_dropout": cb["reasons"][CallbackReason.CUSTOMER_DROPOUT.value],
        "callback_max_attempts": cb["reasons"][CallbackReason.MAX_ATTEMPTS.value],
        "callback_window_expired": cb["reasons"][CallbackReason.WINDOW_EXPIRED.value],
        "callback_unfinished": cb["reasons"][CallbackReason.UNFINISHED.value],
        "callback_success_rate": cb["reason_rates"][CallbackReason.CALLBACK_SUCCESS.value],
        "callback_human_minutes": cb["human_minutes"],
        # terminal event rewards of callback contacts (+2V success, -3V dropout, -1 failures, -4V deadline, -2V
        # unfinished) plus the callback-pending time cost
        "callback_reward_contribution": cb_event_reward + result.episode_reward.callback_pending,
        "mean_valid_actions": m["mask_freedom"]["mean_valid_actions"],
        "valid_action_pct": {str(k): _rate(valid_hist.get(k, 0), n_dec) for k in (4, 3, 2, 1)},
        "reward": {k: reward[k] for k in REWARD_KEYS},
        "reconciled": m["reconciliation"]["balanced"],
        "fairness": {
            "language": {lang.value: _group_metrics(by_language[lang.value]) for lang in LanguageDialect},
            "channel": {ch.value: _group_metrics(by_channel[ch.value]) for ch in Channel},
        },
    }
