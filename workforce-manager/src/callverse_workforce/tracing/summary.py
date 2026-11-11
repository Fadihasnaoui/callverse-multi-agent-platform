"""Aggregate metrics for one simulation run."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from callverse_workforce.domain.enums import ACTION_ORDER, CallbackReason, EventType, Outcome, RoutingAction
from callverse_workforce.domain.records import ContactSummary

if TYPE_CHECKING:
    from callverse_workforce.twin.result import SimulationResult

HIR_DEFINITION = (
    "A contact instance is human-touched if a human advisor actually served it at least once: "
    "service started after direct HUMAN routing, service started after AI visible-failure escalation, "
    "or a callback reached and served by an advisor. Multiple touches count once. Routing to HUMAN "
    "that ends before service (abandonment, missed deadline, end of day) and unanswered callback "
    "dials do not count. Root-level HIR groups instances by root_contact_id."
)

# Terminal outcome semantics (each instance receives exactly one).
OUTCOME_SEMANTICS = {
    Outcome.RESOLVED_AI.value: "correctly resolved by AI",
    Outcome.RESOLVED_HUMAN.value: "correctly resolved by an advisor (direct or after AI escalation)",
    Outcome.RESOLVED_CALLBACK.value: "correctly resolved by an advisor during a callback",
    Outcome.FALSE_AI_DISCOVERED.value: "AI appeared to resolve but was wrong; detected in-day or settled at horizon",
    Outcome.FALSE_AI_UNDISCOVERED.value: "false AI resolution left unsettled (only if settle_at_horizon=false)",
    Outcome.ABANDONED.value: "customer gave up while waiting online (patience exhausted)",
    Outcome.MISSED_DEADLINE.value: "deadline passed while waiting (online or callback queue); service in progress may finish",
    Outcome.CALLBACK_FAILED.value: "MAX_ATTEMPTS: callback dial attempts exhausted without reaching the customer (-1)",
    Outcome.CALLBACK_WINDOW_EXPIRED.value: "WINDOW_EXPIRED: callback not completed within its window during operating time (-1)",
    Outcome.CALLBACK_DROPOUT.value: "CUSTOMER_DROPOUT: customer gave up while waiting for the callback (-3V, a lost customer)",
    Outcome.UNFINISHED.value: "still in the system at the horizon (waiting, in service, or callback pending)",
}


def human_intervention_rates(contacts: Sequence[ContactSummary]) -> dict[str, Any]:
    """Instance-level and root-level Human Intervention Rate (distinct from human busy minutes)."""
    touched_instances = sum(1 for c in contacts if c.human_touched)
    roots: dict[str, bool] = {}
    for c in contacts:
        roots[c.root_contact_id] = roots.get(c.root_contact_id, False) or c.human_touched
    touched_roots = sum(roots.values())
    return {
        "definition": HIR_DEFINITION,
        "instance_level": touched_instances / len(contacts) if contacts else 0.0,
        "root_level": touched_roots / len(roots) if roots else 0.0,
        "touched_instances": touched_instances,
        "total_instances": len(contacts),
        "touched_roots": touched_roots,
        "total_roots": len(roots),
    }


def reconcile_outcomes(contacts: Sequence[ContactSummary], arrival_events: int) -> dict[str, Any]:
    """Every created contact instance must have exactly one terminal outcome."""
    counts = Counter(c.outcome for c in contacts)
    by_outcome = {o.value: counts.get(o, 0) for o in Outcome}
    total_terminal = sum(by_outcome.values())
    exogenous = sum(1 for c in contacts if c.recontact_depth == 0)
    return {
        "instances_created": len(contacts),
        "arrival_events": arrival_events,
        "exogenous_instances": exogenous,
        "recontact_instances": len(contacts) - exogenous,
        "terminal_by_outcome": by_outcome,
        "sum_terminal": total_terminal,
        "balanced": total_terminal == len(contacts) == arrival_events and None not in counts,
        "semantics": OUTCOME_SEMANTICS,
    }


def summarize(result: SimulationResult) -> dict[str, Any]:
    events = Counter(e.event_type for e in result.events)
    outcomes = Counter(c.outcome for c in result.contacts)
    actions = Counter(d.action for d in result.decisions)
    n_valid = [d.n_valid_actions for d in result.decisions]
    waits = np.array([c.online_wait_minutes for c in result.contacts], dtype=float)
    deltas = np.array([t.delta_t_minutes for t in result.transitions], dtype=float)
    n_decisions = len(result.decisions)

    def _stat(arr: np.ndarray, fn) -> float:
        return float(fn(arr)) if arr.size else 0.0

    return {
        "seed": result.seed,
        "scenario": result.scenario_name,
        "horizon_minutes": result.horizon_minutes,
        "generated_contacts": len(result.contacts),
        "resolved_contacts": sum(n for o, n in outcomes.items() if o.is_correct_resolution),
        "outcomes": {o.value: outcomes.get(o, 0) for o in Outcome},
        "human_service_starts": events[EventType.HUMAN_SERVICE_STARTED] + events[EventType.CALLBACK_REACHED],
        "ai_attempts": events[EventType.AI_ATTEMPT_STARTED],
        "ai_correct_resolutions": events[EventType.AI_RESOLVED],
        "ai_visible_failures": events[EventType.AI_VISIBLE_FAILURE],
        "ai_escalations": events[EventType.AI_ESCALATION],
        "false_ai_resolutions": events[EventType.AI_FALSE_RESOLUTION],
        "false_ai_discovered_in_day": events[EventType.FALSE_AI_DISCOVERED],
        "false_ai_settled_at_horizon": events[EventType.FALSE_AI_SETTLED_AT_HORIZON],
        "wait_decisions": actions.get(RoutingAction.WAIT, 0),
        "wait_reevaluations": events[EventType.WAIT_REEVALUATION],
        "callbacks_scheduled": events[EventType.CALLBACK_SCHEDULED],
        "callback_attempts_failed": events[EventType.CALLBACK_ATTEMPT_FAILED],
        "abandonments": outcomes.get(Outcome.ABANDONED, 0),
        "missed_deadlines": outcomes.get(Outcome.MISSED_DEADLINE, 0),
        "unfinished_contacts": outcomes.get(Outcome.UNFINISHED, 0),
        "recontacts_created": events[EventType.RECONTACT_SCHEDULED],
        "recontacts_not_created": events[EventType.RECONTACT_NOT_CREATED],
        "human_intervention_rate": human_intervention_rates(result.contacts),
        "human_busy_minutes": result.episode_reward.human_busy_minutes,
        "human_routing_rate": (actions.get(RoutingAction.HUMAN, 0) / n_decisions) if n_decisions else 0.0,
        "callback": callback_metrics(result.contacts),
        "capacity": capacity_metrics(result),
        "reconciliation": reconcile_outcomes(result.contacts, events[EventType.ARRIVAL]),
        "stress_indicators": _stress_indicators(result),
        "total_reward": result.episode_reward.total,
        "reward_breakdown": result.episode_reward.to_dict(),
        "online_wait_minutes": {
            "mean": _stat(waits, np.mean),
            "p90": _stat(waits, lambda a: np.percentile(a, 90)),
            "max": _stat(waits, np.max),
        },
        "decisions": n_decisions,
        "action_distribution": {a.value: actions.get(a, 0) for a in ACTION_ORDER},
        "mask_freedom": {
            "mean_valid_actions": float(np.mean(n_valid)) if n_valid else 0.0,
            "valid_action_histogram": {str(k): v for k, v in sorted(Counter(n_valid).items())},
            "forced_decision_share": (sum(1 for n in n_valid if n == 1) / n_decisions) if n_decisions else 0.0,
            "action_valid_share": {
                a.value: (sum(1 for d in result.decisions if d.action_mask.is_valid(a)) / n_decisions)
                if n_decisions
                else 0.0
                for a in ACTION_ORDER
            },
        },
        "smdp_delta_t_minutes": {
            "mean": _stat(deltas, np.mean),
            "max": _stat(deltas, np.max),
            "zero_share": float(np.mean(deltas == 0)) if deltas.size else 0.0,
        },
    }



def _stress_indicators(result: SimulationResult) -> dict[str, Any]:
    """Peaks of observable congestion seen at decision instants, plus end-of-day backlog."""
    obs = [t.state for t in result.transitions]
    unfinished_states = Counter(
        e.data["state"].value for e in result.events if e.event_type is EventType.UNFINISHED_AT_HORIZON
    )
    return {
        "max_human_queue_size": max((o.human_queue_size for o in obs), default=0),
        "max_oldest_human_wait": max((o.oldest_human_wait for o in obs), default=0.0),
        "max_wait_queue_size": max((o.wait_queue_size for o in obs), default=0),
        "max_callbacks_pending": max((o.callbacks_pending for o in obs), default=0),
        "max_oldest_callback_age": max((o.oldest_callback_age for o in obs), default=0.0),
        "unfinished_by_state": dict(sorted(unfinished_states.items())),
    }


def callback_metrics(contacts: Sequence[ContactSummary]) -> dict[str, Any]:
    """Callback terminal reasons and timing over contacts that entered the callback path."""
    cb = [c for c in contacts if c.callback_created_at is not None]
    reasons = Counter(c.callback_reason for c in cb)
    n = len(cb)
    attempts = sum(c.callback_attempts for c in cb)
    reached = sum(1 for c in cb if c.callback_completed_at is not None and c.human_touched)
    pending = np.array([c.callback_pending_minutes for c in cb], dtype=float)
    return {
        "scheduled": n,
        "dial_attempts": attempts,
        "answered": reached,
        "per_dial_answer_rate": reached / attempts if attempts else 0.0,
        "reasons": {r.value: reasons.get(r, 0) for r in CallbackReason},
        "reason_rates": {r.value: (reasons.get(r, 0) / n if n else 0.0) for r in CallbackReason},
        "pending_minutes_median": float(np.median(pending)) if n else 0.0,
        "pending_minutes_p90": float(np.percentile(pending, 90)) if n else 0.0,
        "human_minutes": float(sum(c.callback_human_minutes for c in contacts)),
    }


def capacity_metrics(result: SimulationResult) -> dict[str, Any]:
    """capacity_minutes = advisors x available operating minutes (advisors are available the whole horizon)."""
    capacity = result.n_advisors * result.horizon_minutes
    busy = result.episode_reward.human_busy_minutes
    return {
        "n_advisors": result.n_advisors,
        "operating_minutes": result.horizon_minutes,
        "capacity_minutes": capacity,
        "busy_minutes": busy,
        "utilization": busy / capacity if capacity else 0.0,
    }
