"""Frozen V2 baselines and continuation policies (spec §6, §7.1a). Deterministic, observable inputs only."""

from __future__ import annotations

from typing import Any

from callverse_workforce.agent.types import DecisionType
from callverse_workforce.baselines.policies import make_policy
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import Urgency
from callverse_workforce.v2_postfail import actions as A

B1_ID, B2_ID = "B1-LEGACY-v1", "B2-CTX-v1"
B2_QUEUE_MIN = 2  # frozen RulePolicy queue_max (congestion definition)
B2_DEADLINE_MIN = 180.0  # frozen mask.urgent_deadline_threshold_minutes (> 120-min callback window)
INITIAL_POLICIES = ("RulePolicy", "AIFirstPolicy")
POST_POLICIES = ("B1", "B2")


def b1(dp) -> str:
    """B1-LEGACY-v1: POST_AI_FAILURE -> HUMAN_NORMAL (exactly the V1.1 escalation)."""
    return A.HUMAN_NORMAL


def b2(dp) -> str:
    """B2-CTX-v1 (frozen order): HIGH urgency -> HUMAN_HIGH; congested & callback valid & deadline >= 180 -> CALLBACK;
    else HUMAN_NORMAL. Never WAIT."""
    o = dp.observation.base
    if o.urgency_pred is Urgency.HIGH:
        return A.HUMAN_HIGH
    if (dp.mask.is_valid(A.CALLBACK) and o.humans_available == 0 and o.human_queue_size >= B2_QUEUE_MIN
            and o.deadline_remaining_pred >= B2_DEADLINE_MIN):
        return A.CALLBACK
    return A.HUMAN_NORMAL


POST = {"B1": b1, "B2": b2}


class ContinuationPolicy:
    """One row of spec §7.1a: INITIAL by a frozen V1.1 policy (outputs only AI/HUMAN), POST by B1 or B2."""

    def __init__(self, initial: str, post: str, rule_params: dict[str, Any]) -> None:
        if initial not in INITIAL_POLICIES or post not in POST_POLICIES:
            raise ValueError((initial, post))
        self.initial, self.post = initial, post
        self._initial = make_policy(initial, rule_params if initial == "RulePolicy" else {}, 0)
        self._post = POST[post]

    @property
    def spec(self) -> tuple[str, str]:
        return (self.initial, self.post)

    def act(self, dp) -> str:
        if dp.decision_type is DecisionType.INITIAL:
            # V2 INITIAL offers only AI / HUMAN: present the frozen policy with the equivalent V1.1 mask view
            view = ActionMask((dp.mask.is_valid(A.AI), True, False, False))
            a = self._initial.predict(dp.observation.base, view).value
            if a not in (A.AI, A.HUMAN):  # pragma: no cover - impossible under the view
                raise RuntimeError(a)
            return a
        return self._post(dp)
