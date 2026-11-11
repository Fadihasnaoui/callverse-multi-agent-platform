"""Temporary human-approved restriction applied on top of the effective (business-rule) mask.

Restored from the human-in-the-loop work of 2026-11-04 (stash@{0}), unchanged in substance: during an approved
intervention AI stays allowed only for low-risk requests. The restriction only ever REMOVES AI; it never enables an
action the effective mask forbids, and an empty result is an explicit RoutingFailure (no arbitrary action).
"""

from __future__ import annotations

from dataclasses import dataclass

from callverse_workforce.agent.guardrails import RoutingFailure
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import RoutingAction, Sensitivity, Urgency

AI_MIN_SCORE = 0.80
RESTRICTION = "low_risk_ai_only"
RESTRICTION_TEXT = "IA autorisée seulement si score IA ≥ 80 %, sensibilité faible et urgence non élevée"
REASON = "human_approved_temporary_ai_restriction"


@dataclass(frozen=True)
class MaskRestrictionResult:
    mask: ActionMask
    changed: bool
    ai_blocked: bool
    reason: str | None


def low_risk(observation) -> bool:
    return (float(observation.ai_resolution_score) >= AI_MIN_SCORE
            and observation.sensitivity_pred is Sensitivity.LOW
            and observation.urgency_pred is not Urgency.HIGH)


def restrict_for_low_risk_ai(observation, mask: ActionMask) -> MaskRestrictionResult:
    """Keep AI only for low-risk requests; never enable an action masked by the business rules."""
    ai = RoutingAction.AI.index
    if not mask.valid[ai] or low_risk(observation):
        return MaskRestrictionResult(mask, False, False, None)
    valid = list(mask.valid)
    valid[ai] = False
    if not any(valid):
        raise RoutingFailure("temporary restriction would leave no valid routing action")
    reasons = dict(mask.reasons)
    reasons[RoutingAction.AI.value] = tuple(reasons.get(RoutingAction.AI.value, ())) + (REASON,)
    return MaskRestrictionResult(ActionMask(tuple(valid), reasons), True, True, RESTRICTION_TEXT)


class RestrictedPPOPolicy:
    """Same frozen PPO weights and 43-D state, predicting under the restricted mask: the intervention branch of the
    what-if (the policy the manager would actually approve), never a different routing model."""

    def __init__(self, inner) -> None:
        self.inner = inner

    def load(self) -> None:
        self.inner.load()

    def reset(self) -> None:
        self.inner.reset()

    def predict(self, state, observation, mask: ActionMask) -> str:
        return self.inner.predict(state, observation, restrict_for_low_risk_ai(observation, mask).mask)

    def get_policy_metadata(self):
        return self.inner.get_policy_metadata()
