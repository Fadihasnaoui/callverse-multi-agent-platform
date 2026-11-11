"""V2 action space, decision types and the V2 mask (spec §2, §3). The V1.1 mask (routing/mask.py) is reused only
for its AI and CALLBACK reasons; the WAIT rule uses the dedicated V2 ``wait_count`` (U1)."""

from __future__ import annotations

from dataclasses import dataclass, field

from callverse_workforce.agent.actions import ActionRegistry, ActionSpec
from callverse_workforce.agent.types import DecisionType
from callverse_workforce.domain.enums import RoutingAction, Urgency
from callverse_workforce.domain.observation import RoutingObservation
from callverse_workforce.routing.mask import ServiceAvailability, compute_action_mask

INITIAL, POST = DecisionType.INITIAL, DecisionType.POST_AI_FAILURE
AI, HUMAN, HUMAN_NORMAL, HUMAN_HIGH, WAIT, CALLBACK = "AI", "HUMAN", "HUMAN_NORMAL", "HUMAN_HIGH", "WAIT", "CALLBACK"
V2_ACTIONS: tuple[str, ...] = (AI, HUMAN, HUMAN_NORMAL, HUMAN_HIGH, WAIT, CALLBACK)
CONTEXT_ACTIONS = {INITIAL: (AI, HUMAN), POST: (HUMAN_NORMAL, HUMAN_HIGH, WAIT, CALLBACK)}
MAX_WAITS = 3  # frozen wait.max_waits (checked against the config at twin construction)

V2_REGISTRY = ActionRegistry("V2_POSTFAIL", tuple(
    ActionSpec(a, i, tuple(c for c, acts in CONTEXT_ACTIONS.items() if a in acts), f"submit_{a.lower()}", a)
    for i, a in enumerate(V2_ACTIONS)))


@dataclass(frozen=True)
class V2Mask:
    decision_type: DecisionType
    valid: tuple[bool, ...]  # aligned with V2_ACTIONS
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.valid) != len(V2_ACTIONS) or not any(self.valid):
            raise ValueError("invalid V2 mask")

    def is_valid(self, action: str) -> bool:
        return self.valid[V2_ACTIONS.index(action)]

    @property
    def valid_actions(self) -> tuple[str, ...]:
        return tuple(a for a, ok in zip(V2_ACTIONS, self.valid) if ok)


def v2_mask(decision_type: DecisionType, observation: RoutingObservation, wait_count: int,
            availability: ServiceAvailability, mask_config, wait_config) -> V2Mask:
    v1 = compute_action_mask(observation, availability, mask_config, wait_config)  # AI / CALLBACK reasons only
    reasons: dict[str, list[str]] = {a: [] for a in V2_ACTIONS}
    for a in V2_ACTIONS:
        if a not in CONTEXT_ACTIONS[decision_type]:
            reasons[a].append(f"not_in_{decision_type.value}")
    reasons[AI] += list(v1.reasons.get(RoutingAction.AI.value, ()))
    if decision_type is POST:
        reasons[AI].append("post_ai_failure_no_repeat")
    reasons[CALLBACK] += list(v1.reasons.get(RoutingAction.CALLBACK.value, ()))
    if observation.urgency_pred is Urgency.HIGH and observation.deadline_remaining_pred < mask_config.urgent_deadline_threshold_minutes:
        reasons[WAIT].append("urgent_short_deadline")
    if wait_count >= wait_config.max_waits:  # U1: the dedicated counter, never routing_attempt_count
        reasons[WAIT].append("max_waits_reached")
    valid = tuple(not reasons[a] for a in V2_ACTIONS)
    return V2Mask(decision_type, valid, {a: tuple(r) for a, r in reasons.items() if r})
