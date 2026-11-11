"""GuardrailManager: action validity, mask consistency, safe default."""

from __future__ import annotations

from callverse_workforce.agent.actions import ActionRegistry
from callverse_workforce.agent.ppo_engine import TechnicalFailure
from callverse_workforce.agent.types import Percept
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import RoutingAction

SAFE_DEFAULT = RoutingAction.HUMAN.value  # HUMAN is never masked in V1.1


class RoutingFailure(RuntimeError):
    """No safe action can be chosen (PPO and the fallback both failed, or the effective mask is empty). The decision is
    NOT executed: the caller must stop routing this day and report the failure; no action is ever chosen arbitrarily."""


class GuardrailManager:
    def __init__(self, registry: ActionRegistry) -> None:
        self.registry = registry
        self.mask_mismatches = 0
        self.violations = 0

    def validate(self, action_id: object, percept: Percept, mask: ActionMask | None = None) -> str:
        """Raise TechnicalFailure on an invalid policy output (unregistered, wrong decision type, masked). ``mask`` is the
        effective mask of the decision (default: the simulator mask of the percept)."""
        mask = percept.action_mask if mask is None else mask
        if not isinstance(action_id, str) or action_id not in self.registry:
            self.violations += 1
            raise TechnicalFailure(f"invalid policy output {action_id!r}")
        spec = self.registry.get(action_id)
        if percept.decision_type not in spec.valid_decision_types or not mask.valid[spec.ppo_index]:
            self.violations += 1
            raise TechnicalFailure(f"policy output {action_id} is masked / not allowed")
        return action_id

    def effective_mask(self, percept: Percept, recomputed: ActionMask) -> tuple[ActionMask, bool]:
        """Authoritative mask of a decision: an action is allowed only if BOTH the simulator mask of the percept and the
        business-rule mask recomputed by the agent allow it (conservative intersection). Returns (mask, mismatch).
        Raises RoutingFailure if no action is left."""
        provided = tuple(percept.action_mask.valid)
        valid = tuple(bool(a and b) for a, b in zip(provided, recomputed.valid))
        mismatch = tuple(recomputed.valid) != provided
        if mismatch:
            self.mask_mismatches += 1
        if not any(valid):
            raise RoutingFailure(f"masque effectif vide (simulateur {provided}, règles métier {tuple(recomputed.valid)})")
        reasons = dict(percept.action_mask.reasons)
        for k, v in dict(getattr(recomputed, "reasons", {}) or {}).items():
            reasons[k] = tuple(dict.fromkeys(tuple(reasons.get(k, ())) + tuple(v)))
        return (ActionMask(valid, reasons) if valid != provided else percept.action_mask), mismatch

    def check_mask(self, percept: Percept, recomputed: ActionMask) -> bool:
        ok = tuple(recomputed.valid) == tuple(percept.action_mask.valid)
        if not ok:
            self.mask_mismatches += 1
        return ok

    def safe_action(self, action_id: str, percept: Percept) -> str:
        """For non-PPO policies: keep the action if valid, else the never-masked safe default."""
        try:
            return self.validate(action_id, percept)
        except TechnicalFailure:
            return SAFE_DEFAULT
