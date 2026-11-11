"""Phase B2 non-RL baseline routing policies.

Every policy implements ``RoutingPolicy``: it sees ONLY the canonical
``RoutingObservation`` and the ``ActionMask`` (no ground truth, no future, no
scenario). Parameters are immutable after construction, so a policy frozen after
validation cannot drift during evaluation.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import ACTION_ORDER, RoutingAction
from callverse_workforce.domain.observation import RoutingObservation

A = RoutingAction

# Deterministic fallback order used whenever a preferred action is masked. HUMAN is never masked in V1,
# so in practice the first fallback is always HUMAN.
FALLBACK_ORDER: tuple[RoutingAction, ...] = (A.HUMAN, A.CALLBACK, A.WAIT, A.AI)


def _first_valid(mask: ActionMask, preferences: tuple[RoutingAction, ...]) -> RoutingAction:
    for action in preferences:
        if mask.is_valid(action):
            return action
    return mask.valid_actions[0]  # unreachable while HUMAN is always valid


def policy_rng_seed(day_seed: int, policy_name: str) -> np.random.SeedSequence:
    """Policy randomness is keyed separately from every Digital Twin stream."""
    return np.random.SeedSequence(day_seed, spawn_key=(zlib.crc32(f"policy:{policy_name}".encode()),))


class RandomPolicy:
    """Uniform choice among valid actions, from a dedicated reproducible RNG."""

    name = "RandomPolicy"

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)
        self.reset()

    @property
    def params(self) -> dict[str, Any]:
        return {"seed": self._seed}

    def reset(self) -> None:
        self._rng = np.random.Generator(np.random.PCG64(policy_rng_seed(self._seed, self.name)))

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction:
        valid = action_mask.valid_actions
        return valid[int(self._rng.integers(len(valid)))]


class AIFirstPolicy:
    """AI whenever valid; otherwise HUMAN -> CALLBACK -> WAIT (untuned)."""

    name = "AIFirstPolicy"

    @property
    def params(self) -> dict[str, Any]:
        return {"order": ["AI", "HUMAN", "CALLBACK", "WAIT"]}

    def reset(self) -> None:
        pass

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction:
        return _first_valid(action_mask, (A.AI, A.HUMAN, A.CALLBACK, A.WAIT))


@dataclass(frozen=True)
class ThresholdPolicy:
    """AI if valid and ``ai_resolution_score >= theta``; otherwise HUMAN (frozen fallback)."""

    theta: float
    name: str = "ThresholdPolicy"

    @property
    def params(self) -> dict[str, Any]:
        return {"theta": self.theta}

    def reset(self) -> None:
        pass

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction:
        if action_mask.is_valid(A.AI) and observation.ai_resolution_score >= self.theta:
            return A.AI
        return A.HUMAN


CONGESTED_ACTIONS = ("CALLBACK", "WAIT", "AI")


@dataclass(frozen=True)
class RulePolicy:
    """Compact interpretable rule (3 tunable parameters):

    1. confident AI:  AI valid and score >= theta                 -> AI
    2. humans OK:     a human is free, or human queue < queue_max -> HUMAN
    3. congested:     ``congested_action`` if valid, else HUMAN
       (CALLBACK / WAIT are masked for urgent short-deadline contacts, so those go to HUMAN.)
    """

    theta: float
    queue_max: int
    congested_action: str
    name: str = "RulePolicy"

    def __post_init__(self) -> None:
        if self.congested_action not in CONGESTED_ACTIONS:
            raise ValueError(f"congested_action must be one of {CONGESTED_ACTIONS}")

    @property
    def params(self) -> dict[str, Any]:
        return {"theta": self.theta, "queue_max": self.queue_max, "congested_action": self.congested_action}

    def reset(self) -> None:
        pass

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction:
        if action_mask.is_valid(A.AI) and observation.ai_resolution_score >= self.theta:
            return A.AI
        if observation.humans_available > 0 or observation.human_queue_size < self.queue_max:
            return A.HUMAN
        return _first_valid(action_mask, (RoutingAction(self.congested_action), A.HUMAN))


@dataclass(frozen=True)
class FixedActionPolicy:
    """Always the primary action when valid; else the documented ``FALLBACK_ORDER``."""

    action: str
    name: str = "BestFixedPolicy"

    @property
    def params(self) -> dict[str, Any]:
        return {"action": self.action, "fallback_order": [a.value for a in FALLBACK_ORDER]}

    def reset(self) -> None:
        pass

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction:
        primary = RoutingAction(self.action)
        return _first_valid(action_mask, (primary,) + tuple(a for a in FALLBACK_ORDER if a is not primary))


BASELINE_NAMES = ("RandomPolicy", "AIFirstPolicy", "ThresholdPolicy", "RulePolicy", "BestFixedPolicy")


def make_policy(name: str, params: dict[str, Any], day_seed: int):
    """Instantiate a frozen baseline for one evaluation day."""
    if name == "RandomPolicy":
        return RandomPolicy(seed=day_seed)
    if name == "AIFirstPolicy":
        return AIFirstPolicy()
    if name == "ThresholdPolicy":
        return ThresholdPolicy(theta=float(params["theta"]))
    if name == "RulePolicy":
        return RulePolicy(theta=float(params["theta"]), queue_max=int(params["queue_max"]),
                          congested_action=str(params["congested_action"]))
    if name == "BestFixedPolicy":
        return FixedActionPolicy(action=str(params["action"]))
    raise KeyError(name)


assert set(ACTION_ORDER) == set(FALLBACK_ORDER)
