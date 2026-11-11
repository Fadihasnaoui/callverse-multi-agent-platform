"""Routing policy interface and the temporary development policy."""

from __future__ import annotations

from typing import Protocol

from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import ACTION_ORDER, RoutingAction
from callverse_workforce.domain.observation import RoutingObservation


class RoutingPolicy(Protocol):
    """Every routing policy (development, baselines, PPO) consumes the same inputs."""

    def reset(self) -> None: ...

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction: ...


class DevelopmentPolicy:
    """Deterministic round-robin over valid actions.

    NOT a baseline and not meant to route well. It exists only to exercise
    every routing path of the Digital Twin (AI, HUMAN, WAIT, CALLBACK).
    """

    def __init__(self) -> None:
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction:
        n = len(ACTION_ORDER)
        start = self._counter % n
        self._counter += 1
        for offset in range(n):
            action = ACTION_ORDER[(start + offset) % n]
            if action_mask.is_valid(action):
                return action
        raise RuntimeError("action mask has no valid action")  # unreachable: ActionMask forbids it
