"""PPODecisionEngine: calls the active RoutingPolicy; any failure becomes a TechnicalFailure."""

from __future__ import annotations

import time

from callverse_workforce.agent.routing_policy import RoutingPolicyAdapter
from callverse_workforce.agent.state_builder import EncodedState
from callverse_workforce.agent.types import Percept


class TechnicalFailure(RuntimeError):
    pass


class PPODecisionEngine:
    def __init__(self) -> None:
        self.errors = 0
        self.last_error: str | None = None
        self.latencies: list[float] = []

    def decide(self, policy: RoutingPolicyAdapter, state: EncodedState | None, percept: Percept, mask=None) -> str:
        """``mask``: effective mask of the decision (default: the simulator mask of the percept)."""
        t0 = time.perf_counter()
        try:
            action = policy.predict(state, percept.observation, percept.action_mask if mask is None else mask)
        except Exception as exc:  # noqa: BLE001 - every inference failure is a technical failure
            self.errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise TechnicalFailure(self.last_error) from exc
        self.latencies.append((time.perf_counter() - t0) * 1000)
        if len(self.latencies) > 1000:
            del self.latencies[:500]
        return action
