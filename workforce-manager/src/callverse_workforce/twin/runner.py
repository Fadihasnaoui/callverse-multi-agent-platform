"""Run one full simulated day with a routing policy."""

from __future__ import annotations

from typing import Sequence

from callverse_workforce.config.models import SimulationConfig
from callverse_workforce.domain.contact import GroundTruthContact
from callverse_workforce.routing.policy import RoutingPolicy
from callverse_workforce.tracing.summary import summarize
from callverse_workforce.twin.environment import RoutingEnvironment
from callverse_workforce.twin.result import SimulationResult


def run_simulation(
    config: SimulationConfig,
    policy: RoutingPolicy,
    seed: int | None = None,
    contacts: Sequence[GroundTruthContact] | None = None,
) -> SimulationResult:
    env = RoutingEnvironment(config, contacts=contacts)
    policy.reset()
    decision = env.reset(seed)
    while decision is not None:
        action = policy.predict(decision.observation, decision.action_mask)
        decision = env.step(action).next_decision
    result = env.result()
    result.metrics = summarize(result)
    return result
