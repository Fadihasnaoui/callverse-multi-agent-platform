"""RoutingExecutor: ActionRegistry dispatch to substrate handlers (no per-action branching here)."""

from __future__ import annotations

from functools import partial
from typing import Any, Callable, Mapping

from callverse_workforce.agent.actions import ActionRegistry
from callverse_workforce.domain.enums import RoutingAction


class RoutingExecutor:
    def __init__(self, registry: ActionRegistry, handlers: Mapping[str, Callable[[str], Any]]) -> None:
        missing = [s.handler for s in registry.specs if s.handler not in handlers]
        if missing:
            raise ValueError(f"no handler for {missing}")
        self.registry = registry
        self.handlers = dict(handlers)
        self.executed = 0

    def execute(self, action_id: str) -> Any:
        spec = self.registry.get(action_id)
        self.executed += 1
        return self.handlers[spec.handler](spec.action_id)


def _step(env, action_id: str) -> Any:
    return env.step(RoutingAction(action_id))


def simulation_handlers(registry: ActionRegistry, env) -> dict[str, Callable[[str], Any]]:
    """Digital-Twin substrate: every registered action is submitted to the live RoutingEnvironment."""
    return {s.handler: partial(_step, env) for s in registry.specs}
