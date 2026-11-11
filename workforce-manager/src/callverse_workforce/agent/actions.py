"""Central ActionRegistry: the only place that knows the policy action set (V1.1 mapping reproduced exactly)."""

from __future__ import annotations

from dataclasses import dataclass

from callverse_workforce.agent.types import DecisionType
from callverse_workforce.domain.enums import ACTION_ORDER


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    ppo_index: int
    valid_decision_types: tuple[DecisionType, ...]
    handler: str
    log_name: str


class ActionRegistry:
    def __init__(self, version: str, specs: tuple[ActionSpec, ...]) -> None:
        if sorted(s.ppo_index for s in specs) != list(range(len(specs))):
            raise ValueError("ppo indices must be 0..n-1")
        if len({s.action_id for s in specs}) != len(specs):
            raise ValueError("duplicate action ids")
        self.version = version
        self.specs = tuple(sorted(specs, key=lambda s: s.ppo_index))
        self._by_id = {s.action_id: s for s in self.specs}

    @property
    def n_actions(self) -> int:
        return len(self.specs)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(s.action_id for s in self.specs)

    def by_index(self, index: int) -> ActionSpec:
        return self.specs[index]

    def get(self, action_id: str) -> ActionSpec:
        if action_id not in self._by_id:
            raise KeyError(f"action {action_id!r} not registered in {self.version}")
        return self._by_id[action_id]

    def __contains__(self, action_id: object) -> bool:
        return action_id in self._by_id


V1_1_REGISTRY = ActionRegistry("V1_1", tuple(
    ActionSpec(a.value, a.index, (DecisionType.INITIAL,), f"submit_{a.value.lower()}", a.value) for a in ACTION_ORDER))
assert V1_1_REGISTRY.action_ids == tuple(a.value for a in ACTION_ORDER)
