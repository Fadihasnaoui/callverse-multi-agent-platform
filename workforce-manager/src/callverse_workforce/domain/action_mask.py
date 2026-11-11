"""Action mask value object."""

from __future__ import annotations

from dataclasses import dataclass, field

from callverse_workforce.domain.enums import ACTION_ORDER, RoutingAction


@dataclass(frozen=True)
class ActionMask:
    """Validity of each ``RoutingAction`` (in ``ACTION_ORDER``) plus why actions were masked."""

    valid: tuple[bool, bool, bool, bool]
    reasons: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.valid) != len(ACTION_ORDER):
            raise ValueError("mask must have one entry per routing action")
        if not any(self.valid):
            raise ValueError("action mask must leave at least one valid action")

    def is_valid(self, action: RoutingAction) -> bool:
        return self.valid[action.index]

    @property
    def valid_actions(self) -> tuple[RoutingAction, ...]:
        return tuple(a for a, ok in zip(ACTION_ORDER, self.valid) if ok)

    @property
    def n_valid(self) -> int:
        return sum(self.valid)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": {a.value: ok for a, ok in zip(ACTION_ORDER, self.valid)},
            "n_valid": self.n_valid,
            "reasons": {k: list(v) for k, v in self.reasons.items()},
        }
