"""Center-level reward breakdown."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

# Signed reward contributions; their sum is the scalar reward.
REWARD_COMPONENTS: tuple[str, ...] = (
    "resolution",
    "human_busy",
    "ai_busy",
    "online_waiting",
    "callback_pending",
    "abandonment",
    "missed_deadline",
    "false_ai",
    "callback_failure",
    "unfinished",
)


@dataclass
class RewardBreakdown:
    """Reward components accrued over one interval (or a whole episode).

    The first block holds signed reward contributions. The second block holds
    the underlying exposures (minutes and counts) for diagnostics; they do not
    enter ``total``.
    """

    resolution: float = 0.0
    human_busy: float = 0.0
    ai_busy: float = 0.0
    online_waiting: float = 0.0
    callback_pending: float = 0.0
    abandonment: float = 0.0
    missed_deadline: float = 0.0
    false_ai: float = 0.0
    callback_failure: float = 0.0
    unfinished: float = 0.0

    human_busy_minutes: float = 0.0
    ai_busy_minutes: float = 0.0
    online_waiting_contact_minutes: float = 0.0
    callback_pending_contact_minutes: float = 0.0
    n_resolutions: int = 0
    n_abandonments: int = 0
    n_missed_deadlines: int = 0
    n_false_ai: int = 0
    n_callback_failures: int = 0
    n_unfinished: int = 0

    @property
    def total(self) -> float:
        return sum(getattr(self, name) for name in REWARD_COMPONENTS)

    def add(self, other: RewardBreakdown) -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def copy(self) -> RewardBreakdown:
        return RewardBreakdown(**asdict(self))

    def to_dict(self) -> dict[str, float]:
        data = asdict(self)
        data["total"] = self.total
        return data
