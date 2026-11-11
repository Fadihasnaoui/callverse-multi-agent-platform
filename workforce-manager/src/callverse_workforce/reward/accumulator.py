"""Center-level reward accounting.

Event-based terms (resolution, abandonment, ...) are recorded once, at the
moment the event happens. Time-based terms (busy minutes, waiting
contact-minutes, callback-pending contact-minutes) are integrated over all
contacts in the center. ``drain`` hands the interval's breakdown to the current
SMDP transition and starts a new interval.
"""

from __future__ import annotations

from typing import Callable

from callverse_workforce.config.models import RewardConfig
from callverse_workforce.reward.breakdown import RewardBreakdown

# observer(delta, dt): dt == 0 for event terms, dt > 0 for a time-integrated interval ending "now"
RewardObserver = Callable[[RewardBreakdown, float], None]


class RewardAccumulator:
    def __init__(self, coefficients: RewardConfig) -> None:
        self.c = coefficients
        self._interval = RewardBreakdown()
        self.episode = RewardBreakdown()
        self.observer: RewardObserver | None = None  # read-only tap (e.g. discounted headroom returns)

    # --- time-integrated terms -------------------------------------------------
    def accrue_time(
        self,
        dt: float,
        n_human_busy: int,
        n_ai_busy: int,
        n_online_waiting: int,
        n_callback_pending: int,
    ) -> None:
        if dt <= 0:
            return
        delta = RewardBreakdown(
            human_busy_minutes=n_human_busy * dt,
            ai_busy_minutes=n_ai_busy * dt,
            online_waiting_contact_minutes=n_online_waiting * dt,
            callback_pending_contact_minutes=n_callback_pending * dt,
        )
        delta.human_busy = -self.c.human_busy_per_minute * delta.human_busy_minutes
        delta.ai_busy = -self.c.ai_busy_per_minute * delta.ai_busy_minutes
        delta.online_waiting = -self.c.online_waiting_per_minute * delta.online_waiting_contact_minutes
        delta.callback_pending = -self.c.callback_pending_per_minute * delta.callback_pending_contact_minutes
        self._add(delta, dt)

    # --- event terms (each called exactly once per event) ---------------------
    def resolution(self, value_weight: float) -> float:
        amount = self.c.resolution_multiplier * value_weight
        self._add(RewardBreakdown(resolution=amount, n_resolutions=1))
        return amount

    def abandonment(self, value_weight: float) -> float:
        amount = -self.c.abandonment_multiplier * value_weight
        self._add(RewardBreakdown(abandonment=amount, n_abandonments=1))
        return amount

    def missed_deadline(self, value_weight: float) -> float:
        amount = -self.c.missed_deadline_multiplier * value_weight
        self._add(RewardBreakdown(missed_deadline=amount, n_missed_deadlines=1))
        return amount

    def false_ai_detected(self, value_weight: float) -> float:
        """Reverse the earlier resolution credit and add the extra penalty: ``-(2V + 4)``."""
        amount = -(self.c.resolution_multiplier * value_weight + self.c.false_ai_extra_penalty)
        self._add(RewardBreakdown(false_ai=amount, n_false_ai=1))
        return amount

    def callback_failure(self) -> float:
        amount = -self.c.callback_failure_penalty
        self._add(RewardBreakdown(callback_failure=amount, n_callback_failures=1))
        return amount

    def unfinished(self, value_weight: float) -> float:
        amount = -self.c.unfinished_multiplier * value_weight
        self._add(RewardBreakdown(unfinished=amount, n_unfinished=1))
        return amount

    # --- interval handling -----------------------------------------------------
    def drain(self) -> RewardBreakdown:
        interval, self._interval = self._interval, RewardBreakdown()
        return interval

    def _add(self, delta: RewardBreakdown, dt: float = 0.0) -> None:
        self._interval.add(delta)
        self.episode.add(delta)
        if self.observer is not None:
            self.observer(delta, dt)
