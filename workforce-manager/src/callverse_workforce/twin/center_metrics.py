"""Sliding-window center indicators built from observable events only."""

from __future__ import annotations

from collections import deque


class SlidingWindow:
    """Timestamped values; ``values(now)`` returns those within ``[now - window, now]``."""

    def __init__(self, window_minutes: float) -> None:
        self.window = window_minutes
        self._items: deque[tuple[float, float]] = deque()

    def add(self, time: float, value: float = 1.0) -> None:
        self._items.append((time, value))

    def _evict(self, now: float) -> None:
        while self._items and self._items[0][0] < now - self.window:
            self._items.popleft()

    def count(self, now: float) -> int:
        self._evict(now)
        return len(self._items)

    def total(self, now: float) -> float:
        self._evict(now)
        return sum(v for _, v in self._items)


class CenterIndicators:
    """Recent arrival rate, AI failure rate, abandonment rate and service level."""

    def __init__(
        self,
        arrival_window: float,
        ai_failure_window: float,
        abandonment_window: float,
        service_level_window: float,
        service_level_threshold: float,
    ) -> None:
        self.arrivals = SlidingWindow(arrival_window)
        self.ai_completions = SlidingWindow(ai_failure_window)
        self.ai_failures_observed = SlidingWindow(ai_failure_window)  # visible failures + discovered false
        self.abandon_events = SlidingWindow(abandonment_window)
        self.answered_events = SlidingWindow(abandonment_window)
        self.sl_answered = SlidingWindow(service_level_window)  # value 1 if within threshold else 0
        self.sl_abandoned = SlidingWindow(service_level_window)
        self.sl_threshold = service_level_threshold

    def record_arrival(self, t: float) -> None:
        self.arrivals.add(t)

    def record_ai_completion(self, t: float) -> None:
        self.ai_completions.add(t)

    def record_ai_failure_observed(self, t: float) -> None:
        self.ai_failures_observed.add(t)

    def record_answered(self, t: float, online_wait: float) -> None:
        """An online contact started service (human or AI)."""
        self.answered_events.add(t)
        self.sl_answered.add(t, 1.0 if online_wait <= self.sl_threshold else 0.0)

    def record_abandonment(self, t: float) -> None:
        self.abandon_events.add(t)
        self.sl_abandoned.add(t)

    def arrival_rate_per_hour(self, now: float) -> float:
        window = min(self.arrivals.window, now) if now > 0 else self.arrivals.window
        return self.arrivals.count(now) * 60.0 / window

    def ai_failure_rate(self, now: float) -> float:
        done = self.ai_completions.count(now)
        return min(1.0, self.ai_failures_observed.count(now) / done) if done else 0.0

    def abandonment_rate(self, now: float) -> float:
        ab, ans = self.abandon_events.count(now), self.answered_events.count(now)
        return ab / (ab + ans) if ab + ans else 0.0

    def service_level(self, now: float) -> float:
        n = self.sl_answered.count(now) + self.sl_abandoned.count(now)
        return self.sl_answered.total(now) / n if n else 1.0
