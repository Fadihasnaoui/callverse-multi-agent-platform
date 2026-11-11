"""Shared human pool with HIGH priority and the NORMAL aging guarantee (spec §4).

Service order when an advisor is free (non-preemptive):
    0 human-shortage holds (frozen V1.1 mechanism, priority -1)
    1 AGED NORMAL (cumulative human-queue wait >= A_age), FIFO by the instant the aging limit was reached
    2 HIGH, FIFO by queue entry
    3 NORMAL, FIFO by queue entry
    4 CALLBACK, FIFO by queue entry
The order is recomputed at every grant opportunity (simpy triggers on each request and each release), so aging
takes effect exactly when it matters. Without HIGH requests the order equals V1.1's PriorityResource order.
"""

from __future__ import annotations

import itertools

import simpy

HOLD, HIGH, NORMAL, CALLBACK = "HOLD", "HIGH", "NORMAL", "CALLBACK"
RANK = {HOLD: 0, "AGED": 1, HIGH: 2, NORMAL: 3, CALLBACK: 4}
EPS = 1e-9


class PoolRequest(simpy.resources.resource.Request):
    def __init__(self, resource, cls: str, acc_before: float = 0.0, t_entry: float | None = None, owner: str | None = None) -> None:
        self.cls = cls
        self.owner = owner
        # t_entry: original queue-entry time of a RESTORED contact (keeps its FIFO position); live requests use now
        self.t_request = resource._env.now if t_entry is None else t_entry
        self.seq = next(resource._seq)
        # instant the cumulative human-queue wait reaches A_age (NORMAL only)
        self.aging_at = self.t_request + max(0.0, resource.a_age - acc_before) if cls == NORMAL else float("inf")
        super().__init__(resource)


class AgingHumanPool(simpy.Resource):
    def __init__(self, env: simpy.Environment, capacity: int, a_age: float) -> None:
        self.a_age = a_age
        self._seq = itertools.count()
        super().__init__(env, capacity)
        self.aging_overrides = 0  # AGED NORMAL granted while a HIGH request was waiting
        self.aging_violations = 0  # HIGH granted while an AGED NORMAL was waiting (must stay 0: hard check)
        self.grants: list[tuple[float, str, bool]] = []

    def request(self, priority: int | None = None, cls: str | None = None, acc_before: float = 0.0,
                t_entry: float | None = None, owner: str | None = None) -> PoolRequest:
        if cls is None:  # frozen V1.1 call sites: shortage holds use priority -1
            cls = HOLD if priority is not None and priority < 0 else (CALLBACK if priority == 1 else NORMAL)
        return PoolRequest(self, cls, acc_before, t_entry, owner)

    def key(self, r: PoolRequest, now: float):
        if r.cls == NORMAL and now + EPS >= r.aging_at:
            return (RANK["AGED"], r.aging_at, r.seq)
        return (RANK[r.cls], r.t_request, r.seq)

    def _trigger_put(self, get_event) -> None:
        now = self._env.now
        if len(self.put_queue) > 1:
            self.put_queue.sort(key=lambda r: self.key(r, now))
        if self.put_queue and len(self.users) < self.capacity:
            head = self.put_queue[0]
            aged_waiting = any(r.cls == NORMAL and now + EPS >= r.aging_at for r in self.put_queue)
            high_waiting = any(r.cls == HIGH for r in self.put_queue[1:])
            if head.cls == HIGH and aged_waiting:
                self.aging_violations += 1
            is_aged = head.cls == NORMAL and now + EPS >= head.aging_at
            if is_aged and high_waiting:
                self.aging_overrides += 1
            self.grants.append((now, head.cls, is_aged))
        super()._trigger_put(get_event)
