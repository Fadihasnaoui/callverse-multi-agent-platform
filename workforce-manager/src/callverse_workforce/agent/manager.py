"""Human-in-the-loop by exception. The manager validates a POLICY-LEVEL intervention only, never a contact.

``SimulatedManager`` is a deterministic evidence-following manager proxy used for reproducible scientific
evaluation. It does not model realistic human psychology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

APPROVE, REJECT = "APPROVE", "REJECT"
EXPIRE = "EXPIRE"  # no answer before the proposal's deadline (dashboard inbox): never an approval


@dataclass(frozen=True)
class ManagerRequest:
    request_id: str
    sent_t: float
    emergency_id: str
    whatif_id: str
    recommendation: str
    estimated_gain: float
    confidence_pass: bool
    valid_emergency: bool
    valid_whatif: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    restriction: str | None = None  # exactly what an approval activates (None: legacy RulePolicy prudence)
    duration_minutes: float | None = None


@dataclass(frozen=True)
class ManagerDecision:
    request_id: str
    decision: str
    response_t: float
    profile: str


class ManagerInterface(Protocol):
    def notify_manager(self, request: ManagerRequest) -> None: ...

    def poll(self, now: float) -> ManagerDecision | None: ...


class SimulatedManager:
    PROFILES = {"primary": (5.0, 1.0), "slow": (30.0, 1.0), "conservative": (5.0, 2.0)}

    def __init__(self, profile: str, g_min: float, approval_delay: float | None = None) -> None:
        delay, factor = self.PROFILES[profile]
        self.profile = profile
        self.delay = delay if approval_delay is None or profile == "slow" else approval_delay
        self.factor = factor
        self.g_min = g_min
        self._pending: list[ManagerDecision] = []

    def decide(self, r: ManagerRequest) -> str:
        ok = (r.valid_emergency and r.valid_whatif and r.confidence_pass and r.estimated_gain > self.factor * self.g_min)
        return APPROVE if ok else REJECT

    def notify_manager(self, request: ManagerRequest) -> None:
        self._pending.append(ManagerDecision(request.request_id, self.decide(request), request.sent_t + self.delay, self.profile))

    def poll(self, now: float) -> ManagerDecision | None:
        for i, d in enumerate(self._pending):
            if d.response_t <= now:
                return self._pending.pop(i)
        return None


class ExternalManager:
    """Real deployment: responses arrive through the API (``AgentAPI.manager_response``)."""

    def __init__(self) -> None:
        self.requests: list[ManagerRequest] = []
        self._responses: list[ManagerDecision] = []

    def notify_manager(self, request: ManagerRequest) -> None:
        self.requests.append(request)

    def respond(self, request_id: str, decision: str, t: float) -> None:
        if decision not in (APPROVE, REJECT):
            raise ValueError(decision)
        if request_id not in {r.request_id for r in self.requests}:
            raise LookupError(f"unknown manager request {request_id!r}")
        self._responses.append(ManagerDecision(request_id, decision, t, "external"))

    def poll(self, now: float) -> ManagerDecision | None:
        return self._responses.pop(0) if self._responses else None
