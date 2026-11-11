"""Transport-independent agent API (dashboard-ready). No HTTP framework is installed, so the HTTP transport is
deferred to the dashboard / integration phase; each method maps 1:1 to the Stage-1 endpoint contract."""

from __future__ import annotations

from typing import Any

from callverse_workforce.agent.manager import ExternalManager
from callverse_workforce.agent.types import Percept

ENDPOINTS = {
    ("GET", "/agent/status"): "status", ("GET", "/agent/kpis"): "kpis", ("GET", "/agent/decisions"): "decisions",
    ("GET", "/agent/emergencies"): "emergencies", ("GET", "/agent/memory"): "memory", ("GET", "/agent/whatif/{id}"): "whatif",
    ("POST", "/agent/route"): "route", ("POST", "/agent/manager-response"): "manager_response",
}


class AgentAPI:
    def __init__(self, agent) -> None:
        self.agent = agent

    def status(self) -> dict[str, Any]:
        s = self.agent.status()
        snaps = self.agent.memory.kpi_snapshots
        return s | {"last_kpi_snapshot": dict(snaps[-1]) if snaps else None}

    def kpis(self, limit: int = 20) -> dict[str, Any]:
        snaps = list(self.agent.memory.kpi_snapshots)
        return {"latest": snaps[-1] if snaps else None, "history": snaps[-limit:]}

    def decisions(self, limit: int = 50) -> list[dict[str, Any]]:
        return self.agent.memory.records("DecisionRecord", limit)

    def emergencies(self) -> list[dict[str, Any]]:
        mem = self.agent.memory
        out = []
        for e in mem.records("EmergencyRecord"):
            eid = e["emergency_id"]
            link = lambda kind: [r for r in mem.records(kind) if r.get("emergency_id") == eid]
            out.append(e | {"whatif": link("WhatIfRecord"), "recommendation": link("RecommendationRecord"),
                            "manager_request": link("ManagerRequest")})
        return out

    def memory(self, kind: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return self.agent.memory.records(kind, limit)

    def whatif(self, whatif_id: str) -> dict[str, Any] | None:
        w = self.agent.whatifs.get(whatif_id)
        return None if w is None else w.to_dict()

    def route(self, percept: Percept) -> dict[str, Any]:
        return self.agent.route(percept).__dict__

    def manager_response(self, request_id: str, decision: str, t: float) -> dict[str, Any]:
        if not isinstance(self.agent.manager, ExternalManager):
            raise RuntimeError("manager responses are accepted only with an ExternalManager")
        self.agent.manager.respond(request_id, decision, t)
        return {"accepted": True, "request_id": request_id, "decision": decision}
