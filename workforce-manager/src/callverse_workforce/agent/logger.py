"""DecisionLogger: structured, reproducible audit records (JSONL export; wall-clock fields excluded from hashes)."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

from callverse_workforce.agent.memory import MemoryStore

WALL_CLOCK_FIELDS = frozenset({"latency_ms", "duration_ms", "wall_seconds"})


class DecisionLogger:
    def __init__(self, memory: MemoryStore, keep_decisions: bool = True) -> None:
        self.memory = memory
        self.keep_decisions = keep_decisions

    def decision(self, percept, action_id: str, mode: str, policy_used: str, guardrail: str | None, latency_ms: float,
                 extra: dict[str, Any] | None = None) -> None:
        """extra: fields of a decision taken under a human-approved intervention (effective / final mask, intervention)."""
        if self.keep_decisions:
            self.memory.record("DecisionRecord", percept.timestamp, decision_id=percept.decision_id, contact_id=percept.contact_id,
                               decision_type=percept.decision_type.value, mode=mode, policy_used=policy_used, action=action_id,
                               mask=list(percept.action_mask.valid), guardrail=guardrail, latency_ms=latency_ms, **(extra or {}))

    def outcomes(self, events) -> None:
        for e in events:
            self.memory.record("OutcomeRecord", e.timestamp, contact_id=e.contact_id, event_type=e.event_type)


def _clean(rec: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in WALL_CLOCK_FIELDS}


def audit_hash(memory: MemoryStore) -> str:
    h = hashlib.sha256()
    for r in memory.audit:
        h.update(json.dumps(_clean(r), sort_keys=True, default=str).encode())
    return h.hexdigest()


def export_jsonl(memory: MemoryStore, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in memory.audit:
            fh.write(json.dumps(r, sort_keys=True, default=str) + "\n")
