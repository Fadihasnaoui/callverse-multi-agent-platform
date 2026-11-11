"""Structured simulator event log (simulator view: may include hidden truth)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from callverse_workforce.domain.enums import EventType
from callverse_workforce.utils.serialization import to_jsonable


@dataclass(frozen=True)
class EventRecord:
    seq: int
    time: float
    event_type: EventType
    contact_id: str | None = None
    decision_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "time": self.time,
            "event_type": self.event_type.value,
            "contact_id": self.contact_id,
            "decision_id": self.decision_id,
            "data": to_jsonable(self.data),
        }


class EventLog:
    def __init__(self) -> None:
        self._records: list[EventRecord] = []

    def log(
        self,
        time: float,
        event_type: EventType,
        contact_id: str | None = None,
        decision_id: str | None = None,
        **data: Any,
    ) -> EventRecord:
        record = EventRecord(len(self._records), time, event_type, contact_id, decision_id, data)
        self._records.append(record)
        return record

    def of_type(self, event_type: EventType) -> list[EventRecord]:
        return [r for r in self._records if r.event_type is event_type]

    def for_contact(self, contact_id: str) -> list[EventRecord]:
        return [r for r in self._records if r.contact_id == contact_id]

    @property
    def records(self) -> list[EventRecord]:
        return list(self._records)

    def __iter__(self) -> Iterator[EventRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)
