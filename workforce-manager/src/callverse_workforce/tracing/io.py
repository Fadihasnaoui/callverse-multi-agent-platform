"""Persist simulation results as structured JSON / JSONL files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from callverse_workforce.utils.serialization import to_jsonable

if TYPE_CHECKING:
    from callverse_workforce.twin.result import SimulationResult


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(to_jsonable(row)) + "\n")


def save_result(result: SimulationResult, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": out / "summary.json",
        "transitions": out / "transitions.jsonl",
        "decisions": out / "decisions.jsonl",
        "events": out / "events.jsonl",
        "contacts": out / "contacts.jsonl",
    }
    with open(paths["summary"], "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(result.metrics), fh, indent=2)
    _write_jsonl(paths["transitions"], result.transitions)
    _write_jsonl(paths["decisions"], result.decisions)
    _write_jsonl(paths["events"], result.events)
    _write_jsonl(paths["contacts"], result.contacts)
    return paths
