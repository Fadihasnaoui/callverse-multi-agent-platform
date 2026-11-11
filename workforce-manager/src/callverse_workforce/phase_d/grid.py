"""Pre-registered Twin V1.1 sweep grid (see docs/twin_v1_1_rationale.md). Immutable."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path

from callverse_workforce.config import SimulationConfig, apply_overrides
from callverse_workforce.config.loader import PROJECT_ROOT

HANDOFF: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)
MULTIPLIER: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4)
HANGUP: tuple[float, ...] = (0.0, 0.05)
LABEL = "SYNTHETIC_ASSUMPTION_V1_1"
RATIONALE_PATH = PROJECT_ROOT / "docs" / "twin_v1_1_rationale.md"


@dataclass(frozen=True, order=True)
class Cell:
    handoff: float
    multiplier: float
    hangup: float

    @property
    def cell_id(self) -> str:
        return f"h{self.handoff:g}_m{self.multiplier:g}_p{self.hangup:g}"

    @property
    def severity(self) -> float:
        return self.handoff / 4.0 + (1.0 - self.multiplier) / 0.6 + self.hangup / 0.05

    @property
    def order_key(self) -> tuple[float, float, float, float]:
        return (self.severity, self.handoff, -self.multiplier, self.hangup)

    def to_dict(self) -> dict[str, float | str]:
        return {"cell_id": self.cell_id, "handoff_overhead_minutes": self.handoff,
                "frustration_patience_multiplier": self.multiplier,
                "immediate_hangup_prob_after_ai_failure": self.hangup, "severity": self.severity}


GRID: tuple[Cell, ...] = tuple(Cell(h, m, p) for p in HANGUP for h in HANDOFF for m in MULTIPLIER)
V1_CELL = Cell(0.0, 1.0, 0.0)
MAX_CELL = Cell(4.0, 0.4, 0.05)
MID_CELL = Cell(2.0, 0.6, 0.0)
MIXTURE_CELLS: tuple[Cell, ...] = (Cell(0.0, 1.0, 0.0), Cell(1.0, 0.8, 0.0), Cell(2.0, 0.6, 0.05), Cell(4.0, 0.4, 0.05))
assert len(GRID) == 32 and len(set(GRID)) == 32 and V1_CELL in GRID and MAX_CELL in GRID
assert all(c in GRID for c in MIXTURE_CELLS)


def cell_config(base: SimulationConfig, cell: Cell) -> SimulationConfig:
    """V1.1 config for one cell. The V1 cell returns the unchanged base config (exact V1)."""
    if cell == V1_CELL:
        return base
    return apply_overrides(base, {
        "scenario.name": f"v1_1_{cell.cell_id}",
        "scenario.purpose": "TWIN_V1_1_SWEEP",
        "v1_1.handoff_overhead_minutes": cell.handoff,
        "v1_1.frustration_patience_multiplier": cell.multiplier,
        "v1_1.immediate_hangup_prob_after_ai_failure": cell.hangup,
    })


def grid_manifest() -> list[dict[str, float | str]]:
    return [c.to_dict() for c in GRID]


def rationale_hash(path: Path = RATIONALE_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
