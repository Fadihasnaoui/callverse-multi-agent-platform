"""SupervisorController: evidence-based KEEP_PPO vs RECOMMEND_TEMPORARY_PRUDENCE_MODE (policy-agnostic).

It consumes KPIs, the emergency and the what-if summary only; it never assumes a number of actions or a state
dimension.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from callverse_workforce.agent.config import SupervisionConfig
from callverse_workforce.agent.emergency import Emergency

KEEP_PPO = "KEEP_PPO"
RECOMMEND = "RECOMMEND_TEMPORARY_PRUDENCE_MODE"


def confidence_passes(whatif: Any, rule: str) -> bool:
    if rule == "t80":
        return whatif.t80_lower > 0.0
    if rule == "share60":
        return whatif.share_positive >= 0.6
    raise ValueError(rule)


@dataclass(frozen=True)
class Recommendation:
    rec_id: str
    kind: str
    emergency_id: str | None
    whatif_id: str | None
    estimated_gain: float | None
    criteria: dict[str, bool]


class SupervisorController:
    def __init__(self, cfg: SupervisionConfig) -> None:
        self.cfg = cfg
        self.n = 0

    def recommend(self, emergency: Emergency | None, whatif: Any | None) -> Recommendation:
        self.n += 1
        valid_whatif = whatif is not None and whatif.status == "OK" and whatif.k == self.cfg.k_futures
        criteria = {
            "valid_emergency": emergency is not None,
            "valid_whatif": valid_whatif,
            "gain_above_g_min": bool(valid_whatif and whatif.estimated_gain > self.cfg.g_min),
            "confidence": bool(valid_whatif and confidence_passes(whatif, self.cfg.confidence)),
        }
        kind = RECOMMEND if all(criteria.values()) else KEEP_PPO
        return Recommendation(f"R{self.n:03d}", kind, emergency.emergency_id if emergency else None,
                              whatif.whatif_id if whatif is not None else None,
                              whatif.estimated_gain if valid_whatif else None, criteria)
