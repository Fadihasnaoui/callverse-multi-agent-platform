"""Phase C load levels and AI-score noise levels (diagnostic configurations only).

Load levels add nested Poisson demand layers on top of the untouched NORMAL day
(same advisors, AI, patience, callbacks, reward, service times, request mix,
deadlines, masks):

    NORMAL       base day
    NORMAL_X1_3  base + layer 0 (0.3 x base profile)            ~ +30% demand
    NORMAL_X1_5  base + layer 0 + layer 1 (0.2 x base profile)  ~ +50% demand

Noise levels scale ONLY ``intake_noise.ai_score_noise_std`` (the observed
``ai_resolution_score``): LOW 0.5x, MEDIUM 1.0x (V1), HIGH 1.5x.
Label: SYNTHETIC_DIAGNOSTIC_NOISE_LEVEL.
"""

from __future__ import annotations

from callverse_workforce.config import SimulationConfig, apply_overrides

LOAD_LEVELS: dict[str, list[float]] = {
    "NORMAL": [],
    "NORMAL_X1_3": [0.3],
    "NORMAL_X1_5": [0.3, 0.2],
}
NOISE_LEVELS: dict[str, float] = {"LOW": 0.5, "MEDIUM": 1.0, "HIGH": 1.5}
NOISE_LABEL = "SYNTHETIC_DIAGNOSTIC_NOISE_LEVEL"
DIAGNOSTIC_PURPOSE = "HEADROOM_DIAGNOSTIC_ONLY"


def headroom_config(base: SimulationConfig, load: str, noise: str = "MEDIUM") -> SimulationConfig:
    if base.scenario.extra_load_layers:
        raise ValueError("base config must be the plain NORMAL day")
    if load == "NORMAL" and noise == "MEDIUM":
        return base
    overrides: dict[str, object] = {
        "scenario.name": f"headroom_{load.lower()}_{noise.lower()}",
        "scenario.purpose": DIAGNOSTIC_PURPOSE,
        "scenario.extra_load_layers": list(LOAD_LEVELS[load]),
        "intake_noise.ai_score_noise_std": base.intake_noise.ai_score_noise_std * NOISE_LEVELS[noise],
    }
    return apply_overrides(base, overrides)
