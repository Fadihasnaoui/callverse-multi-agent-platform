"""V2 observation contract V2_INITIAL_POSTFAIL (spec §5): a SEPARATE schema. The frozen V1.1 43-D encoder is reused
unchanged for the 28 semantic features and extended by decision_type one-hot (2), wait_count / 3 and escalated."""

from __future__ import annotations

import numpy as np

from callverse_workforce.agent.state_builder import ObservationContract
from callverse_workforce.agent.types import DecisionType
from callverse_workforce.encoding import ObservationEncoder
from callverse_workforce.v2_postfail.actions import MAX_WAITS

V2_DIM = ObservationEncoder.dim + 4


class V2Encoder:
    version = "obs_encoder_v2_postfail_1"
    dim = V2_DIM

    def __init__(self) -> None:
        self._v11 = ObservationEncoder()

    def encode(self, obs) -> np.ndarray:  # obs: V2Observation (or a RoutingObservation-compatible percept field)
        base = obs.base if hasattr(obs, "base") else obs
        dt = getattr(obs, "decision_type", DecisionType.INITIAL)
        extra = [1.0 if dt is DecisionType.INITIAL else 0.0, 1.0 if dt is DecisionType.POST_AI_FAILURE else 0.0,
                 getattr(obs, "wait_count", 0) / MAX_WAITS, 1.0 if getattr(obs, "escalated", False) else 0.0]
        return np.concatenate([self._v11.encode(base), np.asarray(extra, dtype=np.float32)]).astype(np.float32)


V2_CONTRACT = ObservationContract("V2_POSTFAIL", "V2_INITIAL_POSTFAIL", V2_DIM, DecisionType.INITIAL, V2Encoder.version)
