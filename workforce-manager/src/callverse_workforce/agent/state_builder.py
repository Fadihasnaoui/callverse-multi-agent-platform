"""StateBuilder: the frozen V1.1 encoder (single implementation, shared with the research environment)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from callverse_workforce.agent.types import ENABLED_DECISION_TYPES, DecisionType, Percept
from callverse_workforce.encoding import ObservationEncoder


@dataclass(frozen=True)
class ObservationContract:
    policy_version: str
    observation_schema: str
    dim: int
    decision_type: DecisionType
    encoder_version: str


V1_1_CONTRACT = ObservationContract("V1_1", "V1_1_43D", ObservationEncoder.dim, DecisionType.INITIAL, ObservationEncoder.version)


@dataclass(frozen=True)
class EncodedState:
    vector: np.ndarray
    contract: ObservationContract


class StateBuilder:
    def __init__(self, contract: ObservationContract = V1_1_CONTRACT, encoder: ObservationEncoder | None = None) -> None:
        self.contract = contract
        self.encoder = encoder or ObservationEncoder()

    def build(self, percept: Percept) -> EncodedState:
        if percept.decision_type not in ENABLED_DECISION_TYPES or percept.decision_type is not self.contract.decision_type:
            raise NotImplementedError(f"decision_type {percept.decision_type} is not enabled (V2)")
        vec = self.encoder.encode(percept.observation)
        if vec.shape != (self.contract.dim,):
            raise ValueError("observation contract violated")
        return EncodedState(vec, self.contract)
