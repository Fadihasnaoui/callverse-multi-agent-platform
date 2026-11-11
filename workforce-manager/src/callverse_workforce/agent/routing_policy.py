"""Generic RoutingPolicy interface (V2-ready) and its Phase F implementations.

This is the ONLY agent module that touches PPO internals. Weights are loaded read-only and never updated.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from callverse_workforce.agent.actions import V1_1_REGISTRY, ActionRegistry
from callverse_workforce.agent.state_builder import EncodedState
from callverse_workforce.agent.types import DecisionType
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.observation import RoutingObservation

MASK_VERSION_V1_1 = "V1_1_minimal_masks"
_MODELS: dict[str, Any] = {}


@dataclass(frozen=True)
class PolicyMetadata:
    name: str
    policy_version: str
    model_sha256: str | None
    observation_schema: str
    supported_decision_types: tuple[DecisionType, ...]
    action_registry_version: str
    mask_version: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "policy_version": self.policy_version, "model_sha256": self.model_sha256,
                "observation_schema": self.observation_schema,
                "supported_decision_types": [d.value for d in self.supported_decision_types],
                "action_registry_version": self.action_registry_version, "mask_version": self.mask_version}


class RoutingPolicyAdapter(Protocol):
    def load(self) -> None: ...

    def reset(self) -> None: ...

    def predict(self, state: EncodedState, observation: RoutingObservation, mask: ActionMask) -> str: ...

    def get_policy_metadata(self) -> PolicyMetadata: ...


class PPOPolicy:
    """Frozen PPO E.1 (deterministic masked argmax on the frozen 43-D vector; identical to PPORoutingPolicy)."""

    def __init__(self, path: str | Path, name: str = "e1_s500", registry: ActionRegistry = V1_1_REGISTRY) -> None:
        self.path = str(path)
        self.name = name
        self.registry = registry
        self.model = None
        self._sha: str | None = None

    def load(self) -> None:
        self._sha = hashlib.sha256(Path(self.path).read_bytes()).hexdigest()
        if self.path not in _MODELS:
            import torch

            from callverse_workforce.phase_e.smdp import SMDPMaskablePPO

            torch.set_num_threads(1)
            _MODELS[self.path] = SMDPMaskablePPO.load(self.path, device="cpu")
        self.model = _MODELS[self.path]

    def reset(self) -> None:
        pass

    def predict(self, state: EncodedState, observation: RoutingObservation, mask: ActionMask) -> str:
        if self.model is None:
            raise RuntimeError("PPO model not loaded")
        action, _ = self.model.predict(state.vector, action_masks=np.array(mask.valid, dtype=bool), deterministic=True)
        return self.registry.by_index(int(action)).action_id

    def get_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(self.name, "V1_1", self._sha, "V1_1_43D", (DecisionType.INITIAL,), self.registry.version,
                              MASK_VERSION_V1_1)


class RulePolicyAdapter:
    """Frozen V1.1 RulePolicy: scientific baseline, approved prudence policy, technical fallback."""

    def __init__(self, params: dict[str, Any], registry: ActionRegistry = V1_1_REGISTRY) -> None:
        self.params = dict(params)
        self.registry = registry
        self._policy = None

    def load(self) -> None:
        from callverse_workforce.baselines.policies import make_policy

        self._policy = make_policy("RulePolicy", self.params, 0)

    def reset(self) -> None:
        pass

    def predict(self, state: EncodedState, observation: RoutingObservation, mask: ActionMask) -> str:
        if self._policy is None:
            self.load()
        return self._policy.predict(observation, mask).value

    def get_policy_metadata(self) -> PolicyMetadata:
        sha = hashlib.sha256(repr(sorted(self.params.items())).encode()).hexdigest()
        return PolicyMetadata("RulePolicy", "V1_1", sha, "V1_1_semantic", (DecisionType.INITIAL,), self.registry.version,
                              MASK_VERSION_V1_1)
