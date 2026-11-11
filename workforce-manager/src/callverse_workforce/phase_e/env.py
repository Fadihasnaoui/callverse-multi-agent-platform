"""Gymnasium environment over the frozen V1.1 Digital Twin (one episode = one simulated day).

* observation: the frozen 43-D ``ObservationEncoder`` vector (float32); nothing else
* action: Discrete(4) = AI, HUMAN, WAIT, CALLBACK (canonical order); the minimal V1 masks via ``action_masks()``
* reward: center-level raw reward of the transition / ``REWARD_SCALE`` (fixed, never normalised)
* info: delta_t_minutes, raw_reward, reward_breakdown, day metadata (logging only), termination reason
"""

from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np

from callverse_workforce.config import SimulationConfig
from callverse_workforce.domain.enums import ACTION_ORDER
from callverse_workforce.encoding import OBSERVATION_DIM, ObservationEncoder
from callverse_workforce.phase_e.scenarios import DaySpec, day_config
from callverse_workforce.twin.environment import RoutingEnvironment

REWARD_SCALE = 100.0  # scaled_reward = raw_reward / 100 (frozen for every PPO variant and run)
_ENCODER = ObservationEncoder()


class CallVerseRoutingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base: SimulationConfig, day_source: Callable[[], DaySpec]) -> None:
        super().__init__()
        self.base = base
        self.day_source = day_source
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(ACTION_ORDER))
        self._configs: dict[tuple[str, str], SimulationConfig] = {}
        self._env: RoutingEnvironment | None = None
        self._dp = None
        self.day: DaySpec | None = None
        self.day_raw_reward = 0.0

    def _config(self, day: DaySpec) -> SimulationConfig:
        key = (day.family, day.cell.cell_id)
        if key not in self._configs:
            self._configs[key] = day_config(self.base, day.family, day.cell)
        return self._configs[key]

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.day = (options or {}).get("day") or self.day_source()
        self._env = RoutingEnvironment(self._config(self.day))
        self._dp = self._env.reset(self.day.seed)
        self.day_raw_reward = self._env.pre_decision_reward.total
        if self._dp is None:
            raise RuntimeError("day without routing decisions")
        return _ENCODER.encode(self._dp.observation), {"day": self.day.to_dict()}

    def action_masks(self) -> np.ndarray:
        return np.array(self._dp.action_mask.valid, dtype=bool)

    def step(self, action: int):
        a = ACTION_ORDER[int(action)]
        if not self._dp.action_mask.is_valid(a):
            raise ValueError(f"masked action {a.value} chosen")
        step = self._env.step(a)
        tr = step.transition
        self.day_raw_reward += tr.reward
        self._dp = step.next_decision
        terminated = step.done
        info = {"delta_t_minutes": tr.delta_t_minutes, "raw_reward": tr.reward,
                "reward_breakdown": tr.reward_breakdown.to_dict(), "day": self.day.to_dict(),
                "termination_reason": "END_OF_DAY" if terminated else None}
        if terminated:
            info["day_raw_reward"] = self.day_raw_reward
            obs = np.zeros(OBSERVATION_DIM, dtype=np.float32)
        else:
            obs = _ENCODER.encode(self._dp.observation)
        return obs, tr.reward / REWARD_SCALE, terminated, False, info
