"""Smallest clean SMDP extension of sb3-contrib MaskablePPO.

Per transition t (decision k -> k+1, ``delta_t`` simulated minutes apart):

    discount_t = gamma_minute ** delta_t                        (SMDP, primary;  gamma_minute = 0.995)
               = gamma_fixed                                    (Transition-PPO / Myopic ablations)
    delta_k    = r_k + discount_k * V(s_{k+1}) * (1 - done_{k+1}) - V(s_k)
    A_k        = delta_k + discount_k * lambda * (1 - done_{k+1}) * A_{k+1}
    R_k        = A_k + V(s_k)

``done`` marks the true end of the day: the bootstrap after it is exactly 0. Episodes are never truncated.
Only ``collect_rollouts`` (to pass ``info["delta_t_minutes"]`` to the buffer) and the GAE computation are
changed; the PPO loss, clipping, masking and optimisation are stock sb3-contrib.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

GAMMA_MINUTE = 0.995
DiscountMode = Literal["smdp", "fixed"]


def transition_discounts(delta_t: np.ndarray, mode: DiscountMode, gamma_minute: float = GAMMA_MINUTE,
                         gamma_fixed: float | None = None) -> np.ndarray:
    if mode == "smdp":
        return np.power(gamma_minute, delta_t, dtype=np.float64)
    if gamma_fixed is None:
        raise ValueError("fixed discount mode needs gamma_fixed")
    return np.full_like(np.asarray(delta_t, dtype=np.float64), gamma_fixed)


def smdp_gae(rewards: np.ndarray, values: np.ndarray, episode_starts: np.ndarray, discounts: np.ndarray,
             last_values: np.ndarray, last_dones: np.ndarray, gae_lambda: float) -> tuple[np.ndarray, np.ndarray]:
    """GAE with a transition-specific discount. Arrays are (n_steps, n_envs)."""
    n = rewards.shape[0]
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1], dtype=np.float64)
    for k in reversed(range(n)):
        if k == n - 1:
            non_terminal = 1.0 - last_dones.astype(np.float64)
            next_values = last_values
        else:
            non_terminal = 1.0 - episode_starts[k + 1]
            next_values = values[k + 1]
        delta = rewards[k] + discounts[k] * next_values * non_terminal - values[k]
        last = delta + discounts[k] * gae_lambda * non_terminal * last
        adv[k] = last
    return adv, adv + values


class SMDPRolloutBuffer(MaskableRolloutBuffer):
    """Maskable rollout buffer that also stores ``delta_t`` and discounts per transition."""

    def __init__(self, *args, discount_mode: DiscountMode = "smdp", gamma_minute: float = GAMMA_MINUTE,
                 gamma_fixed: float | None = None, **kwargs) -> None:
        self.discount_mode, self.gamma_minute, self.gamma_fixed = discount_mode, gamma_minute, gamma_fixed
        super().__init__(*args, **kwargs)

    def reset(self) -> None:
        self.delta_t = np.zeros((self.buffer_size, self.n_envs), dtype=np.float64)
        super().reset()

    def add(self, *args, delta_t: np.ndarray | None = None, **kwargs) -> None:
        if delta_t is None:
            raise ValueError("SMDP buffer requires delta_t for every transition")
        self.delta_t[self.pos] = np.asarray(delta_t, dtype=np.float64).reshape(self.n_envs)
        super().add(*args, **kwargs)

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        last = last_values.clone().cpu().numpy().flatten()
        disc = transition_discounts(self.delta_t, self.discount_mode, self.gamma_minute, self.gamma_fixed)
        adv, ret = smdp_gae(self.rewards.astype(np.float64), self.values.astype(np.float64), self.episode_starts,
                            disc, last, dones, self.gae_lambda)
        self.advantages = adv.astype(np.float32)
        self.returns = ret.astype(np.float32)


class SMDPMaskablePPO(MaskablePPO):
    """MaskablePPO whose rollouts carry ``delta_t_minutes`` into an ``SMDPRolloutBuffer``."""

    def __init__(self, *args, discount_mode: DiscountMode = "smdp", gamma_minute: float = GAMMA_MINUTE,
                 gamma_fixed: float | None = None, **kwargs) -> None:
        kwargs["rollout_buffer_class"] = SMDPRolloutBuffer
        kwargs["rollout_buffer_kwargs"] = {"discount_mode": discount_mode, "gamma_minute": gamma_minute,
                                           "gamma_fixed": gamma_fixed}
        # sb3's own gamma is unused by the SMDP buffer; set it to the per-minute value for transparency
        kwargs.setdefault("gamma", gamma_minute if discount_mode == "smdp" else float(gamma_fixed))
        super().__init__(*args, **kwargs)

    def collect_rollouts(self, env: VecEnv, callback: BaseCallback, rollout_buffer, n_rollout_steps: int,
                         use_masking: bool = True) -> bool:
        """Copy of MaskablePPO.collect_rollouts (sb3-contrib 2.9) + delta_t; truncation is not allowed."""
        assert isinstance(rollout_buffer, SMDPRolloutBuffer)
        assert self._last_obs is not None
        self.policy.set_training_mode(False)
        n_steps = 0
        rollout_buffer.reset()
        if use_masking and not is_masking_supported(env):
            raise ValueError("Environment does not support action masking")
        callback.on_rollout_start()
        while n_steps < n_rollout_steps:
            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                action_masks = get_action_masks(env) if use_masking else None
                actions, values, log_probs = self.policy(obs_tensor, action_masks=action_masks)
            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)
            n_steps += 1
            if isinstance(self.action_space, spaces.Discrete):
                actions = actions.reshape(-1, 1)
            if any(info.get("TimeLimit.truncated", False) for info in infos):
                raise RuntimeError("episodes must end at the true end of day (no truncation)")
            delta_t = np.array([info["delta_t_minutes"] for info in infos], dtype=np.float64)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs,
                               action_masks=action_masks, delta_t=delta_t)
            self._last_obs = new_obs
            self._last_episode_starts = dones
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()
        return True
