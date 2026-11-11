"""Phase E.1 training: SAME problem as Phase E, stabilised optimiser only (docs/phase_e1_stabilization_prereg.md).

PPO still learns alone: this module never imports the baselines and uses only the Phase E environment
(43-D observation, minimal masks, reward / 100, delta_t, termination) and the unchanged SMDP buffer/GAE.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from callverse_workforce.phase_e.scenarios import TrainingDaySampler
from callverse_workforce.phase_e.smdp import GAMMA_MINUTE, SMDPMaskablePPO

N_ENVS = 8
N_STEPS = 2048
ROLLOUT = N_ENVS * N_STEPS  # 16,384 transitions per update
MAX_DECISIONS = 3_000_000
TOTAL_TIMESTEPS = (MAX_DECISIONS // ROLLOUT) * ROLLOUT  # 183 rollouts = 2,998,272 <= 3M
LR0 = 1e-4
ENT0 = 0.01
TARGET_KL = 0.015
CHECKPOINT_EVERY = 50_000
TIE_TOLERANCE = 1.0
E1_SEEDS = (500, 501, 502, 503, 504)
TRAINING_CONFIG = {
    "algorithm": "SMDP-MaskablePPO (unchanged from Phase E)", "discount": "0.995 ** delta_t_minutes",
    "gamma_minute": GAMMA_MINUTE, "n_envs": N_ENVS, "vec_env": "SubprocVecEnv", "n_steps": N_STEPS,
    "rollout_transitions": ROLLOUT, "batch_size": 512, "n_epochs": 5, "learning_rate": "1e-4 linear -> 0",
    "target_kl": TARGET_KL, "ent_coef": "0.01 linear -> 0", "net_arch": {"pi": [64, 64], "vf": [128, 128]},
    "gae_lambda": 0.95, "clip_range": 0.2, "vf_coef": 0.5, "max_grad_norm": 0.5, "normalize_advantage": True,
    "total_timesteps": TOTAL_TIMESTEPS, "checkpoint_every": CHECKPOINT_EVERY, "tie_tolerance": TIE_TOLERANCE,
    "seeds": list(E1_SEEDS), "reward_scale": "/100 (unchanged)", "ema_polyak_ensemble": "none",
}


def lr_schedule(progress_remaining: float) -> float:
    """sb3 passes progress_remaining = 1 -> 0; linear decay from 1e-4 to 0."""
    return LR0 * progress_remaining


def ent_schedule(progress_remaining: float) -> float:
    return ENT0 * progress_remaining


def worker_sampler_seed(ppo_seed: int, worker: int) -> int:
    """Independent, deterministic day stream per (PPO seed, worker)."""
    return int(np.random.SeedSequence(ppo_seed, spawn_key=(0xE1, worker)).generate_state(1)[0])


def make_env(base_config_path: str, sampler_seed: int):
    from callverse_workforce.config import load_config
    from callverse_workforce.phase_e.env import CallVerseRoutingEnv

    return CallVerseRoutingEnv(load_config(base_config_path), TrainingDaySampler(sampler_seed))


class ScheduledSMDPMaskablePPO(SMDPMaskablePPO):
    """SMDP-MaskablePPO + linear entropy-coefficient decay + logging of KL early stops (smallest extension)."""

    def train(self) -> None:
        self.ent_coef = ent_schedule(self._current_progress_remaining)
        before = self._n_updates
        super().train()
        epochs_done = self._n_updates - before
        self.kl_early_stops = getattr(self, "kl_early_stops", 0) + int(epochs_done < self.n_epochs)
        self.logger.record("train/ent_coef", self.ent_coef)
        self.logger.record("train/epochs_done", epochs_done)
        self.logger.record("train/kl_early_stop", int(epochs_done < self.n_epochs))
        self.logger.record("train/kl_early_stops_total", self.kl_early_stops)


def build_model(env, ppo_seed: int):
    return ScheduledSMDPMaskablePPO(
        "MlpPolicy", env, learning_rate=lr_schedule, ent_coef=ENT0,
        policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [128, 128]}},
        n_steps=N_STEPS, batch_size=512, n_epochs=5, gae_lambda=0.95, clip_range=0.2, vf_coef=0.5,
        max_grad_norm=0.5, normalize_advantage=True, target_kl=TARGET_KL,
        discount_mode="smdp", gamma_minute=GAMMA_MINUTE, seed=ppo_seed, device="cpu", verbose=0)


@dataclass(frozen=True)
class E1RunSpec:
    run_id: str
    ppo_seed: int
    out_dir: str
    base_config_path: str
    total_timesteps: int = TOTAL_TIMESTEPS
    checkpoint_every: int = CHECKPOINT_EVERY


def train_seed(spec: E1RunSpec) -> dict[str, Any]:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import SubprocVecEnv

    torch.set_num_threads(1)
    out = Path(spec.out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    seeds = [worker_sampler_seed(spec.ppo_seed, w) for w in range(N_ENVS)]
    venv = SubprocVecEnv([partial(make_env, spec.base_config_path, s) for s in seeds], start_method="spawn")
    model = build_model(venv, spec.ppo_seed)
    model.set_logger(configure(str(out), ["csv"]))
    episodes = open(out / "training_episodes.jsonl", "w", encoding="utf-8")
    checkpoints: list[dict[str, Any]] = []

    class Tracker(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.next_ckpt = spec.checkpoint_every
            self.counts = np.zeros((N_ENVS, 4))

        def _on_step(self) -> bool:
            for i, (a, done, info) in enumerate(zip(self.locals["actions"], self.locals["dones"], self.locals["infos"])):
                self.counts[i, int(a)] += 1
                if done:
                    n = self.counts[i].sum()
                    episodes.write(json.dumps({"step": self.num_timesteps, "worker": i, "day": info["day"],
                                               "raw_reward": info["day_raw_reward"], "decisions": int(n),
                                               "action_share": dict(zip(("AI", "HUMAN", "WAIT", "CALLBACK"),
                                                                        (self.counts[i] / n).tolist()))}) + "\n")
                    self.counts[i] = 0
            if self.num_timesteps >= self.next_ckpt:
                path = out / "checkpoints" / f"step_{self.next_ckpt}.zip"
                model.save(path)
                checkpoints.append({"nominal_step": self.next_ckpt, "actual_step": self.num_timesteps, "path": str(path)})
                self.next_ckpt += spec.checkpoint_every
            return True

    t0 = time.perf_counter()
    model.learn(total_timesteps=spec.total_timesteps, callback=Tracker())
    final = out / "checkpoints" / f"step_final_{model.num_timesteps}.zip"
    model.save(final)
    checkpoints.append({"nominal_step": model.num_timesteps, "actual_step": model.num_timesteps, "path": str(final)})
    episodes.close()
    venv.close()
    meta = asdict(spec) | {"wall_seconds": time.perf_counter() - t0, "num_timesteps": model.num_timesteps,
                           "worker_sampler_seeds": seeds, "kl_early_stops_total": getattr(model, "kl_early_stops", 0),
                           "training_config": TRAINING_CONFIG}
    (out / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1), encoding="utf-8")
    return {"run_id": spec.run_id, "checkpoints": len(checkpoints), "wall_seconds": meta["wall_seconds"]}


def select_checkpoint(evaluated: list[dict[str, Any]], tolerance: float = TIE_TOLERANCE) -> dict[str, Any]:
    """Pre-registered: highest mean raw reward over ALL 80 validation days; within `tolerance` of the max,
    the EARLIEST checkpoint wins."""
    best = max(e["val_mean_raw_reward"] for e in evaluated)
    within = [e for e in evaluated if e["val_mean_raw_reward"] >= best - tolerance]
    chosen = min(within, key=lambda e: e["actual_step"])
    return chosen | {"best_val_mean": best, "n_within_tolerance": len(within)}
