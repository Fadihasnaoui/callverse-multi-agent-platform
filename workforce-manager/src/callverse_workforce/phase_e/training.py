"""PPO training runs (one process per run). PPO learns ALONE.

This module (with ``smdp.py`` / ``env.py``) never imports ``callverse_workforce.baselines``, never calls a
baseline policy, and builds no expert labels, demonstrations, imitation targets or baseline-shaped
rewards: the agent only sees the 43-D observation, the action mask, the scaled environment reward,
the next observation, ``delta_t_minutes`` and termination. (Enforced by static tests.)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from callverse_workforce.config import SimulationConfig
from callverse_workforce.domain.action_mask import ActionMask
from callverse_workforce.domain.enums import ACTION_ORDER, RoutingAction
from callverse_workforce.domain.observation import RoutingObservation
from callverse_workforce.encoding import ObservationEncoder
from callverse_workforce.phase_e.env import CallVerseRoutingEnv
from callverse_workforce.phase_e.scenarios import DaySpec, TrainingDaySampler
from callverse_workforce.phase_e.smdp import GAMMA_MINUTE, SMDPMaskablePPO

FIXED_PPO = {"n_steps": 2048, "batch_size": 64, "n_epochs": 10, "gae_lambda": 0.95, "clip_range": 0.2,
             "vf_coef": 0.5, "max_grad_norm": 0.5, "normalize_advantage": True, "activation": "tanh (sb3 default)",
             "n_envs": 1, "device": "cpu", "torch_threads": 1}
GRID = [{"learning_rate": lr, "net_arch": net, "ent_coef": ent}
        for lr in (1e-4, 3e-4) for net in ([64, 64], [128, 128]) for ent in (0.0, 0.01)]
SEARCH_BUDGET = 250_000
FULL_BUDGET = 1_000_000
EVAL_EVERY = 100_000
_ENCODER = ObservationEncoder()


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    hyper: dict[str, Any]
    ppo_seed: int
    budget: int
    discount_mode: str  # "smdp" | "fixed"
    gamma_fixed: float | None
    out_dir: str
    eval_days: tuple[tuple[str, str, int], ...]  # validation SUBSET only (family, cell_id, seed)
    eval_every: int = EVAL_EVERY
    base_config_path: str = ""
    fixed: dict[str, Any] = field(default_factory=lambda: dict(FIXED_PPO))


class PPORoutingPolicy:
    """Adapter exposing a trained model through the RoutingPolicy interface (deterministic, masked argmax)."""

    name = "PPO"

    def __init__(self, model) -> None:
        self.model = model

    def reset(self) -> None:
        pass

    def predict(self, observation: RoutingObservation, action_mask: ActionMask) -> RoutingAction:
        obs = _ENCODER.encode(observation)
        action, _ = self.model.predict(obs, action_masks=np.array(action_mask.valid, dtype=bool), deterministic=True)
        return ACTION_ORDER[int(action)]


def evaluate_model(model, base: SimulationConfig, days: Sequence[DaySpec]) -> list[dict[str, Any]]:
    """Run the deterministic policy on fixed days through the Gym env (raw rewards, action shares)."""
    env = CallVerseRoutingEnv(base, lambda: None)
    out = []
    for day in days:
        obs, _ = env.reset(options={"day": day})
        counts = np.zeros(len(ACTION_ORDER))
        done = False
        while not done:
            a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
            counts[int(a)] += 1
            obs, _, done, _, info = env.step(int(a))
        out.append(day.to_dict() | {"raw_reward": info["day_raw_reward"],
                                    "action_share": dict(zip((x.value for x in ACTION_ORDER), (counts / counts.sum()).tolist()))})
    return out


def _cells_by_id():
    from callverse_workforce.phase_d.grid import MIXTURE_CELLS

    return {c.cell_id: c for c in MIXTURE_CELLS}


def train_run(spec: RunSpec) -> dict[str, Any]:
    import torch
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import configure

    from callverse_workforce.config import load_config

    torch.set_num_threads(1)
    out = Path(spec.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = load_config(spec.base_config_path)
    cells = _cells_by_id()
    eval_days = [DaySpec(f, cells[c], s) for f, c, s in spec.eval_days]
    sampler = TrainingDaySampler(spec.ppo_seed * 7919 + 13)
    env = CallVerseRoutingEnv(base, sampler)
    hyper = spec.hyper
    model = SMDPMaskablePPO(
        "MlpPolicy", env, learning_rate=hyper["learning_rate"], ent_coef=hyper["ent_coef"],
        policy_kwargs={"net_arch": {"pi": list(hyper["net_arch"]), "vf": list(hyper["net_arch"])}},
        n_steps=spec.fixed["n_steps"], batch_size=spec.fixed["batch_size"], n_epochs=spec.fixed["n_epochs"],
        gae_lambda=spec.fixed["gae_lambda"], clip_range=spec.fixed["clip_range"], vf_coef=spec.fixed["vf_coef"],
        max_grad_norm=spec.fixed["max_grad_norm"], normalize_advantage=spec.fixed["normalize_advantage"],
        discount_mode=spec.discount_mode, gamma_minute=GAMMA_MINUTE, gamma_fixed=spec.gamma_fixed,
        seed=spec.ppo_seed, device="cpu", verbose=0)
    model.set_logger(configure(str(out), ["csv"]))

    episodes = open(out / "training_episodes.jsonl", "w", encoding="utf-8")
    curve: list[dict[str, Any]] = []

    class Tracker(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.counts = np.zeros(4)
            self.masks = []
            self.dts = []
            self.next_eval = spec.eval_every

        def _on_step(self) -> bool:
            a = int(self.locals["actions"][0])
            self.counts[a] += 1
            info = self.locals["infos"][0]
            self.dts.append(info["delta_t_minutes"])
            if self.locals["dones"][0]:
                n = self.counts.sum()
                episodes.write(json.dumps({"step": self.num_timesteps, "day": info["day"], "raw_reward": info["day_raw_reward"],
                                           "scaled_reward": info["day_raw_reward"] / 100.0, "decisions": int(n),
                                           "mean_delta_t": float(np.mean(self.dts)),
                                           "action_share": dict(zip(("AI", "HUMAN", "WAIT", "CALLBACK"), (self.counts / n).tolist()))}) + "\n")
                episodes.flush()
                self.counts[:] = 0
                self.dts = []
            if self.num_timesteps >= self.next_eval:
                self.next_eval += spec.eval_every
                checkpoint(self.num_timesteps)
            return True

    def checkpoint(step: int) -> None:
        path = out / "checkpoints" / f"step_{step}.zip"
        model.save(path)
        res = evaluate_model(model, base, eval_days)
        share = {a.value: float(np.mean([r["action_share"][a.value] for r in res])) for a in ACTION_ORDER}
        curve.append({"step": step, "checkpoint": str(path), "val_subset_mean_raw_reward": float(np.mean([r["raw_reward"] for r in res])),
                      "val_subset_days": res, "action_share": share})
        (out / "validation_curve.json").write_text(json.dumps(curve, indent=1), encoding="utf-8")

    t0 = time.perf_counter()
    model.learn(total_timesteps=spec.budget, callback=Tracker())
    if not curve or curve[-1]["step"] < model.num_timesteps:
        checkpoint(model.num_timesteps)
    episodes.close()
    meta = asdict(spec) | {"wall_seconds": time.perf_counter() - t0, "num_timesteps": model.num_timesteps,
                           "versions": _versions()}
    (out / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return {"run_id": spec.run_id, "out_dir": spec.out_dir, "curve": [{k: v for k, v in c.items() if k != "val_subset_days"} for c in curve]}


def _versions() -> dict[str, str]:
    import gymnasium
    import sb3_contrib
    import stable_baselines3
    import torch

    return {"torch": torch.__version__, "gymnasium": gymnasium.__version__, "stable_baselines3": stable_baselines3.__version__,
            "sb3_contrib": sb3_contrib.__version__, "numpy": np.__version__}


def select_checkpoint(curve: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered: argmax of the centred 3-point moving average of validation-subset reward
    (edges average the available neighbours); ties -> earliest checkpoint. Avoids isolated spikes."""
    vals = [c["val_subset_mean_raw_reward"] for c in curve]
    smooth = [float(np.mean(vals[max(0, i - 1): i + 2])) for i in range(len(vals))]
    best = max(range(len(vals)), key=lambda i: (smooth[i], -i))
    return {"index": best, "step": curve[best]["step"], "checkpoint": curve[best]["checkpoint"],
            "smoothed": smooth[best], "raw": vals[best], "smoothed_curve": smooth}
