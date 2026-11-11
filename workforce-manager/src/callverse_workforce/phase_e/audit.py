"""Pre-training scenario audits, forecast-leakage checks, benchmark and the training-distribution mean delta_t."""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np

from callverse_workforce.config import SimulationConfig, apply_overrides
from callverse_workforce.phase_d import experiment as D
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e.env import CallVerseRoutingEnv
from callverse_workforce.phase_e.scenarios import FAMILIES, TRAINING_ONLY, DaySpec, TrainingDaySampler, day_config
from callverse_workforce.twin.environment import RoutingEnvironment
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.utils.rng import RngManager


def audit_families(base: SimulationConfig, seeds: Sequence[int], frozen: dict[str, Any], workers: int) -> dict[str, Any]:
    pols = {"AIFirstPolicy": ("AIFirstPolicy", {}), "RulePolicy": ("RulePolicy", frozen["RulePolicy"])}
    jobs = [(day_config(base, fam, MIXTURE_CELLS[i % 4]), name, params, s, f"{fam}|{label}")
            for fam in FAMILIES + TRAINING_ONLY for label, (name, params) in pols.items() for i, s in enumerate(seeds)]
    rows = D.pmap(D.day_job, jobs, workers)
    out: dict[str, Any] = {}
    for r in rows:
        fam, label = r["tag"].split("|")
        out.setdefault(fam, {}).setdefault(label, []).append(r)
    summary = {}
    for fam, by in out.items():
        summary[fam] = {"reconciled_all": all(r["reconciled"] for rs in by.values() for r in rs)}
        for label, rs in by.items():
            summary[fam][label] = {k: float(np.mean([r[k] for r in rs])) for k in
                                   ("total_reward", "instances", "decisions", "abandonment_rate", "missed_deadline_rate",
                                    "human_utilization", "resolution_rate_root", "hir_root", "online_wait_p90")}
    return {"seeds": list(seeds), "families": summary}


def leakage_checks(base: SimulationConfig, seeds: Sequence[int], rule: dict[str, Any]) -> dict[str, Any]:
    """(1) hidden profile never leaks (calendar families): same contacts, hidden rates x2 -> identical observations;
    (2) windowed events invisible before onset: observations equal the NORMAL-family day until the window starts."""
    from callverse_workforce.baselines.policies import make_policy
    from callverse_workforce.phase_e.evaluation import WINDOW_KIND, windows

    checks: dict[str, Any] = {}
    for fam in ("SUMMER", "RAMADAN", "OMRA_HOLIDAYS"):
        ok = True
        for s in seeds:
            cfg = day_config(base, fam, MIXTURE_CELLS[0])
            contacts = ContactGenerator(cfg, RngManager(s)).generate_day()
            hidden = apply_overrides(cfg, {"scenario.hourly_arrival_rates": [r * 2 for r in cfg.scenario.hourly_arrival_rates]})
            obs = []
            for c in (cfg, hidden):
                env = RoutingEnvironment(c, contacts=contacts)
                p = make_policy("RulePolicy", rule, s)
                dp, seq = env.reset(s), []
                while dp is not None:
                    seq.append(dp.observation)
                    dp = env.step(p.predict(dp.observation, dp.action_mask)).next_decision
                obs.append(seq)
            ok &= obs[0] == obs[1]
        checks[f"{fam}_hidden_profile_does_not_leak"] = {"ok": bool(ok), "reference_profile": cfg.forecasting.reference_hourly_arrival_rates}
    for fam, kind in WINDOW_KIND.items():
        if fam == "WORST_DAY":
            continue
        ok, gaps = True, []
        for s in seeds:
            cfg, ref = day_config(base, fam, MIXTURE_CELLS[0]), day_config(base, "NORMAL", MIXTURE_CELLS[0])
            w = windows(cfg, s)[kind]
            seqs = []
            for c in (cfg, ref):
                env = RoutingEnvironment(c)
                p = make_policy("RulePolicy", rule, s)
                dp, seq = env.reset(s), []
                while dp is not None and dp.timestamp < w[0]:
                    seq.append(dp.observation)
                    dp = env.step(p.predict(dp.observation, dp.action_mask)).next_decision
                seqs.append(seq)
            ok &= seqs[0] == seqs[1] and len(seqs[0]) > 0
            gaps.append(w)
        checks[f"{fam}_invisible_before_onset"] = {"ok": bool(ok), "windows": gaps}
    return checks


def benchmark(base: SimulationConfig) -> dict[str, float]:
    import torch

    from callverse_workforce.phase_e.smdp import SMDPMaskablePPO

    torch.set_num_threads(1)
    env = CallVerseRoutingEnv(base, TrainingDaySampler(9))
    rng = np.random.default_rng(0)
    env.reset()
    n, days, t = 0, 0, time.perf_counter()
    while days < 5:
        _, _, done, _, _ = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
        n += 1
        if done:
            days += 1
            env.reset()
    env_sps = n / (time.perf_counter() - t)
    model = SMDPMaskablePPO("MlpPolicy", CallVerseRoutingEnv(base, TrainingDaySampler(10)), n_steps=2048, batch_size=64,
                            n_epochs=10, seed=0, device="cpu")
    t = time.perf_counter()
    model.learn(8192)
    train_sps = 8192 / (time.perf_counter() - t)
    obs, _ = env.reset()
    t, m = time.perf_counter(), 0
    for _ in range(3000):
        a, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=True)
        obs, _, done, _, _ = env.step(int(a))
        m += 1
        if done:
            obs, _ = env.reset()
    return {"env_random_steps_per_s": env_sps, "ppo_train_steps_per_s": train_sps,
            "ppo_eval_steps_per_s": m / (time.perf_counter() - t), "decisions_per_day": n / days}


def mean_training_delta_t(base: SimulationConfig, seed: int, n_days: int) -> float:
    """Mean decision interval on the TRAINING distribution (uniform random valid actions; no baseline logic)."""
    env = CallVerseRoutingEnv(base, TrainingDaySampler(seed))
    rng = np.random.default_rng(seed)
    dts = []
    for _ in range(n_days):
        env.reset()
        done = False
        while not done:
            _, _, done, _, info = env.step(int(rng.choice(np.flatnonzero(env.action_masks()))))
            dts.append(info["delta_t_minutes"])
    return float(np.mean(dts))


assert DaySpec  # re-exported for tests
