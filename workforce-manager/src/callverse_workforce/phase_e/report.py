"""Phase E report rendered from persisted artifacts only (raw rewards; /100 scaling is training-only)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from callverse_workforce.baselines.metrics import REWARD_KEYS
from callverse_workforce.phase_e.scenarios import FAMILIES

KPI = [("total_reward", "reward", 1, 1), ("resolution_rate_root", "resolution %", 100, 1), ("hir_root", "root HIR", 1, 3),
       ("hir_instance", "inst HIR", 1, 3), ("human_routing_rate", "HUMAN routing %", 100, 1),
       ("human_busy_minutes", "human min", 1, 0), ("human_utilization", "util", 1, 3), ("online_wait_mean", "wait mean", 1, 2),
       ("online_wait_p90", "wait p90", 1, 2), ("abandonment_rate", "abandon %", 100, 2), ("missed_deadline_rate", "deadline %", 100, 2),
       ("unfinished_rate", "unfinished %", 100, 2), ("false_ai_rate", "false AI %", 100, 1), ("visible_ai_failures", "vis AI fail", 1, 0),
       ("callback_success", "cb success", 1, 1), ("callback_dropout", "cb dropout", 1, 1), ("callback_max_attempts", "cb max att", 1, 1),
       ("callback_window_expired", "cb window", 1, 1), ("mean_valid_actions", "mean valid", 1, 2)]
ACTIONS = ("AI", "HUMAN", "WAIT", "CALLBACK")


def _r(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _f(x: Any, nd: int = 1) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def day_avg(runs: Sequence[Sequence[dict[str, Any]]], key) -> list[float]:
    """Per-day average over independent runs (PPO seeds), ordered by day seed."""
    maps = [{r["seed"]: key(r) for r in rows} for rows in runs]
    return [float(np.mean([m[d] for m in maps])) for d in sorted(maps[0])]


def kpi_row(name: str, runs, family: str | None = None) -> str:
    rs = [[r for r in rows if family is None or r["family"] == family] for rows in runs]
    cells = [f"{np.mean(day_avg(rs, lambda r, k=k: r[k])) * m:.{nd}f}" for k, _, m, nd in KPI]
    acts = [f"{100 * np.mean(day_avg(rs, lambda r, a=a: r['action_share'][a])):.1f}" for a in ACTIONS]
    return f"| {name} | " + " | ".join(cells + acts) + " |"


def _progress(run_dir: Path) -> dict[str, Any]:
    p = run_dir / "progress.csv"
    if not p.exists():
        return {}
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    last = rows[-1]
    pick = ("train/entropy_loss", "train/policy_gradient_loss", "train/value_loss", "train/explained_variance",
            "train/approx_kl", "train/clip_fraction")
    return {k.split("/")[1]: float(last[k]) for k in pick if last.get(k)}


def render_report(out: Path) -> str:
    plan, envdef = _r(out / "compute_plan.json"), _r(out / "environment_definition.json")
    cands = _r(out / "hyperparameter_search" / "candidates.json")
    sel_cfg = _r(out / "selected_models" / "selected_config.json")
    sel = _r(out / "selected_models" / "selected_checkpoints.json")
    audit, leak = _r(out / "scenario_audits" / "audit.json"), _r(out / "environment_checks" / "leakage_checks.json")
    reg = _r(out / "scenario_registry.json")
    L = ["# Phase E — MaskablePPO on the frozen V1.1 Twin", "",
         f"Primary environment: Phase D 4-cell mixture `{[c['cell_id'] for c in envdef['mixture_cells']]}` "
         f"(p = {envdef['mixture_probabilities']}) × 8 scenario families; training distribution `{envdef['training_distribution']}`.",
         "PPO learns alone from the 43-D observation, masks, env reward /100, next observation, delta_t and termination.", "",
         "## Scenario audit (10 audit days, frozen V1.1 baselines) and leakage checks", "",
         "| family | reconciled | AIFirst reward | Rule reward | Rule abandon % | Rule util | decisions/day |", "|---|---|---|---|---|---|---|"]
    for fam, v in audit["families"].items():
        L.append(f"| {fam} | {v['reconciled_all']} | {v['AIFirstPolicy']['total_reward']:.1f} | {v['RulePolicy']['total_reward']:.1f} | "
                 f"{100 * v['RulePolicy']['abandonment_rate']:.2f} | {v['RulePolicy']['human_utilization']:.3f} | {v['RulePolicy']['decisions']:.0f} |")
    L += ["", "Leakage checks: " + "; ".join(f"{k}: {'OK' if v['ok'] else 'FAIL'}" for k, v in leak.items()), "",
          f"Frozen scenario config hashes: {len(reg['configs'])} family×cell configs (scenario_registry.json, frozen {reg['frozen_at_utc']}).", "",
          "## Compute", "", f"Benchmark: `{plan['benchmark']}`", "",
          f"Planned: {plan['estimated_training_decisions_total']:,} training decisions (~{plan['estimated_training_days_total']:.0f} days), "
          f"est. {plan['estimated_sequential_training_seconds']:.0f} s sequential / {plan['estimated_parallel_training_seconds']:.0f} s parallel. "
          f"Actual wall: search training {cands['train_wall_seconds']:.0f} s, full+ablation {sel['wall_seconds']:.0f} s.", "",
          "## SMDP", "",
          "`discount_k = 0.995 ** delta_t_k`; `delta_k = r_k + discount_k V(s_{k+1})(1-done_{k+1}) - V(s_k)`; "
          "`A_k = delta_k + discount_k * lambda * (1-done_{k+1}) A_{k+1}`; no bootstrap after the true end of day.", "",
          f"Mean training delta_t = {plan['mean_training_delta_t_minutes']:.4f} min → Transition-PPO gamma_fixed = "
          f"{plan['transition_ppo_gamma_fixed']:.6f}; Myopic gamma = 0.", "",
          "## Hyperparameter search (8 configs × 2 seeds × 250k steps; full 80-day validation)", "",
          f"Random validation mean: {cands['random_validation_mean']:.1f}", "",
          "| # | lr | net | ent | seed rewards | mean | spread | AI % | dominant | selected |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in cands["candidates"]:
        h = c["hyper"]
        L.append(f"| {c['index']} | {h['learning_rate']:g} | {h['net_arch']} | {h['ent_coef']:g} | "
                 f"{', '.join(f'{x:.1f}' for x in c['seed_means'])} | {c['mean_val_reward']:.1f} | {c['seed_spread']:.1f} | "
                 f"{100 * c['action_share']['AI']:.1f} | {100 * c['dominant_share']:.1f}% | {'**yes**' if c['index'] == sel_cfg['selected']['index'] else ''} |")
    L += ["", f"Selected: `{sel_cfg['selected']['hyper']}`; excluded (catastrophic): `{sel_cfg['excluded']}`.", "",
          "## Full training (5 SMDP seeds) and ablations (3 + 3) — validation-only checkpoint selection", "",
          "| run | selected step | smoothed val (24-day subset) | AI % | HUMAN % | WAIT % | CB % | dominant >90 / >95 | entropy | expl. var | approx KL | clip frac |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for rid, v in sel["runs"].items():
        run_dir = out / ("training" if rid.startswith("smdp") else "ablation") / rid
        curve = _r(run_dir / "validation_curve.json")
        point = next(c for c in curve if c["step"] == v["step"])
        sh = point["action_share"]
        dom = max(sh.values())
        pr = _progress(run_dir)
        L.append(f"| {rid} | {v['step']} | {v['smoothed']:.1f} | " + " | ".join(f"{100 * sh[a]:.1f}" for a in ACTIONS) +
                 f" | {dom > 0.9} / {dom > 0.95} | {_f(-pr.get('entropy_loss', float('nan')), 3)} | {_f(pr.get('explained_variance'), 3)} | "
                 f"{_f(pr.get('approx_kl'), 4)} | {_f(pr.get('clip_fraction'), 3)} |")
    L += ["", "Validation curves (24-day subset mean raw reward per checkpoint):", ""]
    for rid in sel["runs"]:
        run_dir = out / ("training" if rid.startswith("smdp") else "ablation") / rid
        curve = _r(run_dir / "validation_curve.json")
        L.append(f"- {rid}: " + ", ".join(f"{c['step'] // 1000}k={c['val_subset_mean_raw_reward']:.0f}" for c in curve))
    dev_v = out / "development_evaluation" / "verdict.json"
    if not dev_v.exists():
        return "\n".join(L) + "\n"
    v = _r(dev_v)
    res = _r(out / "development_evaluation" / "per_day.json")
    smdp = [res[k] for k in sorted(res) if k.startswith("smdp_")]
    trans = [res[k] for k in sorted(res) if k.startswith("transition_")]
    myop = [res[k] for k in sorted(res) if k.startswith("myopic_")]
    groups = {"SMDP-PPO (5 seeds)": smdp, "Transition-PPO (3)": trans, "Myopic PPO (3)": myop,
              **{n: [res[n]] for n in ("RulePolicy", "ThresholdPolicy", "AIFirstPolicy", "BestFixedPolicy", "RandomPolicy")}}
    head = "| policy | " + " | ".join(h for _, h, _, _ in KPI) + " | " + " | ".join(f"{a} %" for a in ACTIONS) + " |"
    L += ["", "## Development evaluation (1601–1680, 80 paired days; raw reward)", "", head, "|---|" + "---|" * (len(KPI) + 4)]
    L += [kpi_row(n, runs) for n, runs in groups.items()]
    L += ["", "RewardBreakdown (mean daily raw contribution):", "", "| policy | " + " | ".join(REWARD_KEYS) + " |", "|---|" + "---|" * len(REWARD_KEYS)]
    for n, runs in groups.items():
        L.append(f"| {n} | " + " | ".join(f"{np.mean(day_avg(runs, lambda r, k=k: r['reward'][k])):.1f}" for k in REWARD_KEYS) + " |")
    pr = v["ppo_minus_rule"]
    st = v["secondary_seed_stability"]
    L += ["", "### Primary paired result", "",
          f"- Rule mean reward {v['rule_mean_reward']:.2f}; margin (−2%) = {v['margin']:.2f}",
          f"- D_d = PPO_mean_d − Rule_d over {pr['n_days']} days: mean {pr['mean']:.2f}, 95% paired bootstrap CI "
          f"[{pr['lo']:.2f}, {pr['hi']:.2f}], relative {100 * v['relative_difference']:.2f}%",
          f"- PPO − Random {v['ppo_minus_random']['mean']:.1f} [{v['ppo_minus_random']['lo']:.1f}, {v['ppo_minus_random']['hi']:.1f}]; "
          f"PPO − AIFirst {v['ppo_minus_aifirst']['mean']:.1f} [{v['ppo_minus_aifirst']['lo']:.1f}, {v['ppo_minus_aifirst']['hi']:.1f}]",
          "- other: " + "; ".join(f"PPO − {k} {x['mean']:.1f} [{x['lo']:.1f}, {x['hi']:.1f}]" for k, x in v["other_comparisons"].items()),
          f"- seed stability: {st['noninferior_seed_count']}/{st['n_seeds']} seeds individually non-inferior; per seed " +
          ", ".join(f"{p['mean']:.1f} [{p['lo']:.1f}, {p['hi']:.1f}]" for p in st["per_seed"]),
          f"- ablations vs Rule: Transition {v['transition_vs_rule']['mean']:.1f} [{v['transition_vs_rule']['lo']:.1f}, {v['transition_vs_rule']['hi']:.1f}], "
          f"Myopic {v['myopic_vs_rule']['mean']:.1f} [{v['myopic_vs_rule']['lo']:.1f}, {v['myopic_vs_rule']['hi']:.1f}]; "
          f"SMDP − Transition {v['smdp_vs_transition']['mean']:.1f} [{v['smdp_vs_transition']['lo']:.1f}, {v['smdp_vs_transition']['hi']:.1f}], "
          f"SMDP − Myopic {v['smdp_vs_myopic']['mean']:.1f} [{v['smdp_vs_myopic']['lo']:.1f}, {v['smdp_vs_myopic']['hi']:.1f}]", "",
          f"**{v['verdict']}**", ""]
    L += ["## Scenario-by-scenario (dev)", ""]
    for fam in FAMILIES:
        L += [f"### {fam}", "", head, "|---|" + "---|" * (len(KPI) + 4)]
        L += [kpi_row(n, groups[n], fam) for n in ("SMDP-PPO (5 seeds)", "RulePolicy", "AIFirstPolicy", "ThresholdPolicy",
                                                    "Transition-PPO (3)", "Myopic PPO (3)")]
        L.append("")
    L += ["## Adaptation diagnostics (observable variables; hidden windows only label periods)", ""]
    for fam in ("AI_DEGRADATION", "HUMAN_SHORTAGE", "DEMAND_SHOCK", "WORST_DAY"):
        L.append(f"### {fam}")
        for n in ("SMDP-PPO (5 seeds)", "RulePolicy", "AIFirstPolicy"):
            rows = [r for runs in groups[n] for r in runs if r["family"] == fam]
            per: dict[str, list] = {}
            for r in rows:
                for p, x in r["adaptation"].get("periods", {}).items():
                    per.setdefault(p, []).append(x)
            L.append(f"- {n}: " + "; ".join(
                f"{p}: AI {100 * np.mean([x['AI_share'] for x in xs]):.0f}% HUMAN {100 * np.mean([x['HUMAN_share'] for x in xs]):.0f}% "
                f"(recent AI fail {np.mean([x['recent_ai_failure_rate'] for x in xs]):.2f}, humans avail {np.mean([x['humans_available'] for x in xs]):.2f}, "
                f"queue {np.mean([x['human_queue_size'] for x in xs]):.2f}, forecast {np.mean([x['forecast_30min'] for x in xs]):.1f})"
                for p, xs in sorted(per.items())))
        L.append("")
    L.append("AI share by observed recent_ai_failure_rate (all dev days):")
    for n in ("SMDP-PPO (5 seeds)", "RulePolicy", "AIFirstPolicy"):
        rows = [r for runs in groups[n] for r in runs]
        bins = sorted({b for r in rows for b in r["adaptation"]["ai_share_by_recent_ai_failure_rate"]})
        L.append(f"- {n}: " + ", ".join(
            f"{b}: {100 * np.average([r['adaptation']['ai_share_by_recent_ai_failure_rate'][b]['ai_share'] for r in rows if b in r['adaptation']['ai_share_by_recent_ai_failure_rate']], weights=[r['adaptation']['ai_share_by_recent_ai_failure_rate'][b]['n'] for r in rows if b in r['adaptation']['ai_share_by_recent_ai_failure_rate']]):.0f}%"
            for b in bins))
    L += ["", "## Fairness diagnostics (dev, descriptive)", ""]
    for group in ("language", "channel"):
        L += [f"### by {group}", "", "| policy | group | resolution % | HIR | wait mean | wait p90 | abandon % |", "|---|---|---|---|---|---|---|"]
        for n in ("SMDP-PPO (5 seeds)", "RulePolicy", "AIFirstPolicy"):
            keys = list(groups[n][0][0]["fairness"][group])
            for k in keys:
                vals = {m: np.mean(day_avg(groups[n], lambda r, m=m, k=k: r["fairness"][group][k][m]))
                        for m in ("resolution_rate", "hir_instance", "wait_mean", "wait_p90", "abandonment_rate")}
                L.append(f"| {n} | {k} | {100 * vals['resolution_rate']:.1f} | {vals['hir_instance']:.3f} | {vals['wait_mean']:.2f} | "
                         f"{vals['wait_p90']:.2f} | {100 * vals['abandonment_rate']:.2f} |")
        L.append("")
    return "\n".join(L) + "\n"
