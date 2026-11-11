"""Phase E.1 report (Phase E shown side by side; Phase E verdict never rewritten)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from callverse_workforce.baselines.metrics import REWARD_KEYS
from callverse_workforce.phase_e.report import ACTIONS, KPI, day_avg, kpi_row
from callverse_workforce.phase_e.scenarios import FAMILIES
from callverse_workforce.phase_e1 import evaluation as E1


def _r(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _progress(run_dir: Path) -> list[dict[str, float]]:
    p = run_dir / "progress.csv"
    if not p.exists():
        return []
    return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(p.open(encoding="utf-8"))]


def opt_summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {}
    col = lambda k: [r[k] for r in rows if k in r]
    kl, ev, vl, ent = col("train/approx_kl"), col("train/explained_variance"), col("train/value_loss"), col("train/entropy_loss")
    stops = col("train/kl_early_stop")
    return {"updates": len(rows), "approx_kl_mean": float(np.mean(kl)), "approx_kl_max": float(np.max(kl)),
            "approx_kl_gt_0.0225": float(np.mean(np.array(kl) > 0.0225)),
            "kl_early_stop_share": float(np.mean(stops)) if stops else None,
            "clip_fraction_mean": float(np.mean(col("train/clip_fraction"))),
            "explained_variance_mean": float(np.mean(ev)), "explained_variance_last": ev[-1],
            "value_loss_mean": float(np.mean(vl)), "value_loss_last": vl[-1],
            "policy_entropy_first": -ent[0], "policy_entropy_last": -ent[-1],
            "lr_first": col("train/learning_rate")[0], "lr_last": col("train/learning_rate")[-1],
            "ent_coef_first": (col("train/ent_coef") or [None])[0], "ent_coef_last": (col("train/ent_coef") or [None])[-1]}


def drops(values: list[float], threshold: float = 50.0) -> int:
    """Validation 'collapses': checkpoint-to-checkpoint drops larger than `threshold` reward/day."""
    return int(sum(1 for a, b in zip(values, values[1:]) if a - b > threshold))


def _diag_table(title: str, groups: dict[str, dict[str, Any]], feature: str, action: str) -> list[str]:
    bins = sorted({b for g in groups.values() for b in g.get(feature, {})})
    L = [f"{title} — {action} share by `{feature}`:", "", "| policy | " + " | ".join(bins) + " |", "|---|" + "---|" * len(bins)]
    for name, g in groups.items():
        L.append(f"| {name} | " + " | ".join(
            (f"{100 * g[feature][b][action]:.1f}% (n={g[feature][b]['n']})" if b in g.get(feature, {}) else "–") for b in bins) + " |")
    return L + [""]


def render_report(out: Path, phase_e: Path) -> str:
    pre, cfg, plan = _r(out / "preregistration_hash.json"), _r(out / "training_config.json"), _r(out / "compute_plan.json")
    sel = _r(out / "selected_models" / "selected_checkpoints.json")
    train = _r(out / "training" / "summary.json")
    v = _r(out / "development_evaluation" / "verdict.json")
    res = _r(out / "development_evaluation" / "per_day.json")
    ev_e = _r(phase_e / "development_evaluation" / "verdict.json")
    res_e = _r(phase_e / "development_evaluation" / "per_day.json")
    e_diag = _r(out / "scenario_diagnostics" / "phase_e_models_on_phase_e_dev.json")
    seeds = sorted(k for k in res if k.startswith("e1_s"))
    e1 = [res[k] for k in seeds]
    e_runs = [res_e[k] for k in sorted(res_e) if k.startswith("smdp_")]
    st, st_e = v["stability"], ev_e["secondary_seed_stability"]
    e_means = [p["mean"] for p in st_e["per_seed"]]
    L = ["# Phase E.1 — training-only PPO stabilisation", "",
         f"> Phase E official verdict (unchanged): **{ev_e['verdict']}**", "",
         f"Pre-registration `docs/phase_e1_stabilization_prereg.md` sha256 `{pre['sha256'][:16]}…` frozen {pre['frozen_at_utc']}. "
         f"Frozen problem hashes: environment `{cfg['frozen_problem']['environment_definition_sha256'][:12]}…`, scenarios "
         f"`{cfg['frozen_problem']['scenario_registry_sha256'][:12]}…` (identical to Phase E).", "",
         "## Training configuration (the only change)", "", f"```\n{json.dumps({k: cfg[k] for k in cfg if k != 'frozen_problem'}, indent=1)}\n```", "",
         "## Compute", "",
         f"- benchmark: one seed with 8 SubprocVecEnv workers {plan['one_seed_8env_train_decisions_per_s']:.0f} decisions/s; "
         f"{plan['seconds_per_validation_day_eval']:.2f} s per validation day-evaluation",
         f"- rollout = 16,384 transitions ≈ {plan['rollout_day_equivalents']:.1f} simulated day-equivalents (Phase E: 2,048 ≈ 1.8)",
         f"- planned: {plan['total_training_decisions']:,} decisions; {plan['checkpoint_day_evaluations']:,} checkpoint day-evaluations",
         f"- actual: training {train['wall_seconds']:.0f} s (5 seeds in parallel); checkpoint evaluation {sel['eval_wall_seconds']:.0f} s", "",
         "## Training, validation curves and checkpoints (all 80 validation days)", "",
         "| seed | selected step | val mean | best val | within tol. | checkpoints | drops >50 | val min..max |", "|---|---|---|---|---|---|---|---|"]
    for k in seeds:
        curve = _r(out / "validation" / f"{k}_curve.json")
        vals = [c["val_mean_raw_reward"] for c in curve]
        s = sel["runs"][k]
        L.append(f"| {k} | {s['actual_step']} | {s['val_mean_raw_reward']:.1f} | {s['best_val_mean']:.1f} | {s['n_within_tolerance']} | "
                 f"{len(curve)} | {drops(vals)} | {min(vals):.0f}..{max(vals):.0f} |")
    L += ["", "Phase E curves (24-day subset, every 100k) for reference: " + "; ".join(
        f"{k}: drops>50 = {drops([c['val_subset_mean_raw_reward'] for c in _r(phase_e / 'training' / k / 'validation_curve.json')])}"
        for k in sorted(p.name for p in (phase_e / "training").iterdir() if p.is_dir())), ""]
    for k in seeds:
        curve = _r(out / "validation" / f"{k}_curve.json")
        L.append(f"- {k}: " + ", ".join(f"{c['actual_step'] // 1000}k={c['val_mean_raw_reward']:.0f}" for c in curve[::3]))
    L += ["", "## Optimisation diagnostics", "",
          "| run | updates | KL mean | KL max | share KL>1.5·target | KL early-stop share | clip frac | expl. var mean / last | value loss mean / last | entropy first→last | lr first→last | ent_coef first→last |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    runs = {k: out / "training" / k for k in seeds} | {f"PhaseE_{p.name}": p for p in sorted((phase_e / "training").iterdir()) if p.is_dir()}
    for name, d in runs.items():
        o = opt_summary(_progress(d))
        if o:
            f = lambda x, n=3: "–" if x is None else f"{x:.{n}g}"
            L.append(f"| {name} | {o['updates']} | {f(o['approx_kl_mean'])} | {f(o['approx_kl_max'])} | {f(o['approx_kl_gt_0.0225'])} | "
                     f"{f(o['kl_early_stop_share'])} | {f(o['clip_fraction_mean'])} | {f(o['explained_variance_mean'])} / {f(o['explained_variance_last'])} | "
                     f"{f(o['value_loss_mean'])} / {f(o['value_loss_last'])} | {f(o['policy_entropy_first'])}→{f(o['policy_entropy_last'])} | "
                     f"{f(o['lr_first'])}→{f(o['lr_last'])} | {f(o['ent_coef_first'])}→{f(o['ent_coef_last'])} |")
    head = "| policy | " + " | ".join(h for _, h, _, _ in KPI) + " | " + " | ".join(f"{a} %" for a in ACTIONS) + " |"
    groups = {"E.1 PPO (5 seeds)": e1, **{n: [res[n]] for n in ("RulePolicy", "ThresholdPolicy", "AIFirstPolicy", "BestFixedPolicy", "RandomPolicy")}}
    L += ["", "## E.1 development evaluation (1681–1760, 80 paired days, opened once)", "", head, "|---|" + "---|" * (len(KPI) + 4)]
    L += [kpi_row(n, r) for n, r in groups.items()]
    L += ["", "RewardBreakdown (mean daily raw):", "", "| policy | " + " | ".join(REWARD_KEYS) + " |", "|---|" + "---|" * len(REWARD_KEYS)]
    for n, r in groups.items():
        L.append(f"| {n} | " + " | ".join(f"{np.mean(day_avg(r, lambda x, k=k: x['reward'][k])):.1f}" for k in REWARD_KEYS) + " |")
    pr = v["ppo_minus_rule"]
    L += ["", "## Primary verdict (same implementation as Phase E)", "",
          f"- Rule mean {v['rule_mean_reward']:.2f}; margin {v['margin']:.2f}",
          f"- PPO_mean − Rule over {pr['n_days']} days: {pr['mean']:.2f} [{pr['lo']:.2f}, {pr['hi']:.2f}] ({100 * v['relative_difference']:.2f}%)",
          f"- PPO − Random {v['ppo_minus_random']['mean']:.1f} [{v['ppo_minus_random']['lo']:.1f}, {v['ppo_minus_random']['hi']:.1f}]; "
          f"PPO − AIFirst {v['ppo_minus_aifirst']['mean']:.1f} [{v['ppo_minus_aifirst']['lo']:.1f}, {v['ppo_minus_aifirst']['hi']:.1f}]",
          "- " + "; ".join(f"PPO − {k} {x['mean']:.1f} [{x['lo']:.1f}, {x['hi']:.1f}]" for k, x in v["other_comparisons"].items()), "",
          f"**{v['verdict']}**", "", "## Seed stability (secondary)", "",
          "| seed | PPO − Rule | 95% CI | non-inferior |", "|---|---|---|---|"]
    for k, p in zip(seeds, st["per_seed"]):
        L.append(f"| {k} | {p['mean']:.1f} | [{p['lo']:.1f}, {p['hi']:.1f}] | {p['noninferior']} |")
    L += ["", f"non-inferior {st['noninferior_seed_count']}/5; seed SD {st['seed_sd']:.1f} (threshold 13.5); range {st['seed_range']:.1f}; "
          f"criterion `{st['criterion']}` → **TRAINING STABILITY: {st['status']}**", "",
          "## Scenario analysis (E.1 dev)", ""]
    for fam in FAMILIES:
        L += [f"### {fam}", "", head, "|---|" + "---|" * (len(KPI) + 4)]
        L += [kpi_row(n, groups[n], fam) for n in ("E.1 PPO (5 seeds)", "RulePolicy", "AIFirstPolicy", "ThresholdPolicy")]
        L.append("")
    L += ["## Worst Day: Phase E vs E.1 (each against its OWN dev days and baselines)", "",
          "| run | reward | resolution % | abandon % | human min | false-AI penalty | waiting cost | AI % | HUMAN % | WAIT % | CALLBACK % | − own Rule |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, runs_, rule_rows in (("Phase E PPO", e_runs, res_e["RulePolicy"]), ("E.1 PPO", e1, res["RulePolicy"])):
        rs = [[r for r in x if r["family"] == "WORST_DAY"] for x in runs_]
        m = lambda key: float(np.mean(day_avg(rs, key)))
        rule_w = float(np.mean([r["total_reward"] for r in rule_rows if r["family"] == "WORST_DAY"]))
        L.append(f"| {name} | {m(lambda r: r['total_reward']):.1f} | {100 * m(lambda r: r['resolution_rate_root']):.1f} | "
                 f"{100 * m(lambda r: r['abandonment_rate']):.2f} | {m(lambda r: r['human_busy_minutes']):.0f} | "
                 f"{m(lambda r: r['reward']['false_ai']):.1f} | {m(lambda r: r['reward']['online_waiting']):.1f} | " +
                 " | ".join(f"{100 * m(lambda r, a=a: r['action_share'][a]):.1f}" for a in ACTIONS) +
                 f" | {m(lambda r: r['total_reward']) - rule_w:+.1f} |")
    diag_groups = {"E.1 PPO": E1.sum_diagnostics([r for x in e1 for r in x]),
                   "Phase E PPO (own dev)": E1.sum_diagnostics([r for k, x in e_diag.items() if k.startswith("smdp") for r in x]),
                   "RulePolicy (E.1 dev)": E1.sum_diagnostics(res["RulePolicy"])}
    wd = {"E.1 PPO": E1.sum_diagnostics([r for x in e1 for r in x], "WORST_DAY"),
          "Phase E PPO (own dev)": E1.sum_diagnostics([r for k, x in e_diag.items() if k.startswith("smdp") for r in x], "WORST_DAY")}
    L += ["", "## WAIT diagnostics (observable state only)", ""]
    for feat in ("human_utilization", "human_queue_size", "humans_available", "forecast_30min", "urgency_pred", "deadline_remaining_pred"):
        L += _diag_table("All dev days", diag_groups, feat, "WAIT")
    L += _diag_table("Worst Day", wd, "human_utilization", "WAIT") + _diag_table("Worst Day", wd, "human_queue_size", "WAIT")
    L += ["WAIT share by scenario family: " + "; ".join(
        f"{fam}: E.1 {100 * np.mean(day_avg([[r for r in x if r['family'] == fam] for x in e1], lambda r: r['action_share']['WAIT'])):.1f}% / "
        f"E {100 * np.mean(day_avg([[r for r in x if r['family'] == fam] for x in e_runs], lambda r: r['action_share']['WAIT'])):.1f}%"
        for fam in FAMILIES), "", "## AI / false-AI diagnostics", ""]
    L += _diag_table("All dev days", diag_groups, "ai_resolution_score", "AI") + _diag_table("All dev days", diag_groups, "recent_ai_failure_rate", "AI")
    L += ["| run | AI % | visible AI failures/day | false AI/day | false-AI penalty/day |", "|---|---|---|---|---|"]
    for name, runs_ in (("E.1 PPO", e1), ("Rule (E.1 dev)", [res["RulePolicy"]]), ("AIFirst (E.1 dev)", [res["AIFirstPolicy"]]),
                        ("Phase E PPO (E dev)", e_runs), ("Rule (E dev)", [res_e["RulePolicy"]])):
        m = lambda key: float(np.mean(day_avg(runs_, key)))
        fa = (lambda r: r.get("false_ai_count", r["ai_attempts_completed"] * r["false_ai_rate"]))
        L.append(f"| {name} | {100 * m(lambda r: r['action_share']['AI']):.1f} | {m(lambda r: r['visible_ai_failures']):.1f} | "
                 f"{m(fa):.1f} | {m(lambda r: r['reward']['false_ai']):.1f} |")
    m_e1 = lambda key: float(np.mean(day_avg(e1, key)))
    m_e = lambda key: float(np.mean(day_avg(e_runs, key)))
    rule_e1 = lambda key: float(np.mean([key(r) for r in res["RulePolicy"]]))
    side = [("Mean reward", lambda r: r["total_reward"], 1, 1), ("Resolution %", lambda r: r["resolution_rate_root"], 100, 1),
            ("Root HIR", lambda r: r["hir_root"], 1, 3), ("Human minutes", lambda r: r["human_busy_minutes"], 1, 0),
            ("Abandonment %", lambda r: r["abandonment_rate"], 100, 2), ("False-AI penalty", lambda r: r["reward"]["false_ai"], 1, 1)] + \
           [(f"{a}%", (lambda r, a=a: r["action_share"][a]), 100, 1) for a in ACTIONS]
    L += ["", "## Phase E vs Phase E.1 (each on its own dev days; not paired across phases)", "",
          "| Metric | Phase E PPO | Phase E.1 PPO | Rule (E.1 dev) |", "|---|---:|---:|---:|"]
    for label, key, mul, nd in side:
        L.append(f"| {label} | {m_e(key) * mul:.{nd}f} | {m_e1(key) * mul:.{nd}f} | {rule_e1(key) * mul:.{nd}f} |")
    pe = ev_e["ppo_minus_rule"]
    L += [f"| PPO − Rule | {pe['mean']:.1f} | {pr['mean']:.1f} | — |",
          f"| 95% CI | [{pe['lo']:.1f}, {pe['hi']:.1f}] | [{pr['lo']:.1f}, {pr['hi']:.1f}] | — |",
          f"| Non-inferior seeds | {st_e['noninferior_seed_count']}/5 | {st['noninferior_seed_count']}/5 | — |",
          f"| Seed SD | {np.std(e_means, ddof=1):.1f} | {st['seed_sd']:.1f} | — |",
          f"| Seed range | {max(e_means) - min(e_means):.1f} | {st['seed_range']:.1f} | — |",
          f"| Stability status | NOT STABLE | {st['status']} | — |", "",
          "## Interpretation (pre-registered sentence)", "", v["interpretation"], "",
          f"**{v['verdict']}**", "", f"TRAINING STABILITY: {st['status']}", ""]
    return "\n".join(L) + "\n"
