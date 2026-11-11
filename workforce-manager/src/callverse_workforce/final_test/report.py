"""V1.1 final-test report (sections A-L). Everything beyond the three pre-registered labels is descriptive."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from callverse_workforce.baselines.metrics import REWARD_KEYS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.phase_e.report import ACTIONS, day_avg
from callverse_workforce.phase_e.scenarios import FAMILIES

BASELINES = ("RulePolicy", "AIFirstPolicy", "ThresholdPolicy", "BestFixedPolicy", "RandomPolicy")
FULL_KPI = [("total_reward", "reward", 1, 1), ("resolution_rate_instance", "resolution %", 100, 1),
            ("resolution_rate_root", "root resol %", 100, 1), ("hir_root", "root HIR", 1, 3), ("hir_instance", "inst HIR", 1, 3),
            ("human_routing_rate", "HUMAN routing %", 100, 1), ("human_busy_minutes", "human min", 1, 0),
            ("human_utilization", "util", 1, 3), ("abandonment_rate", "abandon %", 100, 2),
            ("missed_deadline_rate", "deadline %", 100, 2), ("unfinished_rate", "unfinished %", 100, 2),
            ("online_wait_mean", "wait mean", 1, 2), ("online_wait_p90", "wait p90", 1, 2), ("false_ai_count", "false AI n", 1, 1),
            ("false_ai_rate", "false AI %", 100, 1), ("visible_ai_failures", "vis AI fail", 1, 1),
            ("callback_success", "cb success", 1, 2), ("cb_failure", "cb failure", 1, 2), ("callback_unfinished", "cb unfinished", 1, 2),
            ("mean_valid_actions", "mask freedom", 1, 2)]
SHORT_KPI = [("total_reward", 1, 1), ("resolution_rate_instance", 100, 1), ("hir_root", 1, 3), ("human_busy_minutes", 1, 0),
             ("online_wait_p90", 1, 2), ("abandonment_rate", 100, 2)]
REWARD_LABELS = {"resolution": "correctly resolved", "human_busy": "human busy", "ai_busy": "AI busy", "online_waiting": "online waiting",
                 "callback_pending": "callback pending", "abandonment": "abandonment", "missed_deadline": "missed deadline",
                 "false_ai": "false-AI penalty", "callback_failure": "callback failure", "unfinished": "unfinished", "total": "TOTAL"}


def _r(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _val(r: dict[str, Any], k: str) -> float:
    if k == "cb_failure":
        return r["callback_dropout"] + r["callback_max_attempts"] + r["callback_window_expired"]
    return r[k]


def _runs(res: dict[str, list], name: str, ppo: Sequence[str], family: str | None = None) -> list[list[dict[str, Any]]]:
    labels = list(ppo) if name == "PPO" else [name]
    return [[r for r in res[l] if family is None or r["family"] == family] for l in labels]


def _mean(runs, key) -> float:
    return float(np.mean(day_avg(runs, key)))


def _ci(d: dict[str, float]) -> str:
    return f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"


def _short_row(name: str, runs) -> str:
    cells = [f"{_mean(runs, lambda r, k=k: _val(r, k)) * m:.{nd}f}" for k, m, nd in SHORT_KPI]
    acts = [f"{100 * _mean(runs, lambda r, a=a: r['action_share'][a]):.1f}" for a in ACTIONS]
    return f"| {name} | " + " | ".join(cells + acts) + " |"


def _policies(ppo) -> list[str]:
    return ["PPO"] + list(BASELINES)


def render(out: Path, ppo: Sequence[str]) -> tuple[str, dict[str, Any]]:
    res = _r(out / "raw_results" / "per_day.json")
    st = _r(out / "paired_statistics" / "primary.json")
    integ, audit, mh = _r(out / "integrity_checks.json"), _r(out / "seed_collision_audit.json"), _r(out / "model_hashes.json")
    env, reg, pre = _r(out / "environment_hashes.json"), _r(out / "seed_registry.json"), _r(out / "preregistration_hash.json")
    post = _r(out / "integrity_post_run.json")
    flag = _r(out / "raw_results" / "final_test_opened.json")
    extra: dict[str, Any] = {}
    ppo_mean = EV.ppo_mean_by_day([res[k] for k in ppo])
    day_by = {n: EV.by_day(res[n]) for n in BASELINES}

    # ---------------- A. integrity
    L = ["# V1.1 FINAL TEST — out-of-sample evaluation (seeds 2000–2239, run once)", "",
         "> Frozen E.1 PPO (5 independent models, deterministic) vs. frozen V1.1 baselines. No retraining, retuning, "
         "Twin / reward / mask / observation / scenario change, or V2. Historical verdicts unchanged: Phase E FAILURE, "
         "Phase E.1 SUCCESS + STABLE, targeted audit INITIAL ROUTING — AI/HUMAN ONLY.", "",
         "## A. Integrity", "",
         f"- Pre-registration `docs/final_test_v1_1_prereg.md`, SHA-256 `{pre['sha256']}` (frozen {pre['frozen_at_utc']}); "
         f"final test opened {flag['opened_at_utc']} (flag written before the first simulation).",
         f"- Seed collision audit: **{audit['n_collisions_final']} collisions** in 2000–2239, **{audit['n_collisions_v2']}** "
         f"in 2240–2999. Sources checked: " + ", ".join(f"{k} ({v})" for k, v in audit["sources_checked"].items()) + ".",
         f"- Static scan: {audit['static_code_scan']}.",
         f"- Integrity checks: **{sum(integ['checks'].values())}/29 passed** before opening "
         "(E.1 dev day 1681 reproduced bit-for-bit by all 10 frozen policies; current statistics code regenerates the frozen E.1 verdict exactly).",
         "- Disclosure: a first freeze attempt aborted BEFORE opening (check 25 used a string pattern absent from the pre-registration "
         "because of a line break — a check-implementation bug, not a protocol failure). The check pattern was corrected, the freeze "
         "re-run, and the failed record kept as `integrity_checks_attempt1_failed.json`. No final-test seed was simulated in between.",
         f"- Historical artifact trees unchanged after the run: **{post['unchanged']}** "
         f"(Phase E `{post['artifact_trees_post']['phase_e']['sha256'][:12]}…`, E.1 `{post['artifact_trees_post']['phase_e1']['sha256'][:12]}…`, "
         f"targeted `{post['artifact_trees_post']['targeted_headroom']['sha256'][:12]}…`).", "",
         "| PPO model | training seed | step | SHA-256 |", "|---|---:|---:|---|"]
    L += [f"| {k} | {m['training_seed']} | {m['selected_step']:,} | `{m['sha256']}` |" for k, m in sorted(mh["models"].items())]
    L += ["", f"- Evaluation: {mh['evaluation']}.",
          f"- Environment: twin `{env['source']['twin']['combined'][:12]}…`, reward `{env['source']['reward']['combined'][:12]}…`, "
          f"encoder `{env['source']['observation_encoder']['combined'][:12]}…`, masks `{env['source']['mask_logic']['combined'][:12]}…`, "
          f"environment definition `{env['environment_definition_sha256'][:12]}…`, scenario registry `{env['scenario_registry_sha256'][:12]}…` "
          f"(all 36 day-config hashes recomputed and identical to Phase E: {env['day_config_hashes_match_phase_e_registry']}).",
          f"- Canonical families (registry wins): " + "; ".join(f"{k} → `{v}`" for k, v in reg["prompt_name_to_registry_name"].items()) + ".",
          f"- Mapping: {reg['mapping_rule']}; 30 days per family; 240 days. No final-test seed was simulated before pre-registration.",
          "- **2240–2999 remain CLOSED** (never simulated, inspected or audited; guarded in code by `assert_v2_reserved_untouched`).", ""]

    # ---------------- B. overall
    head = ("| Policy | Reward | Resolution % | Root HIR | Human min | Wait p90 | Abandon % | AI% | HUMAN% | WAIT% | CALLBACK% |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    L += ["## B. Overall final result (240 days; raw reward; PPO = day-level mean of the 5 models)", "", head]
    L += [_short_row("PPO (mean of 5)" if n == "PPO" else n, _runs(res, n, ppo)) for n in _policies(ppo)]
    L += ["", "Full KPI set (means over days; raw reward SD across days):", "",
          "| Policy | reward SD | " + " | ".join(h for _, h, _, _ in FULL_KPI) + " |", "|---|" + "---:|" * (len(FULL_KPI) + 1)]
    overall = {}
    for n in _policies(ppo):
        runs = _runs(res, n, ppo)
        vals = {k: _mean(runs, lambda r, k=k: _val(r, k)) for k, _, _, _ in FULL_KPI}
        sd = float(np.std(day_avg(runs, lambda r: r["total_reward"]), ddof=1))
        overall[n] = vals | {"reward_sd": sd, "action_share": {a: _mean(runs, lambda r, a=a: r["action_share"][a]) for a in ACTIONS}}
        L.append(f"| {n} | {sd:.1f} | " + " | ".join(f"{vals[k] * m:.{nd}f}" for k, _, m, nd in FULL_KPI) + " |")
    extra["scenario_results/overall_kpi.json"] = overall
    L.append("")

    # ---------------- C. PPO vs Rule
    pr = st["ppo_minus_rule"]
    L += ["## C. PPO vs RulePolicy (primary, pre-registered)", "",
          f"- PPO mean reward **{st['ppo_mean_reward']:.2f}**; Rule mean **{st['rule_mean_reward']:.2f}**",
          f"- mean paired PPO−Rule **{pr['mean']:+.2f}** (median {pr['median']:+.2f}; day SD {pr['sd']:.1f}); relative {100 * st['relative_difference']:+.2f}%",
          f"- 95% paired day-bootstrap CI **[{pr['lo']:+.2f}, {pr['hi']:+.2f}]** over n = {pr['n_days']} days (B = 10,000)",
          f"- non-inferiority margin −2% × Rule mean = **{st['margin']:.2f}**; CI lower {pr['lo']:+.2f} "
          f"{'>' if pr['lo'] > st['margin'] else '≤'} margin",
          f"- **{st['verdict']}**", ""]

    # ---------------- D. PPO vs AI-first
    va = st["ppo_minus_aifirst"]
    ca = st["close_to_aifirst"]
    L += ["## D. PPO vs AI-first", "",
          f"- PPO−AIFirst {_ci(va)}; relative {100 * va['relative']:+.2f}%  → (A) CI contains 0: **{ca['A_ci_contains_zero']}**",
          "- decision agreement with AI-first (each PPO's own decision states, same mask): "
          + ", ".join(f"{k} {100 * a:.2f}%" for k, a in zip(ppo, st["aifirst_agreement_per_seed"]))
          + f"; mean **{100 * st['aifirst_agreement_mean']:.2f}%** → (B) ≥ 95%: **{ca['B_agreement_ge_95']}**",
          f"- **{ca['label']}**", ""]

    # ---------------- E. other baselines
    L += ["## E. Other baselines (secondary, descriptive)", "", "| comparison | mean [95% CI] | relative |", "|---|---:|---:|"]
    L += [f"| PPO − {k} | {_ci(v)} | {100 * v['relative']:+.2f}% |" for k, v in st["other_comparisons"].items()]
    L.append("")

    # ---------------- F. seed stability
    s = st["stability"]
    L += ["## F. Seed stability (E.1 rule reused exactly)", "", "| Seed | mean PPO reward | PPO−Rule | 95% CI | relative | Individually non-inferior? |",
          "|---|---:|---:|---:|---:|---|"]
    seed_rows = []
    for k, p in zip(ppo, s["per_seed"]):
        m = float(np.mean([r["total_reward"] for r in res[k]]))
        seed_rows.append({"seed": k, "mean_reward": m} | p)
        L.append(f"| {k} | {m:.2f} | {p['mean']:+.2f} | [{p['lo']:+.2f}, {p['hi']:+.2f}] | {100 * p['mean'] / st['rule_mean_reward']:+.2f}% | "
                 f"{'yes' if p['noninferior'] else 'no'} |")
    best = max(seed_rows, key=lambda x: x["mean"])
    worst = min(seed_rows, key=lambda x: x["mean"])
    L += ["", f"- non-inferior seeds **{s['noninferior_seed_count']}/5**; seed SD **{s['seed_sd']:.2f}** (frozen threshold 13.5); "
              f"range {s['seed_range']:.2f}; best {best['seed']} ({best['mean']:+.2f}); worst {worst['seed']} ({worst['mean']:+.2f})",
          f"- **{st['stability_label']}**", ""]

    # ---------------- G. scenario results
    L += ["## G. Scenario results (30 days per family; CIs descriptive only)", ""]
    scen = {}
    for fam in FAMILIES:
        L += [f"### {fam}", "", head.replace("| Policy |", "| Policy |")]
        for n in ("PPO", "RulePolicy", "AIFirstPolicy", "ThresholdPolicy"):
            L.append(_short_row("PPO (mean of 5)" if n == "PPO" else n, _runs(res, n, ppo, fam)))
        seeds_f = sorted(r["seed"] for r in res["RulePolicy"] if r["family"] == fam)
        sub = lambda a, b: [a[d] - b[d] for d in seeds_f]
        ci_r = EV.paired_bootstrap(sub(ppo_mean, day_by["RulePolicy"]), key=200 + FAMILIES.index(fam))
        ci_a = EV.paired_bootstrap(sub(ppo_mean, day_by["AIFirstPolicy"]), key=300 + FAMILIES.index(fam))
        fa = {n: _mean(_runs(res, n, ppo, fam), lambda r: r["reward"]["false_ai"]) for n in ("PPO", "RulePolicy", "AIFirstPolicy")}
        scen[fam] = {"ppo_minus_rule": ci_r, "ppo_minus_aifirst": ci_a, "false_ai_penalty": fa,
                     "kpi": {n: {k: _mean(_runs(res, n, ppo, fam), lambda r, k=k: _val(r, k)) for k, _, _, _ in FULL_KPI}
                             | {"action_share": {a: _mean(_runs(res, n, ppo, fam), lambda r, a=a: r["action_share"][a]) for a in ACTIONS}}
                             for n in _policies(ppo)}}
        L += ["", f"PPO−Rule {_ci(ci_r)}; PPO−AIFirst {_ci(ci_a)}; false-AI penalty PPO {fa['PPO']:.1f} / Rule {fa['RulePolicy']:.1f} / "
                  f"AIFirst {fa['AIFirstPolicy']:.1f}", ""]
    extra["scenario_results/by_family.json"] = scen

    def periods(fam: str) -> list[str]:
        out_l = []
        for n in ("PPO", "RulePolicy", "AIFirstPolicy"):
            agg: dict[str, list[dict[str, float]]] = defaultdict(list)
            for rows in _runs(res, n, ppo, fam):
                for r in rows:
                    for p, v in r["adaptation"].get("periods", {}).items():
                        agg[p].append(v)
            out_l.append(f"- {n}: " + "; ".join(
                f"{p}: AI {100 * np.mean([v['AI_share'] for v in vs]):.1f}% / HUMAN {100 * np.mean([v['HUMAN_share'] for v in vs]):.1f}% / "
                f"WAIT {100 * np.mean([v['WAIT_share'] for v in vs]):.1f}% / CB {100 * np.mean([v['CALLBACK_share'] for v in vs]):.1f}%, "
                f"recent AI fail {np.mean([v['recent_ai_failure_rate'] for v in vs]):.2f}, queue {np.mean([v['human_queue_size'] for v in vs]):.2f}, "
                f"forecast30 {np.mean([v['forecast_30min'] for v in vs]):.1f}" for p, vs in sorted(agg.items())))
        return out_l

    k = lambda fam, n, key: scen[fam]["kpi"][n][key]
    L += ["### Scenario questions (descriptive)", "",
          f"**AI_DEGRADATION** — PPO−AIFirst {_ci(scen['AI_DEGRADATION']['ppo_minus_aifirst'])}, PPO−Rule {_ci(scen['AI_DEGRADATION']['ppo_minus_rule'])}; "
          f"AI share PPO {100 * k('AI_DEGRADATION', 'PPO', 'action_share')['AI']:.1f}% vs AIFirst {100 * k('AI_DEGRADATION', 'AIFirstPolicy', 'action_share')['AI']:.1f}%. "
          "Routing by hidden-window period (labels from hidden windows, analysis only):", ""] + periods("AI_DEGRADATION") + [""]
    L += [f"**HUMAN_SHORTAGE** — utilization PPO {k('HUMAN_SHORTAGE', 'PPO', 'human_utilization'):.3f} / Rule {k('HUMAN_SHORTAGE', 'RulePolicy', 'human_utilization'):.3f}; "
          f"HUMAN routing PPO {100 * k('HUMAN_SHORTAGE', 'PPO', 'human_routing_rate'):.1f}% / Rule {100 * k('HUMAN_SHORTAGE', 'RulePolicy', 'human_routing_rate'):.1f}%; "
          f"abandonment PPO {100 * k('HUMAN_SHORTAGE', 'PPO', 'abandonment_rate'):.2f}% / Rule {100 * k('HUMAN_SHORTAGE', 'RulePolicy', 'abandonment_rate'):.2f}%; "
          f"wait p90 PPO {k('HUMAN_SHORTAGE', 'PPO', 'online_wait_p90'):.2f} / Rule {k('HUMAN_SHORTAGE', 'RulePolicy', 'online_wait_p90'):.2f}; "
          f"PPO−Rule {_ci(scen['HUMAN_SHORTAGE']['ppo_minus_rule'])}.", ""] + periods("HUMAN_SHORTAGE") + [""]
    L += [f"**DEMAND_SHOCK** — WAIT/CALLBACK share PPO {100 * k('DEMAND_SHOCK', 'PPO', 'action_share')['WAIT']:.2f}% / "
          f"{100 * k('DEMAND_SHOCK', 'PPO', 'action_share')['CALLBACK']:.2f}%; abandonment PPO {100 * k('DEMAND_SHOCK', 'PPO', 'abandonment_rate'):.2f}% / "
          f"Rule {100 * k('DEMAND_SHOCK', 'RulePolicy', 'abandonment_rate'):.2f}%; wait p90 PPO {k('DEMAND_SHOCK', 'PPO', 'online_wait_p90'):.2f} / "
          f"Rule {k('DEMAND_SHOCK', 'RulePolicy', 'online_wait_p90'):.2f}; PPO−Rule {_ci(scen['DEMAND_SHOCK']['ppo_minus_rule'])}.", ""] + periods("DEMAND_SHOCK") + [""]

    # ---------------- H. worst day
    wd = "WORST_DAY"
    L += ["## H. Worst Day", "", "| | PPO | Rule | AIFirst |", "|---|---:|---:|---:|"]
    wd_rows = [("reward", "total_reward", 1, 1), ("resolution %", "resolution_rate_instance", 100, 1), ("abandon %", "abandonment_rate", 100, 2),
               ("human min", "human_busy_minutes", 1, 0)]
    for lab, key, m, nd in wd_rows:
        L.append(f"| {lab} | " + " | ".join(f"{k(wd, n, key) * m:.{nd}f}" for n in ("PPO", "RulePolicy", "AIFirstPolicy")) + " |")
    for a in ACTIONS:
        L.append(f"| {a}% | " + " | ".join(f"{100 * k(wd, n, 'action_share')[a]:.1f}" for n in ("PPO", "RulePolicy", "AIFirstPolicy")) + " |")
    for comp in ("false_ai", "online_waiting"):
        L.append(f"| {REWARD_LABELS[comp]} reward | " + " | ".join(
            f"{_mean(_runs(res, n, ppo, wd), lambda r, c=comp: r['reward'][c]):.1f}" for n in ("PPO", "RulePolicy", "AIFirstPolicy")) + " |")
    e1_wd = _e1_family_gap(out, ppo, wd)
    L += ["", f"Final test PPO−Rule on WORST_DAY {_ci(scen[wd]['ppo_minus_rule'])}; E.1 development (10 days) PPO−Rule {e1_wd:+.2f}.", ""]

    # ---------------- I. reward breakdown
    L += ["## I. Reward breakdown (mean daily contribution, raw)", "", "| component | PPO | Rule | AIFirst | PPO−Rule | PPO−AIFirst |",
          "|---|---:|---:|---:|---:|---:|"]
    rb = {}
    for c in REWARD_KEYS:
        v = {n: _mean(_runs(res, n, ppo), lambda r, c=c: r["reward"][c]) for n in ("PPO", "RulePolicy", "AIFirstPolicy")}
        rb[c] = v
        L.append(f"| {REWARD_LABELS.get(c, c)} | {v['PPO']:.2f} | {v['RulePolicy']:.2f} | {v['AIFirstPolicy']:.2f} | "
                 f"{v['PPO'] - v['RulePolicy']:+.2f} | {v['PPO'] - v['AIFirstPolicy']:+.2f} |")
    extra["reward_breakdown/breakdown.json"] = rb
    gains = sorted(((c, rb[c]["PPO"] - rb[c]["RulePolicy"]) for c in REWARD_KEYS if c != "total"), key=lambda x: x[1])
    L += ["", "vs Rule — PPO loses most on: " + ", ".join(f"{REWARD_LABELS[c]} ({d:+.2f})" for c, d in gains[:3])
          + "; gains most on: " + ", ".join(f"{REWARD_LABELS[c]} ({d:+.2f})" for c, d in gains[::-1][:3]) + ".", ""]

    # ---------------- J. policy behaviour
    L += ["## J. Policy behaviour", "", "| PPO seed | agree AI-first | agree Rule | exact AI-first days /240 | PPO−AIFirst | AI% | HUMAN% | WAIT% | CALLBACK% |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sim = {}
    for i, lab in enumerate(ppo):
        d = EV.paired_bootstrap(EV.paired_diff(EV.by_day(res[lab]), day_by["AIFirstPolicy"]), key=400 + i)
        sh = {a: float(np.mean([r["action_share"][a] for r in res[lab]])) for a in ACTIONS}
        sim[lab] = {"agree_aifirst": st["aifirst_agreement_per_seed"][i], "agree_rule": st["rule_agreement_per_seed"][i],
                    "exact_aifirst_days": st["exact_aifirst_days_per_seed"][i], "ppo_minus_aifirst": d, "action_share": sh}
        L.append(f"| {lab} | {100 * sim[lab]['agree_aifirst']:.2f}% | {100 * sim[lab]['agree_rule']:.2f}% | {sim[lab]['exact_aifirst_days']} | "
                 f"{_ci(d)} | " + " | ".join(f"{100 * sh[a]:.2f}" for a in ACTIONS) + " |")
    L += [f"| **5-seed mean** | {100 * st['aifirst_agreement_mean']:.2f}% | {100 * st['rule_agreement_mean']:.2f}% | "
          f"{np.mean(st['exact_aifirst_days_per_seed']):.1f} | {_ci(va)} | " + " | ".join(f"{100 * overall['PPO']['action_share'][a]:.2f}" for a in ACTIONS) + " |", ""]
    ctx: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for lab in ppo:
        for r in res[lab]:
            for name, groups in r["agreement"]["rule_disagreement_by_context"].items():
                for g, v in groups.items():
                    ctx[name][g][0] += v["n"]
                    ctx[name][g][1] += v["disagree"]
    L += ["PPO-vs-Rule disagreement rate by observable context (all 5 seeds pooled; interpretation only, no new rule):", ""]
    for name, groups in ctx.items():
        L.append(f"- {name}: " + "; ".join(f"{g} {100 * d / n:.1f}% (n={n:,})" for g, (n, d) in sorted(groups.items()) if n))
    sim["rule_disagreement_by_context"] = {k_: {g: {"n": v[0], "disagree": v[1]} for g, v in d.items()} for k_, d in ctx.items()}
    extra["policy_similarity/similarity.json"] = sim
    wc = {fam: {a: 100 * k(fam, "PPO", "action_share")[a] for a in ("WAIT", "CALLBACK")} for fam in FAMILIES}
    L += ["", "WAIT / CALLBACK usage of the frozen PPO (targeted conclusion frozen, not reopened): overall WAIT "
          f"{100 * overall['PPO']['action_share']['WAIT']:.2f}%, CALLBACK {100 * overall['PPO']['action_share']['CALLBACK']:.2f}%; by family: "
          + "; ".join(f"{f} {v['WAIT']:.2f}/{v['CALLBACK']:.2f}" for f, v in wc.items()) + " (WAIT%/CALLBACK%).", ""]
    extra["policy_similarity/wait_callback.json"] = wc

    # ---------------- K. fairness
    L += ["## K. Fairness / subgroups — DESCRIPTIVE ONLY (no influence on any label)", ""]
    fair = {}
    for dim in ("language", "channel"):
        L += [f"| {dim} | policy | contacts/day | resolution % | inst HIR | wait mean | abandon % | AI routing % | HUMAN routing % |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        groups = sorted({g for r in res["RulePolicy"] for g in r["fairness"][dim]})
        for g in groups:
            for n in ("PPO", "RulePolicy", "AIFirstPolicy"):
                runs = _runs(res, n, ppo)
                gm = lambda key: _mean(runs, lambda r: r["fairness"][dim][g][key])
                tot = sum(sum(r["agreement"]["routing_by_group"][dim].get(g, {}).values()) for rows in runs for r in rows)
                route = {a: (sum(r["agreement"]["routing_by_group"][dim].get(g, {}).get(a, 0) for rows in runs for r in rows) / tot if tot else 0.0)
                         for a in ("AI", "HUMAN")}
                fair.setdefault(dim, {}).setdefault(g, {})[n] = {"instances": gm("instances"), "resolution_rate": gm("resolution_rate"),
                                                                 "hir_instance": gm("hir_instance"), "wait_mean": gm("wait_mean"),
                                                                 "abandonment_rate": gm("abandonment_rate"), "routing": route, "decisions": tot}
                f_ = fair[dim][g][n]
                L.append(f"| {g} | {n} | {f_['instances']:.1f} | {100 * f_['resolution_rate']:.1f} | {f_['hir_instance']:.3f} | {f_['wait_mean']:.2f} | "
                         f"{100 * f_['abandonment_rate']:.2f} | {100 * route['AI']:.1f} | {100 * route['HUMAN']:.1f} |")
        L.append("")
    extra["fairness/subgroups.json"] = fair

    # ---------------- L. conclusions
    noninf = st["verdict"].endswith("NON-INFERIOR TO RULEPOLICY") and "NOT" not in st["verdict"]
    fam_gap = {f: scen[f]["ppo_minus_rule"]["mean"] for f in FAMILIES}
    strongest, weakest = max(fam_gap, key=fam_gap.get), min(fam_gap, key=fam_gap.get)
    beats_rule = pr["lo"] > 0
    beyond_aif = va["lo"] > 0
    L += ["## L. Scientific conclusions", "",
          f"1. **Generalisation of E.1:** {'yes' if noninf else 'no'} — the pre-registered non-inferiority result "
          f"{'was reproduced' if noninf else 'was not reproduced'} on 240 untouched days.",
          f"2. **Non-inferior to Rule:** {st['verdict']}.",
          f"3. **Gap:** PPO−Rule {pr['mean']:+.2f} reward/day ({100 * st['relative_difference']:+.2f}%), 95% CI [{pr['lo']:+.2f}, {pr['hi']:+.2f}], "
          f"margin {st['margin']:.2f}. " + ("The CI lies above 0, so PPO also beats Rule." if beats_rule else "No claim that PPO beats Rule is supported."),
          f"4. **Seed stability:** {s['noninferior_seed_count']}/5 seeds individually non-inferior, seed SD {s['seed_sd']:.2f} (≤ 13.5 required) → {st['stability_label']}.",
          f"5. **Close to AI-first:** {ca['label']} (CI contains 0: {ca['A_ci_contains_zero']}; mean agreement {100 * st['aifirst_agreement_mean']:.2f}%).",
          f"6. **Value beyond AI-first:** PPO−AIFirst {_ci(va)} — "
          + ("measurable but small positive value (CI above 0, +" f"{100 * va['relative']:.2f}%)" if beyond_aif else "no measurable value beyond AI-first (CI not above 0)")
          + f"; {sum(d == 240 for d in st['exact_aifirst_days_per_seed'])}/5 models reproduce AI-first exactly on all 240 days, so any gain "
            "comes from the remaining models (descriptive).",
          f"7. **Strongest scenario for PPO (vs Rule, descriptive):** {strongest} ({fam_gap[strongest]:+.2f}).",
          f"8. **Weakest scenario:** {weakest} ({fam_gap[weakest]:+.2f}).",
          f"9. **Worst Day:** final-test PPO−Rule {_ci(scen[wd]['ppo_minus_rule'])} vs E.1 development {e1_wd:+.2f} (descriptive).",
          "10. **Phase C/D limited headroom:** " + ("consistent — the learned policy stays within a few reward points of AI-first and Rule, "
                                                     "as predicted by the small forced-action headroom." if abs(va["mean"]) < abs(st["margin"]) else
                                                     "the PPO–AI-first gap exceeds the non-inferiority margin; see D/I."),
          f"11. **WAIT/CALLBACK:** PPO uses WAIT {100 * overall['PPO']['action_share']['WAIT']:.2f}% and CALLBACK "
          f"{100 * overall['PPO']['action_share']['CALLBACK']:.2f}% of decisions — consistent with INITIAL ROUTING — AI/HUMAN ONLY.",
          "12. **V1.1 scientifically complete:** yes — the pre-registered sequence (baselines → headroom → PPO → stabilisation → targeted audit → "
          "untouched final test) is closed; no further V1.1 tuning is permitted.",
          "13. **Move to V2:** yes — remaining headroom in the V1.1 formulation is small; improvements require the V2 formulation "
          "(evaluated on the still-closed seeds 2240–2999).", "",
          "### Pre-registered interpretation", "", f"> {st['interpretation']}", "",
          st["verdict"], "", st["stability_label"], "", ca["label"], "", "V1.1 FINAL TEST COMPLETE", ""]
    return "\n".join(L), extra


def _e1_family_gap(out: Path, ppo: Sequence[str], family: str) -> float:
    p = out.parent / "ppo_v1_1_e1" / "development_evaluation" / "per_day.json"
    res = _r(p)
    pm = EV.ppo_mean_by_day([[r for r in res[k] if r["family"] == family] for k in ppo])
    rule = {r["seed"]: r["total_reward"] for r in res["RulePolicy"] if r["family"] == family}
    return float(np.mean(EV.paired_diff(pm, rule)))
