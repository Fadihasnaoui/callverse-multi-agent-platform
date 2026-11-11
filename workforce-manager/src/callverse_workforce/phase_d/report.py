"""Phase D markdown report + machine-readable headroom map, rendered from persisted artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from callverse_workforce.headroom.experiment import LABELS
from callverse_workforce.phase_d.grid import GRID, HANDOFF, HANGUP, MULTIPLIER, Cell

POL = ("RandomPolicy", "AIFirstPolicy", "ThresholdPolicy", "RulePolicy", "BestFixedPolicy")


def _r(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _f(x: Any, nd: int = 2) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def final_decision(confirm: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered rule (rationale section 9), applied to V1.1 D3 cells only."""
    evidence = {}
    sufficient, fragile = False, False
    for cid, s in confirm.items():
        c = s["rule"]["classification"]
        aif = s["aifirst"]["regret"]["RulePolicy"]["cross_fitted_ci"]
        learn = s["rule"]["learnability"]["learnable_gain_vs_rule"]
        ok_robust = aif["lo"] > 0
        ok_learn = learn["lo"] > 0 and learn["mean"] >= 0.25
        evidence[cid] = {"class": c["label"], "R": c["R"], "ci": [c["ci_lo"], c["ci_hi"]], "aifirst_cf_ci_lo": aif["lo"],
                         "learnable_gain": learn, "robust": ok_robust, "learnable": ok_learn}
        if c["label"] == "CLEARLY_POSITIVE" and ok_robust and ok_learn:
            sufficient = True
        if c["ci_lo"] > 0:
            fragile = True
    verdict = ("V1.1 HEADROOM SUFFICIENT — READY TO DESIGN PPO" if sufficient else
               "V1.1 HEADROOM FRAGILE — MORE VALIDATION BEFORE PPO" if fragile else
               "V1.1 HEADROOM INSUFFICIENT — REFORMULATE AGAIN")
    return {"verdict": verdict, "evidence": evidence}


def headroom_map(screen: dict[str, Any], evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for c in GRID:
        s = screen[c.cell_id]
        cl, reg = s["classification"], s["regret"]
        ev = evaluation["table"][c.cell_id]["RulePolicy"]
        rows.append(c.to_dict() | {
            "class": cl["label"], "rule_cf_regret": cl["R"], "ci_lo": cl["ci_lo"], "ci_hi": cl["ci_hi"],
            "relative_to_mean_J": cl["relative_to_mean_J"],
            "aifirst_cf_regret": reg["AIFirstPolicy"]["cross_fitted"]["mean"],
            **{f"{l}_share": s["label_distribution"][l] for l in LABELS},
            "mean_J_human_minus_ai": s["advantage_vs_ai"]["HUMAN"].get("mean"),
            "near_tie_lt_5": s["near_tie_fractions"]["lt_5"],
            "rule_resolution": ev["resolution_rate_root"]["mean"], "rule_hir_root": ev["hir_root"]["mean"],
            "rule_human_minutes": ev["human_busy_minutes"]["mean"], "rule_abandonment": ev["abandonment_rate"]["mean"],
            "screening_states": s["n_analyzed_states"],
        })
    return rows


def _grid_table(rows: list[dict[str, Any]], p: float, fmt) -> list[str]:
    by = {(r["handoff_overhead_minutes"], r["frustration_patience_multiplier"]): r for r in rows
          if r["immediate_hangup_prob_after_ai_failure"] == p}
    L = ["| Handoff min | " + " | ".join(f"Patience ×{m:g}" for m in MULTIPLIER) + " |", "|---|" + "---:|" * len(MULTIPLIER)]
    for h in HANDOFF:
        L.append(f"| {h:g} | " + " | ".join(fmt(by[(h, m)]) for m in MULTIPLIER) + " |")
    return L


def _confirm_block(name: str, s: dict[str, Any]) -> list[str]:
    r, a = s["rule"], s["aifirst"]
    c = r["classification"]
    ra = r["regret"]
    learn = r["learnability"]
    sep = learn.get("feature_separation", {})
    L = [f"#### {name}", "",
         f"- class **{c['label']}**: R = {c['R']:.3f} [{c['ci_lo']:.3f}, {c['ci_hi']:.3f}] per eligible decision; "
         f"relative to mean J {c['relative_to_mean_J']:.5f}; ties {_pct(c['tie_share'])}; near-ties <5 {_pct(c['near_tie_lt_5'])}",
         f"- states {r['n_analyzed_states']} (reconstruction {_pct(r['reconstruction_success_rate'])}), branches {r['branch_replays']}",
         "- labels: " + ", ".join(f"{l} {_pct(r['label_distribution'][l])}" for l in LABELS),
         f"- Rule regret raw {ra['RulePolicy']['raw']['mean']:.3f} (median {ra['RulePolicy']['raw']['median']:.3f}, "
         f"p90 {ra['RulePolicy']['raw']['p90']:.3f}), clear {ra['RulePolicy']['clear']['mean']:.3f}; AIFirst regret raw "
         f"{ra['AIFirstPolicy']['raw']['mean']:.3f}, cross-fitted {ra['AIFirstPolicy']['cross_fitted']['mean']:.3f} "
         f"[{ra['AIFirstPolicy']['cross_fitted_ci']['lo']:.3f}, {ra['AIFirstPolicy']['cross_fitted_ci']['hi']:.3f}]",
         "- advantage vs AI: " + "; ".join(
             f"{k}: mean {_f(v.get('mean'))}, clearly>0 {_pct(v['clearly_positive_fraction'])}, clearly<0 {_pct(v['clearly_negative_fraction'])}"
             for k, v in r["advantage_vs_ai"].items() if v.get("n")),
         f"- AIFirst continuation (subset {a['n_analyzed_states']} states): Rule cross-fitted "
         f"{a['regret']['RulePolicy']['cross_fitted']['mean']:.3f} [{a['regret']['RulePolicy']['cross_fitted_ci']['lo']:.3f}, "
         f"{a['regret']['RulePolicy']['cross_fitted_ci']['hi']:.3f}], class {a['classification']['label']}",
         f"- learnability (leave-one-day-out ridge on 43-D obs): gain vs Rule {learn['learnable_gain_vs_rule']['mean']:.3f} "
         f"[{learn['learnable_gain_vs_rule']['lo']:.3f}, {learn['learnable_gain_vs_rule']['hi']:.3f}], changes Rule's action in "
         f"{_pct(learn['share_changed_from_rule'])}; always-AI gain vs Rule {learn['always_ai_gain_vs_rule']['mean']:.3f} "
         f"[{learn['always_ai_gain_vs_rule']['lo']:.3f}, {learn['always_ai_gain_vs_rule']['hi']:.3f}]",
         f"- feature separation HUMAN_BEST vs AI_BEST (n={sep.get('n_HUMAN_BEST')}/{sep.get('n_AI_BEST')}, standardised mean "
         f"difference): " + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(sep.get("smd", {}).items(), key=lambda x: -abs(x[1]))),
         ""]
    return L


def render_report(out: Path) -> str:
    rat = _r(out / "rationale" / "rationale.json")
    audit = _r(out / "audit" / "audit.json")
    prim = _r(out / "baselines" / "validation" / "primary_mixture.json")
    frozen = _r(out / "baselines" / "frozen" / "frozen_v1_1_policies.json")
    ev = _r(out / "baselines" / "evaluation_v1_1_cells.json")
    diag = _r(out / "baselines" / "validation" / "diagnostic_cell_specific.json")
    screen = _r(out / "headroom" / "screening" / "summaries.json")
    screen.pop("_wall_seconds", None)
    sel = _r(out / "headroom" / "confirmatory" / "selection.json")
    rows = headroom_map(screen, ev)
    (out / "headroom" / "screening" / "map.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    L = ["# Phase D — Twin V1.1 controlled reformulation", "",
         f"> Rationale `docs/twin_v1_1_rationale.md` frozen at {rat['frozen_at_utc']} (sha256 `{rat['sha256'][:16]}…`) "
         "before any result. V1.1 mechanisms are SYNTHETIC_ASSUMPTION_V1_1; scenarios are PREDEFINED_V1_SCENARIO.", "",
         "## D1 — 32-cell audit (seeds 1401–1410, mean per day)", "",
         "| cell | policy | vis. AI fail | escalations | hangups | handoff min | abandon % | human min | util | queue p90 | resolution % | false AI % | reward | flags |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(GRID, key=lambda c: c.order_key):
        for pol, e in audit["cells"][c.cell_id].items():
            m = lambda k: e[k]["mean"]
            L.append(f"| {c.cell_id} | {pol} | {m('visible_ai_failures'):.0f} | {m('escalations'):.0f} | {m('post_ai_failure_hangups'):.1f} | "
                     f"{m('handoff_minutes_added'):.0f} | {100 * m('abandonment_rate'):.2f} | {m('human_busy_minutes'):.0f} | "
                     f"{m('human_utilization'):.3f} | {m('human_queue_minutes_p90'):.2f} | {100 * m('resolution_rate_root'):.1f} | "
                     f"{100 * m('false_ai_rate'):.1f} | {m('total_reward'):.1f} | {', '.join(e['flags']) or '–'} |")
    L += ["", "## Baselines", "", f"PRIMARY mixture cells {frozen['mixture_cells']}, validation seeds {frozen['validation_seeds']}. "
          f"Frozen V1.1: `{frozen['frozen']}`. Frozen V1 (B2) RulePolicy shown as V1_RulePolicy.", ""]
    for name, v in prim["policies"].items():
        top = sorted(v["candidates"], key=lambda r: -r["mean_reward"])[:5]
        L.append(f"- {name}: selected `{v['selected']}`; top-5 mixture rewards "
                 + "; ".join(f"`{t['params']}` {t['mean_reward']:.1f}" for t in top))
    same = sum(1 for cid, d in diag.items() if {k: d["RulePolicy"][k] for k in ("theta", "queue_max", "congested_action")}
               == frozen["frozen"]["RulePolicy"])
    L += [f"- DIAGNOSTIC_CELL_SPECIFIC tuning: the per-cell RulePolicy equals the PRIMARY one in {same}/32 cells; "
          f"per-cell θ: `{sorted({d['ThresholdPolicy']['theta'] for d in diag.values()})}`; per-cell BestFixed: "
          f"`{sorted({d['BestFixedPolicy']['action'] for d in diag.values()})}`.", "",
          "Evaluation on seeds 1421–1430 (mean total reward per day):", "",
          "| cell | " + " | ".join(POL) + " | V1_RulePolicy | Rule − AIFirst | Rule resolution % | Rule root HIR | Rule human min | Rule abandon % |",
          "|---|" + "---|" * (len(POL) + 6)]
    for c in sorted(GRID, key=lambda c: c.order_key):
        t = ev["table"][c.cell_id]
        L.append(f"| {c.cell_id} | " + " | ".join(f"{t[p]['total_reward']['mean']:.1f}" for p in POL) +
                 f" | {t['V1_RulePolicy']['total_reward']['mean']:.1f} | "
                 f"{t['RulePolicy']['total_reward']['mean'] - t['AIFirstPolicy']['total_reward']['mean']:+.1f} | "
                 f"{100 * t['RulePolicy']['resolution_rate_root']['mean']:.1f} | {t['RulePolicy']['hir_root']['mean']:.3f} | "
                 f"{t['RulePolicy']['human_busy_minutes']['mean']:.0f} | {100 * t['RulePolicy']['abandonment_rate']['mean']:.2f} |")
    L += ["", "## D2 — headroom screening map (days 1421–1423, 10 states/day, K = 4, Rule continuation)", "",
          "Cell value: cross-fitted Rule regret per decision [95% day-cluster CI] and pre-registered class "
          "(CP = CLEARLY_POSITIVE, EM = EMERGING, NG = NEGLIGIBLE). Screening only locates regions.", ""]
    abbrev = {"CLEARLY_POSITIVE": "CP", "EMERGING": "EM", "NEGLIGIBLE": "NG"}
    for p in HANGUP:
        L += [f"### Immediate hangup p = {p:g}", ""]
        L += _grid_table(rows, p, lambda r: f"{r['rule_cf_regret']:+.2f} [{r['ci_lo']:+.2f},{r['ci_hi']:+.2f}] {abbrev[r['class']]}")
        L += ["", "TIE share / HUMAN_BEST share / mean J(HUMAN)−J(AI):", ""]
        L += _grid_table(rows, p, lambda r: f"{_pct(r['TIE_share'])} / {_pct(r['HUMAN_BEST_share'])} / {_f(r['mean_J_human_minus_ai'])}")
        L.append("")
    L += [f"D3 selection (pre-registered rules): `{sel['cells']}`; screened positive: `{sel['positive_screened']}`", ""]
    confirm = {}
    for cid, _ in sel["cells"]:
        d = out / "headroom" / "confirmatory" / cid
        if (d / "summary_rule.json").exists():
            confirm[cid] = {"rule": _r(d / "summary_rule.json"), "aifirst": _r(d / "summary_aifirst_continuation.json")}
    L += ["## D3 — confirmatory headroom (seeds 1421–1430, 30 states/day, K = 8)", ""]
    for cid, s in confirm.items():
        L += _confirm_block(cid, s)
    L += ["## Predefined V1 scenarios (V1.1 mechanisms OFF; frozen B2 policies)", ""]
    sev = _r(out / "scenarios" / "baselines_evaluation.json")["table"]
    per = _r(out / "scenarios" / "period_behaviour.json")
    for sname in ("SCENARIO_5", "SCENARIO_7", "SCENARIO_8"):
        t = sev[sname]
        L += [f"### {sname}", "", "| policy | reward | resolution % | root HIR | human min | util | abandon % | deadline miss % | AI % | HUMAN % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for p in POL:
            x = t[p]
            L.append(f"| {p} | {x['total_reward']['mean']:.1f} ± {x['total_reward']['std']:.1f} | {100 * x['resolution_rate_root']['mean']:.1f} | "
                     f"{x['hir_root']['mean']:.3f} | {x['human_busy_minutes']['mean']:.0f} | {x['human_utilization']['mean']:.3f} | "
                     f"{100 * x['abandonment_rate']['mean']:.2f} | {100 * x['missed_deadline_rate']['mean']:.2f} | "
                     f"{100 * x['action_share']['AI']:.1f} | {100 * x['action_share']['HUMAN']:.1f} |")
        pb = per.get(sname.lower(), {})
        for pol, days in pb.items():
            agg: dict[str, list] = {}
            for d in days:
                for period, v in d["periods"].items():
                    agg.setdefault(period, []).append(v)
            L.append(f"- {pol} by period: " + "; ".join(
                f"{period}: AI {100 * sum(v['action_share']['AI'] for v in vs) / len(vs):.0f}% / HUMAN "
                f"{100 * sum(v['action_share']['HUMAN'] for v in vs) / len(vs):.0f}%, recent AI fail "
                f"{sum(v['recent_ai_failure_rate'] for v in vs) / len(vs):.3f}, queue {sum(v['human_queue_size'] for v in vs) / len(vs):.2f}, "
                f"arrival rate {sum(v['recent_arrival_rate'] for v in vs) / len(vs):.0f}/h"
                for period, vs in sorted(agg.items())))
        hd = out / "scenarios" / sname.lower() / "headroom"
        if (hd / "summary_rule.json").exists():
            L.append("")
            L += _confirm_block(f"{sname} headroom", {"rule": _r(hd / "summary_rule.json"),
                                                       "aifirst": _r(hd / "summary_aifirst_continuation.json")})
            pr = _r(hd / "periods.json")["periods"]
            L += ["| period | n | " + " | ".join(LABELS) + " | Rule regret raw / cross-fit | AIFirst cross-fit | mean J(HUMAN)−J(AI) |",
                  "|---|---|" + "---|" * (len(LABELS) + 3)]
            for period, v in sorted(pr.items()):
                L.append(f"| {period} | {v['n']} | " + " | ".join(_pct(v["labels"][l]) for l in LABELS) +
                         f" | {v['rule_regret_raw']:.3f} / {v['rule_regret_cross_fitted']:.3f} | {v['aifirst_regret_cross_fitted']:.3f} | "
                         f"{v['mean_adv_human_vs_ai']:.3f} |")
        L.append("")
    if confirm:
        dec = final_decision(confirm)
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "reports" / "final_decision.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
        L += ["## Pre-registered decision", "", f"**{dec['verdict']}**", "", f"```\n{json.dumps(dec['evidence'], indent=1)}\n```", ""]
    return "\n".join(L) + "\n"


assert all(isinstance(c, Cell) for c in GRID)
