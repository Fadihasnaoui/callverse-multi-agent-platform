"""Phase C markdown report, rendered from the persisted artifacts only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from callverse_workforce.headroom import experiment as H
from callverse_workforce.headroom.loads import LOAD_LEVELS

LABELS = H.LABELS
POLICIES = ("RulePolicy", "AIFirstPolicy", "ThresholdPolicy")


def _read(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _f(x: Any, nd: int = 2) -> str:
    return "–" if x is None else f"{x:.{nd}f}"


def _labels_row(name: str, s: dict[str, Any]) -> str:
    d = s["label_distribution"]
    return f"| {name} | {s['n_analyzed_states']} | " + " | ".join(_pct(d[l]) for l in LABELS) + " |"


def _regret_rows(name: str, s: dict[str, Any]) -> list[str]:
    rows = []
    for pol in POLICIES:
        r = s["regret"][pol]
        rows.append(
            f"| {name} | {pol} | {_f(r['raw']['mean'])} [{_f(r['raw_ci']['lo'])}, {_f(r['raw_ci']['hi'])}] | "
            f"{_f(r['raw']['median'])} | {_f(r['raw']['p90'])} | {_f(r['clear']['mean'])} | "
            f"{_f(r['cross_fitted']['mean'])} [{_f(r['cross_fitted_ci']['lo'])}, {_f(r['cross_fitted_ci']['hi'])}] | "
            f"{_pct(r['fraction_policy_is_best'])} | {_pct(r['fraction_raw_below_1'])} | "
            f"{r['rough_daily_raw']:.0f} / {r['rough_daily_cross_fitted']:.0f} |")
    return rows


def _summary_block(title: str, summaries: dict[str, dict[str, Any]]) -> list[str]:
    L = [f"### {title}", "", "Best-action / tie distribution:", "",
         "| load | states | " + " | ".join(LABELS) + " |", "|---|---|" + "---|" * len(LABELS)]
    L += [_labels_row(n, s) for n, s in summaries.items()]
    L += ["", "Regret per sampled decision (raw = max_a J − J(π); clear = significance-filtered; cross-fitted = "
          "selection-bias-free). Bracketed: 95% day-clustered bootstrap CI. Last column: "
          f"{H.ROUGH_LABEL} (raw / cross-fitted).", "",
          "| load | policy | raw mean [CI] | raw median | raw p90 | clear mean | cross-fitted mean [CI] | π is best | raw < 1 | rough daily |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        L += _regret_rows(n, s)
    L += ["", "Near ties (best − second-best mean, descriptive only) and mean action values:", "",
          "| load | < 5 | < 10 | < 25 | gap median | gap p90 | J oracle (per-state best, not deployable) | J Rule | J AIFirst | J Threshold |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        nt, g, j = s["near_tie_fractions"], s["best_minus_second"], s["mean_J"]
        L.append(f"| {n} | {_pct(nt['lt_5'])} | {_pct(nt['lt_10'])} | {_pct(nt['lt_25'])} | {_f(g['median'])} | "
                 f"{_f(g['p90'])} | {_f(j['oracle_best_forced_action'])} | {_f(j['RulePolicy_action'])} | "
                 f"{_f(j['AIFirstPolicy_action'])} | {_f(j['ThresholdPolicy_action'])} |")
    L += ["", "Action advantage against AI (states where both are valid):", "",
          "| load | alternative | n | mean | median | p10 | p90 | > 0 | clearly > 0 | clearly < 0 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, s in summaries.items():
        for alt, a in s["advantage_vs_ai"].items():
            if a.get("n"):
                L.append(f"| {n} | J({alt})−J(AI) | {a['n']} | {_f(a['mean'])} | {_f(a['median'])} | {_f(a['p10'])} | "
                         f"{_f(a['p90'])} | {_pct(a['positive_fraction'])} | {_pct(a['clearly_positive_fraction'])} | "
                         f"{_pct(a['clearly_negative_fraction'])} |")
    return L + [""]


def _context(s: dict[str, Any], load: str) -> list[str]:
    L = [f"#### {load}", "", "| group | value | n | AI_BEST | HUMAN_BEST | WAIT_BEST | CALLBACK_BEST | TIE | "
         "mean J(HUMAN)−J(AI) | mean J(WAIT)−J(AI) | mean J(CB)−J(AI) | Rule regret raw / cross-fit |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for group, values in s["context"].items():
        for value, g in values.items():
            lab, adv = g["labels"], g["mean_adv_vs_ai"]
            L.append(f"| {group} | {value} | {g['n']} | " + " | ".join(_pct(lab[l]) for l in LABELS) +
                     f" | {_f(adv['HUMAN'])} | {_f(adv['WAIT'])} | {_f(adv['CALLBACK'])} | "
                     f"{_f(g['mean_rule_regret_raw'])} / {_f(g['mean_rule_regret_cross_fitted'])} |")
    return L + [""]


def _medium_subset_summary(out: Path, load: str) -> dict[str, Any] | None:
    ids_path = out / "score_noise_sensitivity" / "medium_subset" / f"{load.lower()}.json"
    if not ids_path.exists():
        return None
    ids = set(_read(ids_path))
    states = [s for s in _read(out / "sampled_states" / f"{load.lower()}.json")["states"] if s["state_id"] in ids]
    analyses = [a for a in _read(out / "primary" / load.lower() / "analysis.json") if a["state_id"] in ids]
    results = [r for r in _jsonl(out / "primary" / load.lower() / "branches.jsonl") if r["state_id"] in ids]
    metas = _read(out / "sampled_states" / f"{load.lower()}.json")["meta"]
    return H.summarize(states, analyses, metas, results, f"{load}|MEDIUM_SUBSET")


def render_report(out: Path) -> str:
    plan, bench = _read(out / "replay_plan.json"), _read(out / "benchmark.json")
    manifests = _read(out / "nested_load_manifests.json")
    primary = {l: _read(out / "primary" / l.lower() / "summary.json") for l in LOAD_LEVELS}
    L = ["# Phase C — forced-action headroom analysis", "",
         "> Diagnostic only. Frozen V1 Twin and frozen B2 policies; no learning. NORMAL_X1_3 / NORMAL_X1_5 are "
         "`HEADROOM_DIAGNOSTIC_ONLY`. Score-noise levels are `SYNTHETIC_DIAGNOSTIC_NOISE_LEVEL`.", "",
         f"Seeds: `{plan['seeds']}`; K = {plan['K']} paired futures per state-action; pre-target policy: frozen "
         f"RulePolicy `{plan['frozen_b2']['RulePolicy']}`.", "",
         "## Computation", "",
         "| load | benchmark s/branch | planned branches (max) | actual branches | states | reconstruction OK | wall s (C1) |",
         "|---|---|---|---|---|---|---|"]
    for l in LOAD_LEVELS:
        p, s = plan["per_load"][l], primary[l]
        L.append(f"| {l} | {bench[l]['seconds_per_branch']:.3f} | {p['branches']} ({p['max_branches_if_all_valid']}) | "
                 f"{s['branch_replays']} | {s['n_sampled_states']} | {_pct(s['reconstruction_success_rate'])} | "
                 f"{s['wall_seconds']:.0f} |")
    ext = _read(out / "primary" / "extension_decision.json")
    L += ["", f"Pre-registered extension rule: `{plan['pre_registered_extension_rule']}` → extend: **{ext['extend']}** "
          f"({ {k: (round(v['ci']['lo'], 3), round(v['ci']['hi'], 3)) for k, v in ext['per_load'].items()} }).", "",
          "## Nested load construction", "", "| seed | NORMAL | X1.3 | X1.5 | NORMAL ⊂ X1.3 | X1.3 ⊂ X1.5 | shared identical |",
          "|---|---|---|---|---|---|---|"]
    for m in manifests:
        c = m["counts"]
        L.append(f"| {m['seed']} | {c['NORMAL']} | {c['NORMAL_X1_3']} | {c['NORMAL_X1_5']} | {m['normal_subset_of_x1_3']} | "
                 f"{m['x1_3_subset_of_x1_5']} | {m['shared_contacts_identical']} |")
    L += ["", "## Primary (C1): MEDIUM noise, RulePolicy continuation", ""]
    L += _summary_block("C1 primary", primary)
    L += ["### Sample and mask freedom", "", "| load | n_valid 4/3/2 | masked action counts | strata | urgency |", "|---|---|---|---|---|"]
    for l, s in primary.items():
        nv = s["mask_freedom"]["n_valid"]
        L.append(f"| {l} | {nv.get('4', nv.get(4, 0))}/{nv.get('3', nv.get(3, 0))}/{nv.get('2', nv.get(2, 0))} | "
                 f"{s['mask_freedom']['masked_action_counts']} | {s['sample_distribution']['time_of_day']} | "
                 f"{s['sample_distribution']['urgency']} |")
    L += ["", "Further sample distributions (request type / AI-score band / utilization band):", ""]
    for l, s in primary.items():
        sd = s["sample_distribution"]
        L.append(f"- {l}: {sd['request_type']} / {sd['ai_score']} / {sd['human_utilization']}")
    L += ["", "### Context dependence (C1)", ""]
    for l, s in primary.items():
        L += _context(s, l)
    L += ["### Rule vs AI-first disagreement states (secondary diagnostic subset of the primary sample)", "",
          "| load | n | share | action pairs | verdicts (Rule − AIFirst, paired t) | mean Rule−AIFirst | mean Rule regret | mean AIFirst regret |",
          "|---|---|---|---|---|---|---|---|"]
    for l, s in primary.items():
        d = s["disagreement"]
        L.append(f"| {l} | {d['n']} | {_pct(d['share_of_states'])} | {d['action_pairs']} | {d['verdicts']} | "
                 f"{_f(d['rule_minus_aifirst'].get('mean'))} | {_f(d['mean_rule_regret'])} | {_f(d['mean_aifirst_regret'])} |")
    L.append("")
    c2_paths = {l: out / "continuation_sensitivity" / l.lower() / "summary.json" for l in LOAD_LEVELS}
    if all(p.exists() for p in c2_paths.values()):
        L += ["## C2: AIFirstPolicy continuation (same states)", ""]
        L += _summary_block("C2 AIFirst continuation", {l: _read(p) for l, p in c2_paths.items()})
    noise_blocks = {}
    for l in LOAD_LEVELS:
        med = _medium_subset_summary(out, l)
        if med is None:
            continue
        noise_blocks[f"{l} MEDIUM (subset)"] = med
        for n in ("LOW", "HIGH"):
            p = out / "score_noise_sensitivity" / n.lower() / l.lower() / "summary.json"
            if p.exists():
                noise_blocks[f"{l} {n} (linked)"] = _read(p)
    if noise_blocks:
        L += ["## C3: score-noise robustness", "",
              f"Deterministic stratified subset of 10 of the 30 primary states per day ({H.SUBSET_QUOTA}). LOW/HIGH "
              "use LINKED states: the noise change alters observed scores, Rule decisions and therefore the "
              "trajectory, so each primary state is matched to the same contact/attempt when possible, else to the "
              "nearest-time eligible decision of the same stratum.", ""]
        links = {}
        for n in ("LOW", "HIGH"):
            for l in LOAD_LEVELS:
                p = out / "score_noise_sensitivity" / n.lower() / l.lower() / "linked_states.json"
                if p.exists():
                    ls = _read(p)
                    links[f"{l} {n}"] = (sum(x["link_type"] == "same_contact" for x in ls), len(ls),
                                         max(x["link_time_gap"] for x in ls))
        L += [f"Link quality (same-contact links / states, max time gap min): `{links}`", ""]
        L += _summary_block("C3 score noise", noise_blocks)
    ph = out / "threshold_posthoc" / "threshold_posthoc.json"
    if ph.exists():
        d = _read(ph)
        L += ["## POST_HOC_THRESHOLD_SENSITIVITY", "", d["note"], "",
              f"Seeds `{d['seeds']}`; frozen B2 θ remains **{d['frozen_b2_theta']}**.", "",
              "| θ | mean reward ± std | AI share | resolution % | false AI % |", "|---|---|---|---|---|"]
        for r in d["rows"]:
            L.append(f"| {r['theta']:.1f} | {r['mean_reward']:.1f} ± {r['std_reward']:.1f} | {_pct(r['ai_share'])} | "
                     f"{_pct(r['resolution_rate_root'])} | {_pct(r['false_ai_rate'])} |")
        L.append("")
    return "\n".join(L) + "\n"
