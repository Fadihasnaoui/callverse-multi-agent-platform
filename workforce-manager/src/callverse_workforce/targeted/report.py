"""Markdown report for the targeted initial-routing headroom audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _r(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _ci(d: dict[str, Any]) -> str:
    return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]" if d else "–"


def _block(name: str, s: dict[str, Any]) -> list[str]:
    c = s["classification"]
    adv = s["crossfit_advantage"]
    L = [f"#### {name}", "",
         f"- states replayed {s['n_states']}, analysed {s['n_analyzed']} (reconstruction failures {s['n_failed_reconstruction']}), "
         f"unique days {s['unique_days']}",
         "- best-action shares (Phase C paired-t TIE logic): " + ", ".join(f"{k} {_pct(v)}" for k, v in s["label_shares"].items()),
         f"- clearly-best share of {s['target']}: {_ci(s['clearly_best_share_ci'])}",
         f"- cross-fitted {s['target']} advantage vs best competitor: mean {adv['mean']:.3f}, median {adv['median']:.3f}, "
         f"p10 {adv['p10']:.3f}, p90 {adv['p90']:.3f}; day-clustered 95% CI [{adv['ci']['lo']:.3f}, {adv['ci']['hi']:.3f}]",
         f"- best − second gap: median {s['best_second_gap']['median']:.3f}, p90 {s['best_second_gap']['p90']:.3f}",
         "- pairwise: " + "; ".join(f"{k}: mean {v['mean']:.3f} CI [{v['ci']['lo']:.3f}, {v['ci']['hi']:.3f}], clearly>0 "
                                    f"{_pct(v['clearly_positive_share'])}, clearly<0 {_pct(v['clearly_negative_share'])}"
                                    for k, v in sorted(s["pairwise"].items())),
         f"- **classification: {c['label']}**" + (" (LIMITED STATISTICAL POWER)" if c.get("limited_power") else ""), "",
         "Descriptive strata (clearly-best share / TIE share / mean cross-fitted advantage):", ""]
    for name_, groups in s["strata"].items():
        L.append(f"- {name_}: " + "; ".join(
            f"{g} (n={v['n']}): {_pct(v['clearly_best'])}" + (f" / {_pct(v['tie'])}" if "tie" in v else "") + f" / {v['mean_crossfit_adv']:+.3f}"
            for g, v in groups.items()))
    return L + [""]


def render(out: Path, res: dict[str, Any], dec: dict[str, str], rob: dict[str, str]) -> str:
    pre, seeds, prev = _r(out / "preregistration.json"), _r(out / "seed_audit.json"), _r(out / "prevalence.json")
    col, amend = _r(out / "state_collection.json"), _r(out / "replay_sample_amendment.json")
    mapping = _r(out / "observable_field_mapping.json")
    L = ["# Targeted initial-routing headroom audit (WAIT / CALLBACK)", "",
         "> Diagnostic only. Twin, reward, observation, masks, PPO E.1 and baselines frozen; no training; no V2.", "",
         "## State-generation protocol", "",
         f"- behaviour policy: `{pre['state_generation_policy']}` {pre['state_generation_policy_params']}",
         f"- environment: {pre['environment']}",
         f"- seed block {seeds['seed_block']} chosen by the collision audit (blocks tried: "
         f"{[(t['block'], t['n_collisions']) for t in seeds['tried']]}); sources checked: {seeds['sources_checked']}",
         f"- initial states inspected: {col['initial_states_inspected']} over {len(col['days'])} days "
         f"({col['stopped_because']}); families simulated: {col['families_simulated']}",
         f"- initial-routing definition: {pre['initial_state_definition']}",
         f"- replay amendment (written before any replay): {amend['reason']}", "",
         "## Observable-field mapping", ""] + [f"- {k} → `{v}`" for k, v in mapping["mapping"].items()] + [
         f"- unavailable fields: {mapping['unavailable_fields'] or 'none (all requested fields exist exactly)'}", ""]
    for subset in ("wait", "callback"):
        p = prev[subset]
        L += [f"## {subset.upper()} prevalence", "",
              f"- qualifying {p['qualifying']} / {prev['initial_states_inspected']} = **{100 * p['prevalence']:.3f}%**; unique days {p['unique_days']}",
              f"- families {p['families']}", f"- request types {p['request_types']}", f"- urgency {p['urgency']}",
              f"- human queue {p['human_queue']}; callbacks pending {p['callbacks_pending']}",
              f"- waiting_time_so_far {p['waiting_time_so_far']}; AI score {p['ai_score']}; deadline remaining {p['deadline_remaining']}", "",
              f"## {subset.upper()} counterfactual result (K = 16 paired futures)", ""]
        L += _block("RulePolicy continuation (primary)", res[f"{subset}_rule"])
        L += _block("AIFirst continuation (robustness)", res[f"{subset}_aifirst"])
    L += ["## Decision", "", f"Primary (Rule continuation): **{dec['verdict']}** — Case {dec['case']}: {dec['statement']}", "",
          f"Robustness (AIFirst continuation): {rob['verdict']} (Case {rob['case']})", "", dec["verdict"], ""]
    return "\n".join(L) + "\n"
