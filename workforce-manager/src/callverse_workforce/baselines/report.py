"""Markdown rendering for Phase B2 validation and evaluation artifacts."""

from __future__ import annotations

from typing import Any

from callverse_workforce.baselines.metrics import REWARD_KEYS
from callverse_workforce.baselines.policies import BASELINE_NAMES
from callverse_workforce.domain.enums import ACTION_ORDER


def _ms(a: dict[str, float], pct: bool = False, nd: int = 1) -> str:
    k = 100.0 if pct else 1.0
    return f"{a['mean'] * k:.{nd}f} ± {a['std'] * k:.{nd}f}"


def validation_table(block: dict[str, Any]) -> list[str]:
    keys = list(block["candidates"][0]["params"])
    head = "| " + " | ".join(keys) + " | mean reward | std | resolution % | root HIR | abandon % | wait p90 | AI % | HUMAN % | WAIT % | CALLBACK % |"
    L = [head, "|" + "---|" * (len(keys) + 10)]
    for row in block["candidates"]:
        p = row["params"]
        mark = " **(selected)**" if p == block["selected"] else ""
        s = row["action_share"]
        L.append("| " + " | ".join(str(p[k]) for k in keys) +
                 f" | {row['mean_reward']:.1f}{mark} | {row['std_reward']:.1f} | {row['resolution_rate_root'] * 100:.1f} | "
                 f"{row['hir_root']:.3f} | {row['abandonment_rate'] * 100:.2f} | {row['online_wait_p90']:.2f} | "
                 + " | ".join(f"{s[a.value] * 100:.1f}" for a in ACTION_ORDER) + " |")
    return L


def render_validation(val: dict[str, Any]) -> str:
    L = ["# Phase B2 — validation (NORMAL seeds only)", "", f"Seeds: `{val['validation_seeds']}`", "",
         "Selection metric: mean center-level total reward; ties -> first candidate in grid order.", ""]
    for title, key in (("ThresholdPolicy", "threshold"), ("RulePolicy", "rule"), ("BestFixedPolicy", "best_fixed")):
        L += [f"## {title}", ""] + validation_table(val[key]) + ["", f"Selected: `{val[key]['selected']}`", ""]
    L += ["## Frozen parameters", "", f"```\n{val['frozen']}\n```", ""]
    return "\n".join(L)


def main_table(ev: dict[str, Any]) -> list[str]:
    L = ["| Policy | Reward | Resolution % | Root HIR | Instance HIR | Human min | Wait p90 | Abandon % | "
         "Deadline miss % | False AI % | AI % | HUMAN % | WAIT % | CALLBACK % |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in BASELINE_NAMES:
        a = ev["policies"][name]["aggregate"]
        sh = a["action_share"]
        L.append(
            f"| {name} | {_ms(a['total_reward'])} | {_ms(a['resolution_rate_root'], True)} | "
            f"{_ms(a['hir_root'], nd=3)} | {_ms(a['hir_instance'], nd=3)} | {_ms(a['human_busy_minutes'], nd=0)} | "
            f"{_ms(a['online_wait_p90'], nd=2)} | {_ms(a['abandonment_rate'], True, 2)} | "
            f"{_ms(a['missed_deadline_rate'], True, 2)} | {_ms(a['false_ai_rate'], True)} | "
            + " | ".join(_ms(sh[x.value], True) for x in ACTION_ORDER) + " |"
        )
    return L


def reward_table(ev: dict[str, Any]) -> list[str]:
    L = ["| Policy | " + " | ".join(REWARD_KEYS) + " |", "|---|" + "---|" * len(REWARD_KEYS)]
    for name in BASELINE_NAMES:
        r = ev["policies"][name]["aggregate"]["reward"]
        L.append(f"| {name} | " + " | ".join(f"{r[k]['mean']:.1f}" for k in REWARD_KEYS) + " |")
    return L


def detail_table(ev: dict[str, Any]) -> list[str]:
    keys = [("human_routing_rate", True, "HUMAN routing %"), ("human_utilization", True, "utilization %"),
            ("unfinished_rate", True, "unfinished %"), ("online_wait_mean", False, "wait mean"),
            ("ai_visible_failure_rate", True, "AI visible fail %"), ("ai_escalations", False, "AI escalations"),
            ("wait_limit_count", False, "WAIT-limit"), ("abandonments_in_wait", False, "aband. in WAIT"),
            ("callback_success", False, "cb success"), ("callback_dropout", False, "cb dropout"),
            ("callback_max_attempts", False, "cb max att."), ("callback_window_expired", False, "cb window"),
            ("callback_unfinished", False, "cb unfinished"), ("callback_human_minutes", False, "cb human min"),
            ("mean_valid_actions", False, "mean valid")]
    L = ["| Policy | " + " | ".join(k[2] for k in keys) + " | 4/3/2/1 valid % |", "|---|" + "---|" * (len(keys) + 1)]
    for name in BASELINE_NAMES:
        a = ev["policies"][name]["aggregate"]
        v = a["valid_action_pct"]
        L.append(f"| {name} | " + " | ".join(
            f"{a[k]['mean'] * (100 if p else 1):.2f}" for k, p, _ in keys) +
            " | " + "/".join(f"{v[k]['mean'] * 100:.1f}" for k in ("4", "3", "2", "1")) + " |")
    return L


def fairness_table(ev: dict[str, Any], group: str) -> list[str]:
    first = ev["policies"][BASELINE_NAMES[0]]["aggregate"]["fairness"][group]
    L = [f"| Policy | {group} | resolution % | HIR | wait mean | wait p90 | abandon % |", "|---|---|---|---|---|---|---|"]
    for name in BASELINE_NAMES:
        f = ev["policies"][name]["aggregate"]["fairness"][group]
        for key in first:
            g = f[key]
            L.append(f"| {name} | {key} | {g['resolution_rate']['mean'] * 100:.1f} | {g['hir_instance']['mean']:.3f} | "
                     f"{g['wait_mean']['mean']:.2f} | {g['wait_p90']['mean']:.2f} | {g['abandonment_rate']['mean'] * 100:.2f} |")
    return L


def render_evaluation(ev: dict[str, Any]) -> str:
    p = ev["paired"]
    L = [f"# Phase B2 — {ev['label']} evaluation ({len(ev['seeds'])} paired days)", "",
         f"Purpose: `{ev['purpose']}`. Seeds: `{ev['seeds']}`. Values: mean ± std over days. "
         "Resolution % is root-level (customer problems correctly resolved).", ""]
    if ev["purpose"] == "DEVELOPMENT_LOAD_CHECK_ONLY":
        L += ["> DEVELOPMENT_LOAD_CHECK_ONLY — stressed development configuration, not a scientific scenario.", ""]
    L += ["## Baseline table", ""] + main_table(ev) + ["", "## RewardBreakdown (mean per day)", ""] + reward_table(ev)
    L += ["", "## Operational details (mean per day)", ""] + detail_table(ev)
    L += ["", "## Diagnostics", "", "| Policy | dominant action | share | >90% | >95% | CALLBACK share | CALLBACK over-use |",
          "|---|---|---|---|---|---|---|"]
    for name in BASELINE_NAMES:
        d = ev["policies"][name]["diagnostics"]
        L.append(f"| {name} | {d['dominant_action']} | {d['dominant_action_share'] * 100:.1f}% | {d['dominant_over_90']} | "
                 f"{d['dominant_over_95']} | {d['callback_share'] * 100:.1f}% | {d['callback_overuse']} |")
    for name in BASELINE_NAMES:
        d = ev["policies"][name]["diagnostics"]
        for w in d["warnings"]:
            L.append(f"- **{name}**: {w}")
        if d.get("callback_detail"):
            L.append(f"  - callback detail: `{ {k: round(v, 4) for k, v in d['callback_detail'].items()} }`")
    L += ["", "## Paired per-day variability", "", f"Best by mean reward: **{p['best_by_mean_reward']}**. "
          f"Per-day winners: `{p['per_day_winner_counts']}`", "",
          "| Policy | diff vs best mean ± std | min | max | days better | days worse |", "|---|---|---|---|---|---|"]
    for name, d in p["diff_vs_best"].items():
        L.append(f"| {name} | {d['mean']:.1f} ± {d['std']:.1f} | {d['min']:.1f} | {d['max']:.1f} | "
                 f"{d['days_better_than_best']} | {d['days_worse_than_best']} |")
    L += ["", "## Fairness diagnostics (descriptive only, no hypothesis tests)", "", "### By language/dialect", ""]
    L += fairness_table(ev, "language") + ["", "### By channel", ""] + fairness_table(ev, "channel") + [""]
    return "\n".join(L)
