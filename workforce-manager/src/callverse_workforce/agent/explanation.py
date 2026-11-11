"""ExplanationService: structured evidence FIRST, text afterwards. Text never drives control logic.

The default renderer is a deterministic template. An LLM renderer may be plugged in (a callable taking the
evidence JSON and returning text); if it is absent or fails, the template is used. The LLM never routes,
detects, sets thresholds, computes gains, approves, or touches PPO / reward / Twin / guardrails.
"""

from __future__ import annotations

import json
from typing import Any, Callable


def evidence(emergency, kpis, forecast, whatif, recommendation, manager_decision=None, duration=None) -> dict[str, Any]:
    ev: dict[str, Any] = {"emergency": None if emergency is None else {
        "trigger": emergency.trigger, "observed": round(emergency.observed, 4), "threshold": emergency.threshold,
        "all_triggers": [list(t) for t in emergency.all_triggers]}}
    ev["kpis"] = {k: kpis[k] for k in ("ai_failure_rate_30", "abandonment_rate_30", "human_queue_size", "human_queue_growth",
                                       "human_utilization") if k in kpis}
    ev["forecast"] = forecast
    if whatif is not None:
        ev["twin"] = {"ppo_reward": whatif.branch_a["reward"]["mean"], "prudence_then_ppo_reward": whatif.branch_b["reward"]["mean"],
                      "estimated_gain": whatif.estimated_gain, "t80_lower": whatif.t80_lower,
                      "share_positive": whatif.share_positive, "K": whatif.k, "ai_multiplier": whatif.world["ai_multiplier"]}
    ev["recommendation"] = recommendation.kind if recommendation is not None else None
    ev["manager"] = manager_decision
    ev["duration_minutes"] = duration
    return ev


class TemplateExplainer:
    name = "template"

    def render(self, ev: dict[str, Any]) -> str:
        lines = []
        if ev.get("emergency"):
            e = ev["emergency"]
            lines.append(f"Emergency {e['trigger']}: observed {e['observed']} > threshold {e['threshold']}.")
        if ev.get("forecast"):
            lines.append(f"Forecast: {ev['forecast'].get('forecast_30min', 0):.1f} contacts in the next 30 min.")
        if ev.get("twin"):
            t = ev["twin"]
            lines.append(f"Twin ({t['K']} futures): PPO {t['ppo_reward']:.1f} vs prudence-then-PPO {t['prudence_then_ppo_reward']:.1f}; "
                         f"estimated gain {t['estimated_gain']:+.1f} (80% lower {t['t80_lower']:+.1f}).")
        if ev.get("recommendation"):
            lines.append(f"Recommendation: {ev['recommendation']}.")
        if ev.get("manager"):
            lines.append(f"Manager: {ev['manager']}.")
        if ev.get("duration_minutes"):
            lines.append(f"Duration: {ev['duration_minutes']:g} min, then automatic return to PPO.")
        return " ".join(lines)


class LLMExplainer:
    """Optional: wraps a text generator (e.g. an LLM client). Disabled in DEV/TEST."""

    name = "llm"

    def __init__(self, generate: Callable[[str], str]) -> None:
        self.generate = generate

    def render(self, ev: dict[str, Any]) -> str:
        return self.generate(json.dumps(ev, sort_keys=True, default=str))


class ExplanationService:
    def __init__(self, renderer=None) -> None:
        self.template = TemplateExplainer()
        self.renderer = renderer or self.template

    def explain(self, ev: dict[str, Any]) -> tuple[str, str]:
        try:
            return self.renderer.render(ev), self.renderer.name
        except Exception:  # noqa: BLE001 - LLM unavailable -> deterministic template
            return self.template.render(ev), self.template.name
