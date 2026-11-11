"""WorkforceManagerAgent: ONE agent = frozen RL routing core + agentic supervision loop.

FAST LOOP (every contact):  perceive -> build state -> PPO -> guardrails -> execute -> log / remember.
SUPERVISION LOOP (every 15 simulated minutes): get_kpis / forecast / get_recent_context -> EmergencyDetector;
only on an operational emergency: leakage-free Twin what-if -> recommendation -> manager approval by exception
-> temporary prudence (D minutes) -> automatic return to PPO. Technical PPO failure -> automatic,
separately logged TECHNICAL_FALLBACK.

Prudence comes in two forms (constructor ``prudence``): "rule" (Phase F, frozen evaluation: RulePolicy routes during
the D minutes) and "restricted_ppo" (human-in-the-loop product: the SAME PPO keeps routing, with AI removed from the
effective mask for risky requests, see agent/intervention.py; the what-if evaluates exactly that intervention).

PPO is trained offline in the Digital Twin and frozen; the agent never learns online.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from callverse_workforce.agent.actions import V1_1_REGISTRY, ActionRegistry
from callverse_workforce.agent.config import SupervisionConfig
from callverse_workforce.agent.emergency import EmergencyDetector
from callverse_workforce.agent.explanation import ExplanationService, evidence
from callverse_workforce.agent.guardrails import GuardrailManager, RoutingFailure
from callverse_workforce.agent.intervention import RESTRICTION, RESTRICTION_TEXT, RestrictedPPOPolicy, restrict_for_low_risk_ai
from callverse_workforce.agent.logger import DecisionLogger
from callverse_workforce.agent.manager import APPROVE, EXPIRE, REJECT, ManagerInterface, ManagerRequest
from callverse_workforce.agent.memory import MemoryStore
from callverse_workforce.agent.ppo_engine import PPODecisionEngine, TechnicalFailure
from callverse_workforce.agent.routing_policy import RoutingPolicyAdapter
from callverse_workforce.agent.state_builder import V1_1_CONTRACT, ObservationContract, StateBuilder
from callverse_workforce.agent.supervisor import KEEP_PPO, SupervisorController
from callverse_workforce.agent.tools import MaskProvider, ToolRegistry
from callverse_workforce.agent.types import AgentMode, ObservableEvent, Percept, PublicDayContext
from callverse_workforce.agent.whatif import WhatIfPlan, whatif_seed

M = AgentMode
ALLOWED = {
    M.NORMAL_PPO: {M.EMERGENCY_ANALYSIS, M.TECHNICAL_FALLBACK},
    M.EMERGENCY_ANALYSIS: {M.NORMAL_PPO, M.WAITING_MANAGER_APPROVAL, M.TECHNICAL_FALLBACK},
    M.WAITING_MANAGER_APPROVAL: {M.NORMAL_PPO, M.TEMPORARY_PRUDENCE, M.TECHNICAL_FALLBACK},
    M.TEMPORARY_PRUDENCE: {M.NORMAL_PPO, M.TECHNICAL_FALLBACK},
    M.TECHNICAL_FALLBACK: {M.NORMAL_PPO},
}
POLICY_BY_MODE = {M.NORMAL_PPO: "PPO", M.EMERGENCY_ANALYSIS: "PPO", M.WAITING_MANAGER_APPROVAL: "PPO",
                  M.TEMPORARY_PRUDENCE: "RULE_PRUDENCE", M.TECHNICAL_FALLBACK: "RULE_TECH_FALLBACK"}


class IllegalTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class RoutingOutcome:
    decision_id: str
    action: str
    policy_used: str
    mode: str
    guardrail: str | None
    latency_ms: float


class WorkforceManagerAgent:
    def __init__(self, policy: RoutingPolicyAdapter, fallback_policy: RoutingPolicyAdapter, cfg: SupervisionConfig,
                 public: PublicDayContext, base_config, manager: ManagerInterface, *, registry: ActionRegistry = V1_1_REGISTRY,
                 contract: ObservationContract = V1_1_CONTRACT, state_builder: StateBuilder | None = None,
                 explanation: ExplanationService | None = None, day_seed: int = 0, whatif_cache: dict | None = None,
                 keep_decision_records: bool = True, mask_provider=None, prudence: str = "rule") -> None:
        if prudence not in ("rule", "restricted_ppo"):
            raise ValueError(f"unknown prudence {prudence!r}")
        self.prudence = prudence
        self.policy, self.fallback = policy, fallback_policy
        # what-if intervention branch = the policy an approval would actually activate
        self.candidate = RestrictedPPOPolicy(policy) if prudence == "restricted_ppo" else fallback_policy
        self.cfg, self.public, self.manager, self.registry = cfg, public, manager, registry
        self.state_builder = state_builder or StateBuilder(contract)
        self.memory = MemoryStore()
        self.logger = DecisionLogger(self.memory, keep_decision_records)
        self.engine = PPODecisionEngine()
        self.guardrails = GuardrailManager(registry)
        self.detector = EmergencyDetector(cfg)
        self.supervisor = SupervisorController(cfg)
        self.explanation = explanation or ExplanationService()
        self.tools = ToolRegistry(self.memory, public, base_config, cfg, self.status, mask_provider or MaskProvider(base_config))
        self.day_seed = day_seed
        self.whatif_cache = whatif_cache if whatif_cache is not None else {}
        self.mode = M.NORMAL_PPO
        self.next_supervision = cfg.cadence_minutes
        self.cycle = 0
        self.cooldown_until = -math.inf
        self.prudence_end: float | None = None
        self.pending_request: ManagerRequest | None = None
        self.current_emergency: str | None = None
        self.ppo_unhealthy = False
        self.last_supervision: float | None = None
        self.routed_by = {"PPO": 0, "RULE_PRUDENCE": 0, "RULE_TECH_FALLBACK": 0}
        self.whatifs: dict[str, Any] = {}
        self._fallback_start: float | None = None
        self.supervision_errors = 0
        self.supervision_degraded = False
        self.last_supervision_error: str | None = None
        self.routing_failure: str | None = None
        self.intervention: dict[str, Any] | None = None  # active human-approved restriction (prudence="restricted_ppo")
        for p in (policy, fallback_policy):
            try:
                p.load()
            except Exception as exc:  # noqa: BLE001 - routing must continue
                if p is policy:
                    self.ppo_unhealthy = True
                    self.engine.last_error = f"load: {exc}"
        self.policy_version = policy.get_policy_metadata().policy_version
        self.memory.record("AgentStarted", 0.0, policy=policy.get_policy_metadata().to_dict(),
                           fallback=fallback_policy.get_policy_metadata().to_dict(), registry_version=registry.version,
                           observation_contract=self.state_builder.contract.__dict__ | {"decision_type": self.state_builder.contract.decision_type.value},
                           supervision=cfg.to_dict(), public_context=public.to_dict())
        if self.ppo_unhealthy:
            self._transition(M.TECHNICAL_FALLBACK, 0.0, "ppo_load_failure")
            self._fallback_start = 0.0

    # ------------------------------------------------------------------------------------------------
    def policy_for(self, mode: AgentMode) -> str:
        if mode is M.TEMPORARY_PRUDENCE and self.prudence == "restricted_ppo":
            return "PPO"
        return POLICY_BY_MODE[mode]

    def status(self) -> dict[str, Any]:
        return {"mode": self.mode.value, "policy": self.policy.get_policy_metadata().to_dict(),
                "active_policy": self.policy_for(self.mode), "ppo_loaded": not self.ppo_unhealthy,
                "ppo_health": {"errors": self.engine.errors, "last_error": self.engine.last_error, "unhealthy": self.ppo_unhealthy},
                "current_emergency": self.current_emergency, "last_supervision": self.last_supervision,
                "prudence_end": self.prudence_end, "fallback": self.mode is M.TECHNICAL_FALLBACK,
                "supervision": {"degraded": self.supervision_degraded, "errors": self.supervision_errors,
                                "last_error": self.last_supervision_error},
                "routing_failure": self.routing_failure,
                "intervention": None if self.intervention is None else dict(self.intervention)}

    def _transition(self, new: AgentMode, t: float, reason: str) -> None:
        if new is self.mode:
            return
        if new not in ALLOWED[self.mode]:
            raise IllegalTransition(f"{self.mode} -> {new}")
        self.memory.record("ModeTransition", t, **{"from": self.mode.value, "to": new.value, "reason": reason,
                                                    "active_policy_before": self.policy_for(self.mode),
                                                    "active_policy_after": self.policy_for(new)})
        self.mode = new

    def observe(self, events: list[ObservableEvent]) -> None:
        self.memory.add_events(events)

    # ------------------------------------------------------------------------------------------------
    # FAST LOOP
    # ------------------------------------------------------------------------------------------------
    def route(self, percept: Percept) -> RoutingOutcome:
        """One routing decision. Raises RoutingFailure (nothing executed, nothing recorded as decided) when no safe action
        exists; every other failure is contained: a supervision failure never stops routing, a PPO failure switches to the
        configured fallback policy."""
        t0 = time.perf_counter()
        now = percept.timestamp
        self.memory.observe_percept(percept)  # current state first: supervision must never read the previous observation
        if self.cfg.enabled:
            try:
                self._poll_manager(now)
                self._check_prudence_end(now)
                if now >= self.next_supervision:
                    self._supervise(percept, now)
                    self.supervision_degraded = False
            except Exception as exc:  # noqa: BLE001 - supervision is advisory: its failure must not stop routing
                self._supervision_failed(now, exc)
        mask, guard = self._effective_mask(percept, now)
        extra = None
        if self.intervention is not None and self.mode is M.TEMPORARY_PRUDENCE:
            mask, extra = self._intervention_mask(percept, mask, now)
        used = self.policy_for(self.mode)
        state = None
        if used == "PPO":
            try:
                state = self.state_builder.build(percept)
                action = self.guardrails.validate(self.engine.decide(self.policy, state, percept, mask), percept, mask)
            except TechnicalFailure as exc:
                self._enter_fallback(now, str(exc))
                used, guard = "RULE_TECH_FALLBACK", (guard + ";" if guard else "") + "technical_failure"
            except Exception as exc:  # noqa: BLE001 - encoder failure etc.
                self.engine.errors += 1
                self._enter_fallback(now, f"{type(exc).__name__}: {exc}")
                used, guard = "RULE_TECH_FALLBACK", (guard + ";" if guard else "") + "technical_failure"
        if used != "PPO":
            action = self._fallback_action(state, percept, mask, used, now)
        self.routed_by[used] += 1
        self.memory.add_decision(percept, action, self.mode.value, used)
        lat = (time.perf_counter() - t0) * 1000
        self.logger.decision(percept, action, self.mode.value, used, guard, lat, extra)
        return RoutingOutcome(percept.decision_id, action, used, self.mode.value, guard, lat)

    def _intervention_mask(self, percept: Percept, mask, now: float):
        """(final mask, DecisionRecord fields) under the active human-approved restriction: the effective mask (simulator
        AND business rules) AND the restriction. AI is only ever removed; nothing left -> RoutingFailure."""
        try:
            res = restrict_for_low_risk_ai(percept.observation, mask)
        except RoutingFailure as exc:
            self._routing_failed(now, percept, f"masque final vide sous l'intervention {self.intervention['id']} : {exc}")
            raise
        if res.ai_blocked:
            self.intervention["affected_decisions"] += 1
        self.intervention["decisions"] += 1
        return res.mask, {"effective_mask": list(mask.valid), "final_mask": list(res.mask.valid),
                          "intervention_id": self.intervention["id"], "ai_blocked_by_intervention": res.ai_blocked}

    def _effective_mask(self, percept: Percept, now: float):
        """(effective mask, guard label). The business-rule mask is recomputed and intersected with the simulator mask;
        a contradiction is traced (MaskMismatch). If the business-rule mask cannot be recomputed, or nothing is left
        allowed, routing fails explicitly (RoutingFailure): an action is never allowed because another mask allowed it."""
        try:
            recomputed = self.tools.get_action_mask(self.policy_version, percept.decision_type, percept.observation)
            mask, mismatch = self.guardrails.effective_mask(percept, recomputed)
        except Exception as exc:  # noqa: BLE001 - includes RoutingFailure (empty effective mask)
            self._routing_failed(now, percept, f"masque effectif non établi : {type(exc).__name__}: {exc}")
            raise (exc if isinstance(exc, RoutingFailure) else RoutingFailure(str(exc))) from exc
        if not mismatch:
            return mask, None
        self.memory.record("MaskMismatch", now, decision_id=percept.decision_id, contact_id=percept.contact_id,
                           simulator=list(percept.action_mask.valid), business_rules=list(recomputed.valid),
                           effective=list(mask.valid))
        return mask, "mask_mismatch"

    def _fallback_action(self, state, percept: Percept, mask, used: str, now: float) -> str:
        """Configured fallback policy (approved prudence rule or technical fallback), protected on its own. Its output must
        belong to the effective mask; otherwise, or if it raises, routing fails explicitly: no arbitrary action."""
        try:
            raw = self.fallback.predict(state, percept.observation, mask)
            return self.guardrails.validate(raw, percept, mask)
        except Exception as exc:  # noqa: BLE001
            msg = f"{used} : {type(exc).__name__}: {exc}"
            self._routing_failed(now, percept, msg)
            raise RoutingFailure(msg) from exc

    def _routing_failed(self, now: float, percept: Percept, error: str) -> None:
        self.routing_failure = error
        self.memory.record("RoutingFailure", now, decision_id=percept.decision_id, contact_id=percept.contact_id,
                           mode=self.mode.value, error=error, executed=False)

    def _supervision_failed(self, now: float, exc: Exception) -> None:
        """A failed control is recorded and visible (status: supervision degraded); routing continues with the active
        policy. An analysis interrupted half-way is abandoned (back to PPO, cooldown) rather than left pending."""
        self.supervision_errors += 1
        self.supervision_degraded = True
        self.last_supervision_error = f"{type(exc).__name__}: {exc}"
        self.memory.record("SupervisionFailure", now, error=self.last_supervision_error, mode=self.mode.value)
        if self.mode is M.EMERGENCY_ANALYSIS:
            self._transition(M.NORMAL_PPO, now, "supervision_failure")
            self.cooldown_until = now + self.cfg.cooldown_minutes
            self.current_emergency = None

    def _enter_fallback(self, now: float, error: str) -> None:
        self.ppo_unhealthy = True
        if self.pending_request is not None:
            self.memory.record("ManagerRequestCancelled", now, request_id=self.pending_request.request_id, reason="technical_fallback")
            self.pending_request = None
        if self.intervention is not None:  # the safety fallback takes over: the intervention ends here, traced
            self._end_intervention(now, "interrupted", "technical_fallback")
            self.prudence_end = None
        if self.mode is not M.TECHNICAL_FALLBACK:
            self._transition(M.TECHNICAL_FALLBACK, now, "ppo_technical_failure")
            self._fallback_start = now
            self.memory.record("TechnicalFallbackEvent", now, phase="start", error=error)

    # ------------------------------------------------------------------------------------------------
    # SUPERVISION LOOP
    # ------------------------------------------------------------------------------------------------
    def _poll_manager(self, now: float) -> None:
        if self.mode is not M.WAITING_MANAGER_APPROVAL or self.pending_request is None:
            return
        d = self.manager.poll(now)
        if d is None:
            return
        req = self.pending_request
        problem = ("autre proposition" if d.request_id != req.request_id
                   else "réponse antérieure à l'envoi" if d.response_t < req.sent_t
                   else "réponse dans le futur" if d.response_t > now
                   else "décision inconnue" if d.decision not in (APPROVE, REJECT, EXPIRE) else None)
        if problem is not None:  # never applied to the pending proposal: it keeps waiting for its own answer
            self.memory.record("ManagerResponseRejected", now, request_id=d.request_id, pending_request_id=req.request_id,
                               decision=d.decision, response_t=d.response_t, reason=problem)
            return
        self.memory.record("ManagerResponse", now, request_id=d.request_id, decision=d.decision, response_t=d.response_t,
                           profile=d.profile)
        self.pending_request = None
        if d.decision == APPROVE:
            # exactly the intervention of the approved proposal (its duration; legacy requests: the frozen D)
            duration = req.duration_minutes if req.duration_minutes is not None else self.cfg.d_minutes
            self._transition(M.TEMPORARY_PRUDENCE, now, "manager_approved")
            self.prudence_end = d.response_t + duration
            self.memory.record("PrudencePeriod", now, phase="start", start=d.response_t, scheduled_end=self.prudence_end,
                               request_id=d.request_id)
            if self.prudence == "restricted_ppo":
                self.intervention = {"id": f"I-{d.request_id}", "request_id": d.request_id, "start": d.response_t,
                                     "end": self.prudence_end, "duration_minutes": duration, "restriction": req.restriction,
                                     "text": RESTRICTION_TEXT, "decisions": 0, "affected_decisions": 0}
                self.memory.record("InterventionEvent", now, event="activated", intervention_id=self.intervention["id"],
                                   request_id=d.request_id, start=d.response_t, end=self.prudence_end,
                                   duration_minutes=duration, restriction=req.restriction)
        else:
            reason = "proposal_expired" if d.decision == EXPIRE else "manager_rejected"
            if self.prudence == "restricted_ppo":
                self.memory.record("InterventionEvent", now, event="expired_unanswered" if d.decision == EXPIRE else "refused",
                                   request_id=d.request_id)
            self._transition(M.NORMAL_PPO, now, reason)
            self.cooldown_until = now + self.cfg.cooldown_minutes
            self.current_emergency = None

    def _check_prudence_end(self, now: float) -> None:
        if self.mode is M.TEMPORARY_PRUDENCE and self.prudence_end is not None and now >= self.prudence_end:
            self.memory.record("PrudencePeriod", now, phase="end", actual_end=self.prudence_end)
            if self.intervention is not None:
                self._end_intervention(self.prudence_end, "expired", None)
            if self.ppo_unhealthy:
                self._transition(M.TECHNICAL_FALLBACK, now, "prudence_end_ppo_unhealthy")
            else:
                self._transition(M.NORMAL_PPO, now, "prudence_expired_auto_return")
            self.cooldown_until = self.prudence_end + self.cfg.cooldown_minutes
            self.prudence_end = None
            self.current_emergency = None

    def _end_intervention(self, t: float, event: str, reason: str | None) -> None:
        iv, self.intervention = self.intervention, None
        self.memory.record("InterventionEvent", t, event=event, intervention_id=iv["id"], request_id=iv["request_id"],
                           start=iv["start"], scheduled_end=iv["end"], actual_end=t, reason=reason,
                           decisions=iv["decisions"], affected_decisions=iv["affected_decisions"])

    def advance_time(self, now: float) -> None:
        """Time-driven supervisor state between two decisions (live clock): a manager answer or an expired proposal is
        applied, and an intervention ends at its scheduled time even if no decision arrives. Same protection as route():
        a failure here is recorded (supervision degraded), never raised to the clock."""
        if not self.cfg.enabled:
            return
        try:
            self._poll_manager(now)
            self._check_prudence_end(now)
        except Exception as exc:  # noqa: BLE001
            self._supervision_failed(now, exc)

    def cancel_intervention(self, intervention_id: str, now: float) -> None:
        """The manager ends the active intervention early (the intervention shown on the screen only)."""
        if self.intervention is None or self.mode is not M.TEMPORARY_PRUDENCE:
            raise LookupError("aucune intervention active")
        if intervention_id != self.intervention["id"]:
            raise LookupError(f"intervention {intervention_id} non active (active : {self.intervention['id']})")
        self.memory.record("PrudencePeriod", now, phase="cancel", actual_end=now, scheduled_end=self.prudence_end)
        self._end_intervention(now, "cancelled", "manager")
        self._transition(M.NORMAL_PPO, now, "intervention_cancelled")
        self.cooldown_until = now + self.cfg.cooldown_minutes
        self.prudence_end = None
        self.current_emergency = None

    def _health_check(self, percept: Percept, now: float) -> None:
        try:
            state = self.state_builder.build(percept)
            self.guardrails.validate(self.engine.decide(self.policy, state, percept), percept)
        except Exception:  # noqa: BLE001
            return
        self.ppo_unhealthy = False
        if self.mode is M.TECHNICAL_FALLBACK:
            self.memory.record("TechnicalFallbackEvent", now, phase="end", started=self._fallback_start)
            self._transition(M.NORMAL_PPO, now, "ppo_recovered")

    def _supervise(self, percept: Percept, now: float) -> None:
        self.cycle += 1
        lag = now - self.next_supervision
        while self.next_supervision <= now:
            self.next_supervision += self.cfg.cadence_minutes
        self.last_supervision = now
        kpis = self.tools.get_kpis(now)
        if kpis.get("observation_t") != now:  # never analyse silently a stale state of the centre
            raise RuntimeError(f"observation périmée : état du centre de t={kpis.get('observation_t')} lu à t={now}")
        fc = self.tools.forecast(now)
        ctx = self.tools.get_recent_context(now)
        if self.ppo_unhealthy:
            self._health_check(percept, now)
        eligible = self.mode is M.NORMAL_PPO and now >= self.cooldown_until
        em, hits = self.detector.detect(kpis, now, eligible)
        self.memory.record("SupervisionSnapshot", now, cycle=self.cycle, cadence_lag=lag, observation_t=kpis["observation_t"],
                           mode=self.mode.value,
                           active_policy=self.policy_for(self.mode), eligible=eligible, triggers=[list(h) for h in hits],
                           kpis={k: kpis[k] for k in ("ai_failure_rate_30", "n_ai_30", "abandonment_rate_30", "n_arrivals_30",
                                                      "human_queue_growth", "human_queue_size", "human_utilization")},
                           forecast=fc, context={"decisions": ctx["decisions"], "open_contacts": ctx["open_contacts"]})
        if em is None:
            return  # KEEP PPO, no Twin call
        self.current_emergency = em.emergency_id
        self.memory.record("EmergencyRecord", now, emergency_id=em.emergency_id, trigger=em.trigger, observed=em.observed,
                           threshold=em.threshold, all_triggers=[list(h) for h in em.all_triggers], debounce=em.debounce_count)
        self._transition(M.EMERGENCY_ANALYSIS, now, f"emergency_{em.trigger}")
        snap = self.tools.observable_snapshot(now, kpis)
        c = self.cfg
        plan = WhatIfPlan(c.approval_delay_minutes, c.d_minutes, c.horizon_minutes, c.world_model, c.mean_reversion_minutes,
                          c.f_ref, c.m_lo, c.n_min_ai)
        seeds = [whatif_seed(self.day_seed, self.cycle, k) for k in range(c.k_futures)]
        wid = f"W{self.cycle:03d}"
        key = (snap.sha256(), plan, tuple(seeds))
        wi = self.whatif_cache.get(key)
        if wi is None:
            wi = self.tools.run_twin_whatif(snap, self.policy, self.candidate, plan, seeds, self.state_builder, wid)
            self.whatif_cache[key] = wi
        self.whatifs[wid] = wi
        self.memory.record("WhatIfRecord", now, whatif_id=wid, emergency_id=em.emergency_id, snapshot_sha256=wi.snapshot_sha256,
                           seeds=wi.seeds, k=wi.k, horizon=c.horizon_minutes, d=c.d_minutes, world=wi.world,
                           branch_a=wi.branch_a, branch_b=wi.branch_b, diffs=wi.diffs, estimated_gain=wi.estimated_gain,
                           diff_sd=wi.diff_sd, t80_lower=wi.t80_lower, share_positive=wi.share_positive)
        rec = self.supervisor.recommend(em, wi)
        ev = evidence(em, kpis, fc, wi, rec, duration=c.d_minutes if rec.kind != KEEP_PPO else None)
        text, renderer = self.explanation.explain(ev)
        self.memory.record("RecommendationRecord", now, rec_id=rec.rec_id, emergency_id=em.emergency_id, whatif_id=wid,
                           recommendation=rec.kind, criteria=rec.criteria, estimated_gain=rec.estimated_gain)
        self.memory.record("ExplanationRecord", now, ref=rec.rec_id, evidence=ev, text=text, renderer=renderer)
        if rec.kind == KEEP_PPO:
            self._transition(M.NORMAL_PPO, now, "keep_ppo")
            self.cooldown_until = now + c.cooldown_minutes
            self.current_emergency = None
            return
        restricted = self.prudence == "restricted_ppo"
        req = ManagerRequest(f"Q{self.cycle:03d}", now, em.emergency_id, wid, rec.kind, wi.estimated_gain,
                             rec.criteria["confidence"], rec.criteria["valid_emergency"], rec.criteria["valid_whatif"], ev,
                             restriction=RESTRICTION if restricted else None, duration_minutes=c.d_minutes if restricted else None)
        self.manager.notify_manager(req)
        self.pending_request = req
        self.memory.record("ManagerRequest", now, request_id=req.request_id, emergency_id=em.emergency_id, whatif_id=wid,
                           estimated_gain=wi.estimated_gain,
                           **({"restriction": req.restriction, "duration_minutes": req.duration_minutes} if restricted else {}))
        self._transition(M.WAITING_MANAGER_APPROVAL, now, "recommendation_sent")
