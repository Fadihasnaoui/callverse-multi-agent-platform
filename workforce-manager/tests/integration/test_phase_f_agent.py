"""Phase F — Workforce Manager agentic runtime: required Stage-2 tests (1-40) + V2-readiness checks.

Only Phase F DEV seeds (3000-3159) are simulated here; no TEST seed and nothing in 2240-2999 is ever opened.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import sys

import numpy as np
import pytest

from callverse_workforce.agent import evaluation as AE
from callverse_workforce.agent.actions import V1_1_REGISTRY, ActionRegistry, ActionSpec
from callverse_workforce.agent.config import PPO_ALONE, SupervisionConfig
from callverse_workforce.agent.executor import RoutingExecutor
from callverse_workforce.agent.explanation import ExplanationService, LLMExplainer
from callverse_workforce.agent.manager import APPROVE, REJECT, ManagerDecision, ManagerRequest, SimulatedManager
from callverse_workforce.agent.perception import SAFE_DATA_KEYS, SimulationPerception
from callverse_workforce.agent.routing_policy import PPOPolicy, RulePolicyAdapter
from callverse_workforce.agent.state_builder import V1_1_CONTRACT, EncodedState, ObservationContract, StateBuilder
from callverse_workforce.agent.supervisor import KEEP_PPO, RECOMMEND, SupervisorController
from callverse_workforce.agent.types import AgentMode, DecisionType, Percept
from callverse_workforce.agent.whatif import WHATIF_SEED_BASE, WhatIfPlan, build_world, hypothetical_config, simulate_branch, whatif_seed, world_parameters
from callverse_workforce.agent.workforce_manager import ALLOWED, IllegalTransition, WorkforceManagerAgent
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.domain.enums import ACTION_ORDER
from callverse_workforce.encoding import OBSERVATION_DIM, ObservationEncoder
from callverse_workforce.phase_d.grid import MIXTURE_CELLS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.seeds import FINAL_V1_1_SEEDS, PHASE_F_DEV_SEEDS, PHASE_F_TEST_SEEDS, V2_RESERVED_SEEDS, SeedPolicyError

OUT = PROJECT_ROOT / "artifacts" / "phase_f"
AGENT_DIR = PROJECT_ROOT / "src" / "callverse_workforce" / "agent"
BASE = str(PROJECT_ROOT / "configs" / "normal_day.yaml")
MODELS = json.loads((PROJECT_ROOT / "artifacts" / "final_test_v1_1" / "model_hashes.json").read_text(encoding="utf-8"))["models"]
MODEL = MODELS["e1_s500"]["path"]
RULE = json.loads((PROJECT_ROOT / "artifacts/v1_1/baselines/frozen/frozen_v1_1_policies.json").read_text(encoding="utf-8"))["frozen"]["RulePolicy"]
DAYS = {d.seed: d for d in AE.phase_f_days(PHASE_F_DEV_SEEDS, 20)}
NORMAL_DAY, AI_DAY = DAYS[3000], DAYS[3080]
MONITOR = SupervisionConfig(t_ai=None, t_ab=None, t_q=None)
ALWAYS = SupervisionConfig(t_ai=0.0, n_min_ai=1, g_min=-1e9, k_futures=2, confidence="share60", d_minutes=30.0)


def _code(path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def _attrs(path) -> set[str]:
    return {n.attr for n in ast.walk(_code(path)) if isinstance(n, ast.Attribute)}


def _calls(path) -> set[str]:
    out = set()
    for n in ast.walk(_code(path)):
        if isinstance(n, ast.Call):
            out.add(n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", ""))
    return out


RUNTIME_MODULES = [p for p in AGENT_DIR.glob("*.py") if p.name != "evaluation.py"]


class StubManager:
    def __init__(self, decision: str, delay: float = 5.0) -> None:
        self.decision, self.delay, self.q = decision, delay, []

    def notify_manager(self, r: ManagerRequest) -> None:
        self.q.append(ManagerDecision(r.request_id, self.decision, r.sent_t + self.delay, "stub"))

    def poll(self, now: float):
        return self.q.pop(0) if self.q and self.q[0].response_t <= now else None


class Forbidden:
    """Any call fails the test (proves a component is NOT used in the fast loop)."""

    name = "forbidden"

    def __getattr__(self, item):
        raise AssertionError(f"forbidden component used: {item}")


def drive(day, cfg, until=1e9, manager=None, policy=None, fallback=None, explanation=None, mask_provider=None):
    live = EV._config(BASE, day)
    base = AE._base(BASE)
    from callverse_workforce.agent.executor import simulation_handlers
    from callverse_workforce.agent.perception import public_context_from
    from callverse_workforce.twin.environment import RoutingEnvironment

    env = RoutingEnvironment(live)
    dp = env.reset(day.seed)
    public = public_context_from(live, base)
    per = SimulationPerception(env, public)
    agent = WorkforceManagerAgent(policy or PPOPolicy(MODEL), fallback or RulePolicyAdapter(RULE), cfg, public, base,
                                  manager or SimulatedManager(cfg.manager, cfg.g_min, cfg.approval_delay_minutes),
                                  day_seed=day.seed, explanation=explanation, mask_provider=mask_provider)
    ex = RoutingExecutor(V1_1_REGISTRY, simulation_handlers(V1_1_REGISTRY, env))
    outs, events = [], []
    while dp is not None and dp.timestamp < until:
        ev = per.new_events()
        events += ev
        agent.observe(ev)
        o = agent.route(per.percept(dp))
        outs.append((dp.timestamp, o))
        dp = ex.execute(o.action).next_decision
    return agent, env, outs, events


@pytest.fixture(scope="module")
def normal_monitor():
    return drive(NORMAL_DAY, MONITOR, until=300.0)


@pytest.fixture(scope="module")
def ai_always():
    return drive(AI_DAY, ALWAYS, until=420.0, manager=StubManager(APPROVE))


# 1-6 -------------------------------------------------------------------------------------------------------
def test_01_e1_s500_loads():
    p = PPOPolicy(MODEL)
    p.load()
    md = p.get_policy_metadata()
    assert md.name == "e1_s500" and md.policy_version == "V1_1" and md.observation_schema == "V1_1_43D"


def test_02_model_hash_unchanged():
    assert hashlib.sha256(open(MODEL, "rb").read()).hexdigest() == MODELS["e1_s500"]["sha256"]
    p = PPOPolicy(MODEL)
    p.load()
    assert p.get_policy_metadata().model_sha256 == MODELS["e1_s500"]["sha256"]


def test_03_encoder_unchanged(normal_monitor):
    from callverse_workforce.phase_e import env as research_env

    agent, *_ = normal_monitor
    sb = StateBuilder()
    assert type(sb.encoder) is type(research_env._ENCODER)
    pct = agent.memory.last_percept
    assert np.array_equal(sb.build(pct).vector, research_env._ENCODER.encode(pct.observation))


def test_04_43d_representation():
    assert V1_1_CONTRACT.dim == OBSERVATION_DIM == 43 and V1_1_CONTRACT.observation_schema == "V1_1_43D"
    assert ObservationEncoder.dim == 43


def test_05_masks_unchanged(normal_monitor):
    agent, *_ = normal_monitor
    assert agent.guardrails.mask_mismatches == 0
    assert "compute_action_mask" in _calls(AGENT_DIR / "tools.py")


def test_06_no_ppo_retraining(normal_monitor):
    forbidden = {"learn", "train", "train_seed", "build_model", "zero_grad", "backward", "step_optimizer"}
    for p in RUNTIME_MODULES:
        assert not (_calls(p) & forbidden), p.name
    pol = PPOPolicy(MODEL)
    pol.load()
    before = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in pol.model.policy.state_dict().values())).hexdigest()
    drive(NORMAL_DAY, MONITOR, until=60.0, policy=pol)
    after = hashlib.sha256(b"".join(v.cpu().numpy().tobytes() for v in pol.model.policy.state_dict().values())).hexdigest()
    assert before == after


# 7-12 ------------------------------------------------------------------------------------------------------
def test_07_normal_fast_loop_uses_ppo_bit_identical():
    a = AE.run_agent_day(NORMAL_DAY, MODEL, PPO_ALONE, RULE, BASE)
    m = AE.run_agent_day(NORMAL_DAY, MODEL, MONITOR, RULE, BASE)
    assert a["action_hash"] == m["action_hash"] and a["total_reward"] == m["total_reward"]
    assert m["agent"]["routed_by"]["RULE_PRUDENCE"] == m["agent"]["routed_by"]["RULE_TECH_FALLBACK"] == 0


def test_08_no_llm_in_fast_loop():
    agent, *_ = drive(NORMAL_DAY, MONITOR, until=120.0, explanation=Forbidden())
    assert agent.routed_by["PPO"] > 0


def test_09_no_manager_in_normal_fast_loop():
    agent, *_ = drive(NORMAL_DAY, MONITOR, until=120.0, manager=Forbidden())
    assert agent.routed_by["PPO"] > 0


def test_10_no_rule_comparison_in_normal_fast_loop():
    class Counting(RulePolicyAdapter):
        calls = 0

        def predict(self, *a):
            Counting.calls += 1
            return super().predict(*a)

    drive(NORMAL_DAY, MONITOR, until=120.0, fallback=Counting(RULE))
    assert Counting.calls == 0


def test_11_supervision_every_15_minutes(normal_monitor):
    agent, *_ = normal_monitor
    snaps = agent.memory.records("SupervisionSnapshot")
    assert snaps and all(s["cadence_lag"] >= 0 for s in snaps)
    assert all(abs((s["t"] - s["cadence_lag"]) / 15.0 - round((s["t"] - s["cadence_lag"]) / 15.0)) < 1e-9 for s in snaps)
    boundaries = [round((s["t"] - s["cadence_lag"]) / 15.0) for s in snaps]
    assert boundaries == sorted(set(boundaries)) and len(snaps) >= 18  # 300 min -> >= 18 cycles


def test_12_no_emergency_no_whatif(normal_monitor):
    agent, *_ = normal_monitor
    assert not agent.memory.records("EmergencyRecord")
    assert not [c for c in agent.tools.calls if c["tool"] == "run_twin_whatif"]
    assert {c["tool"] for c in agent.tools.calls} >= {"get_kpis", "forecast", "get_recent_context"}


# 13-20 -----------------------------------------------------------------------------------------------------
def test_13_emergency_triggers_whatif(ai_always):
    agent, *_ = ai_always
    em, wi = agent.memory.records("EmergencyRecord"), agent.memory.records("WhatIfRecord")
    assert em and len(em) == len(wi)
    assert len([c for c in agent.tools.calls if c["tool"] == "run_twin_whatif"]) <= len(wi)


def test_14_no_hidden_scenario_access(ai_always):
    forbidden = {"truth", "degradation_window", "shortage_window", "shock_window", "family", "_arrival_queue", "events_config"}
    for p in RUNTIME_MODULES:
        if p.name == "perception.py":
            continue
        assert not (_attrs(p) & forbidden), (p.name, _attrs(p) & forbidden)
    *_, events = ai_always
    assert events and all(set(e.data) <= SAFE_DATA_KEYS for e in events)
    assert not {e.event_type for e in events} & {"AI_FALSE_RESOLUTION", "AI_RESOLVED", "REWARD", "RECONTACT_SCHEDULED"}


def test_15_hypothetical_rng_independent():
    seeds = [whatif_seed(3080, c, k) for c in range(1, 60) for k in range(32)]
    assert min(seeds) >= WHATIF_SEED_BASE > 2**31 and len(set(seeds)) == len(seeds)
    assert not set(seeds) & (set(PHASE_F_DEV_SEEDS) | set(PHASE_F_TEST_SEEDS) | set(FINAL_V1_1_SEEDS))


def _snapshot(agent):
    t = agent.memory.last_percept.timestamp
    return agent.tools.observable_snapshot(t, agent.tools.get_kpis(t))


def test_16_true_future_never_copied(ai_always):
    agent, env, *_ = ai_always
    snap = _snapshot(agent)
    base = AE._base(BASE)
    params = world_parameters(snap, base, WhatIfPlan(5.0, 30.0))
    live_future = {round(c.arrival_time, 9) for c in env.twin.truth if c.arrival_time > snap.now}
    cfg = hypothetical_config(snap, base, params, MIXTURE_CELLS[0])
    for k in range(3):
        contacts, _, _ = build_world(snap, cfg, params, whatif_seed(3080, 99, k))
        assert not {round(c.arrival_time, 9) for c in contacts if c.arrival_time > snap.now} & live_future


def test_17_paired_hypothetical_futures(ai_always):
    agent, *_ = ai_always
    snap = _snapshot(agent)
    base = AE._base(BASE)
    params = world_parameters(snap, base, WhatIfPlan(5.0, 30.0))
    cfg = hypothetical_config(snap, base, params, MIXTURE_CELLS[1])
    s = whatif_seed(3080, 98, 0)
    w1, w2 = build_world(snap, cfg, params, s), build_world(snap, cfg, params, s)
    assert w1 == w2
    ppo = PPOPolicy(MODEL)
    ppo.load()
    a = simulate_branch(snap, cfg, *w1, s, [], ppo, StateBuilder())
    b = simulate_branch(snap, cfg, *w1, s, [(snap.now + 5, snap.now + 35, ppo)], ppo, StateBuilder())
    assert a["reward"] == b["reward"]  # same future + same policy => identical outcome (pairing is exact)


def test_18_leakage_invariance_gate():
    res = json.loads((OUT / "leakage_audit.json").read_text(encoding="utf-8"))
    assert res["passed"] and res["label"] == "LEAKAGE TEST PASSED" and all(res["checks"].values())


class Recorder:
    def __init__(self, inner, name):
        self.inner, self.name, self.times = inner, name, []

    def predict(self, state, obs, mask):
        self.times.append(self.t())
        return self.inner.predict(state, obs, mask)


def _branch_policies(ai_always, segments_fn):
    agent, *_ = ai_always
    snap = _snapshot(agent)
    base = AE._base(BASE)
    params = world_parameters(snap, base, WhatIfPlan(5.0, 30.0))
    cfg = hypothetical_config(snap, base, params, MIXTURE_CELLS[0])
    s = whatif_seed(3080, 97, 0)
    world = build_world(snap, cfg, params, s)
    ppo, rule = PPOPolicy(MODEL), RulePolicyAdapter(RULE)
    ppo.load()
    clock = {"t": 0.0}

    class Clocked:
        def __init__(self, inner):
            self.inner, self.times = inner, []

        def predict(self, state, obs, mask):
            self.times.append(obs.hour_sin)
            return self.inner.predict(state, obs, mask)

    p, r = Clocked(ppo), Clocked(rule)
    res = simulate_branch(snap, cfg, *world, s, segments_fn(snap, r), p, StateBuilder())
    return snap, res, p, r


def test_19_branch_a_is_ppo_continuously(ai_always):
    snap, res, p, r = _branch_policies(ai_always, lambda snap, r: [])
    assert res["decisions_by_candidate"] == 0 and not r.times and p.times


def test_20_branch_b_rule_for_d_then_ppo(ai_always):
    from callverse_workforce.agent import whatif as W

    agent, *_ = ai_always
    snap = _snapshot(agent)
    seen = []
    orig = W.simulate_branch

    def spy(snapshot, cfg, contacts, holds, forced, seed, segments, default_policy, sb):
        seen.append([(t0 - snapshot.now, t1 - snapshot.now) for t0, t1, _ in segments])
        return orig(snapshot, cfg, contacts, holds, forced, seed, segments, default_policy, sb)

    W.simulate_branch = spy
    try:
        ppo, rule = PPOPolicy(MODEL), RulePolicyAdapter(RULE)
        ppo.load()
        W.run_twin_whatif(snap, AE._base(BASE), ppo, rule, WhatIfPlan(5.0, 30.0), [whatif_seed(3080, 96, 0)])
    finally:
        W.simulate_branch = orig
    assert seen == [[], [(5.0, 35.0)]]  # A: PPO only; B: Rule on [approval, approval + D), PPO elsewhere


# 21-28 -----------------------------------------------------------------------------------------------------
def _req(gain=10.0, conf=True):
    return ManagerRequest("Q1", 100.0, "E1", "W1", RECOMMEND, gain, conf, True, True)


def test_21_manager_delay():
    m = SimulatedManager("primary", g_min=5.0, approval_delay=5.0)
    m.notify_manager(_req())
    assert m.poll(104.99) is None and m.poll(105.0).decision == APPROVE
    s = SimulatedManager("slow", g_min=5.0, approval_delay=5.0)
    s.notify_manager(_req())
    assert s.poll(129.9) is None and s.poll(130.0).response_t == 130.0


def test_22_manager_rule_deterministic():
    for prof, gain, conf, exp in (("primary", 10, True, APPROVE), ("primary", 4, True, REJECT), ("primary", 10, False, REJECT),
                                  ("conservative", 9, True, REJECT), ("conservative", 11, True, APPROVE)):
        outs = []
        for _ in range(3):
            m = SimulatedManager(prof, g_min=5.0)
            m.notify_manager(_req(gain, conf))
            outs.append(m.poll(1e9).decision)
        assert outs == [exp] * 3


def test_23_reject_ppo_continues():
    agent, _, outs, _ = drive(AI_DAY, ALWAYS, until=420.0, manager=StubManager(REJECT))
    assert agent.memory.records("ManagerRequest") and agent.routed_by["RULE_PRUDENCE"] == 0
    assert any(r["reason"] == "manager_rejected" and r["to"] == "NORMAL_PPO" for r in agent.memory.records("ModeTransition"))


def test_24_approve_temporary_prudence(ai_always):
    agent, *_ = ai_always
    assert any(r["to"] == "TEMPORARY_PRUDENCE" for r in agent.memory.records("ModeTransition"))
    assert agent.routed_by["RULE_PRUDENCE"] > 0


def test_25_26_d_respected_and_auto_return(ai_always):
    agent, _, outs, _ = ai_always
    periods = [(r["start"], r["scheduled_end"]) for r in agent.memory.records("PrudencePeriod") if r["phase"] == "start"]
    assert periods and all(abs((e - s) - 30.0) < 1e-9 for s, e in periods)
    for t, o in outs:
        inside = any(s <= t < e for s, e in periods)
        assert (o.policy_used == "RULE_PRUDENCE") == inside, (t, o.policy_used)
    ends = [r for r in agent.memory.records("ModeTransition") if r["reason"] == "prudence_expired_auto_return"]
    assert ends and all(r["to"] == "NORMAL_PPO" for r in ends)


class FlakyPPO(PPOPolicy):
    """Fails on calls in [start, stop) (fault injection)."""

    def __init__(self, start, stop):
        super().__init__(MODEL)
        self.n, self.start, self.stop = 0, start, stop

    def predict(self, state, obs, mask):
        self.n += 1
        if self.start <= self.n < self.stop:
            raise RuntimeError("injected inference failure")
        return super().predict(state, obs, mask)


def test_27_technical_fallback_separate():
    agent, *_ = drive(NORMAL_DAY, MONITOR, until=200.0, policy=FlakyPPO(50, 10**9))
    assert agent.mode is AgentMode.TECHNICAL_FALLBACK and agent.routed_by["RULE_TECH_FALLBACK"] > 0
    assert agent.routed_by["RULE_PRUDENCE"] == 0 and not agent.memory.records("ManagerRequest")
    assert not agent.memory.records("PrudencePeriod")
    assert [r["phase"] for r in agent.memory.records("TechnicalFallbackEvent")] == ["start"]


def test_28_technical_recovery_returns_to_ppo():
    agent, *_ = drive(NORMAL_DAY, MONITOR, until=200.0, policy=FlakyPPO(50, 52))
    reasons = [r["reason"] for r in agent.memory.records("ModeTransition")]
    assert "ppo_technical_failure" in reasons and "ppo_recovered" in reasons and agent.mode is AgentMode.NORMAL_PPO
    assert [r["phase"] for r in agent.memory.records("TechnicalFallbackEvent")] == ["start", "end"]


# 29-34 -----------------------------------------------------------------------------------------------------
def test_29_memory_logs_events(ai_always):
    agent, _, outs, _ = ai_always
    mem = agent.memory
    assert len(mem.records("DecisionRecord")) == len(outs)
    assert mem.records("AgentStarted")[0]["policy"]["model_sha256"] == MODELS["e1_s500"]["sha256"]
    for kind in ("SupervisionSnapshot", "EmergencyRecord", "WhatIfRecord", "RecommendationRecord", "ManagerRequest",
                 "ManagerResponse", "ModeTransition", "PrudencePeriod", "ExplanationRecord", "ToolCall"):
        assert mem.records(kind), kind
    w = mem.records("WhatIfRecord")[0]
    assert {"seeds", "branch_a", "branch_b", "estimated_gain", "diffs", "snapshot_sha256"} <= set(w)


def test_30_explanations_cannot_change_control():
    rnd = np.random.default_rng(0)
    gib = LLMExplainer(lambda ev: "".join(rnd.choice(list("APPROVE REJECT KEEP"), 40)))
    a, _, outs_a, _ = drive(AI_DAY, ALWAYS, until=360.0, manager=StubManager(APPROVE))
    b, _, outs_b, _ = drive(AI_DAY, ALWAYS, until=360.0, manager=StubManager(APPROVE), explanation=ExplanationService(gib))
    assert [o.action for _, o in outs_a] == [o.action for _, o in outs_b]
    strip = lambda ag: [(r["t"], r["from"], r["to"]) for r in ag.memory.records("ModeTransition")]
    assert strip(a) == strip(b)
    assert {r["renderer"] for r in b.memory.records("ExplanationRecord")} == {"llm"}


def test_31_runtime_without_llm(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    def broken(_):
        raise ImportError("no LLM")
    agent, *_ = drive(AI_DAY, ALWAYS, until=330.0, manager=StubManager(APPROVE), explanation=ExplanationService(LLMExplainer(broken)))
    recs = agent.memory.records("ExplanationRecord")
    assert recs and {r["renderer"] for r in recs} == {"template"} and all(r["text"] for r in recs)


def test_32_dev_test_separated():
    assert not set(PHASE_F_DEV_SEEDS) & set(PHASE_F_TEST_SEEDS)
    assert PHASE_F_DEV_SEEDS == tuple(range(3000, 3160)) and PHASE_F_TEST_SEEDS == tuple(range(3500, 3770))
    from callverse_workforce.phase_e.scenarios import DaySpec

    with pytest.raises(SeedPolicyError):
        EV.days_for("phase_f_dev", [DaySpec("NORMAL", MIXTURE_CELLS[0], 3500)])
    test_map = AE.test_days(range(3500, 3740), range(3740, 3770))
    assert len(test_map) == 270 and sum(d.family == "AI_DEGRADATION" for d in test_map) == 60
    assert all(sum(d.family == f for d in test_map) == 30 for f in {d.family for d in test_map} - {"AI_DEGRADATION"})


def test_33_seed_collisions_rejected(monkeypatch):
    import importlib.util

    spec = importlib.util.spec_from_file_location("rpf", PROJECT_ROOT / "scripts" / "run_phase_f.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "consumed_seed_sources", lambda root: {"debug_run": {3005, 7}})
    a = mod.collision_audit(set(PHASE_F_DEV_SEEDS), "x")
    assert a["n_collisions"] == 1 and a["collisions"] == {"debug_run": [3005]}
    audit = json.loads((OUT / "dev_seed_audit.json").read_text(encoding="utf-8"))
    assert audit["n_collisions"] == 0 and audit["test_range_precheck"]["n_collisions"] == 0


def test_34_v2_reserve_untouched():
    from callverse_workforce.phase_e.scenarios import DaySpec

    for purpose in ("phase_f_dev", "phase_f_test"):
        with pytest.raises(SeedPolicyError):
            EV.days_for(purpose, [DaySpec("NORMAL", MIXTURE_CELLS[0], 2500)])
    assert not (set(PHASE_F_DEV_SEEDS) | set(PHASE_F_TEST_SEEDS)) & set(V2_RESERVED_SEEDS)
    assert json.loads((OUT / "dev_seed_audit.json").read_text(encoding="utf-8"))["sources_checked"]["v2_reserved_closed"] == 760


# 35-40 -----------------------------------------------------------------------------------------------------
def test_35_routing_policy_interface():
    for p in (PPOPolicy(MODEL), RulePolicyAdapter(RULE)):
        for m in ("load", "reset", "predict", "get_policy_metadata"):
            assert callable(getattr(p, m))
    src = (AGENT_DIR / "workforce_manager.py").read_text(encoding="utf-8")
    assert "phase_e" not in src and "SMDPMaskablePPO" not in src and "PPORoutingPolicy" not in src


def test_36_decision_type_present(normal_monitor):
    agent, *_ = normal_monitor
    recs = agent.memory.records("DecisionRecord")
    assert recs and {r["decision_type"] for r in recs} == {"INITIAL"}


def test_37_action_registry():
    assert V1_1_REGISTRY.action_ids == tuple(a.value for a in ACTION_ORDER)
    assert [s.ppo_index for s in V1_1_REGISTRY.specs] == [a.index for a in ACTION_ORDER]
    got = []
    ex = RoutingExecutor(V1_1_REGISTRY, {s.handler: got.append for s in V1_1_REGISTRY.specs})
    for a in V1_1_REGISTRY.action_ids:
        ex.execute(a)
    assert got == list(V1_1_REGISTRY.action_ids)
    with pytest.raises(KeyError):
        ex.execute("HUMAN_HIGH")
    src = (AGENT_DIR / "executor.py").read_text(encoding="utf-8")
    assert "== \"AI\"" not in src and "RoutingAction.AI" not in src


def test_38_supervisor_action_count_agnostic():
    src = (AGENT_DIR / "supervisor.py").read_text(encoding="utf-8")
    assert not any(t in src for t in ("ACTION_ORDER", "RoutingAction", "n_actions", "43", "V1_1_REGISTRY"))

    @dataclasses.dataclass
    class W:
        status: str = "OK"
        k: int = 16
        estimated_gain: float = 12.0
        t80_lower: float = 1.0
        share_positive: float = 0.8
        whatif_id: str = "W"

    sup = SupervisorController(SupervisionConfig(g_min=5.0, k_futures=16))
    from callverse_workforce.agent.emergency import Emergency

    em = Emergency("E1", 0.0, "AI_FAILURE_RATE", 0.6, 0.45, (), 1)
    assert sup.recommend(em, W()).kind == RECOMMEND
    assert sup.recommend(em, W(estimated_gain=4.0)).kind == KEEP_PPO
    assert sup.recommend(None, W()).kind == KEEP_PPO
    assert sup.recommend(em, None).kind == KEEP_PPO


def test_39_mock_alternative_policy_loads():
    """A mock 3-action policy with its own registry, 10-D schema and mask provider runs the SAME agent shell."""
    reg = ActionRegistry("MOCK", (ActionSpec("AI", 0, (DecisionType.INITIAL,), "h_ai", "AI"),
                                  ActionSpec("HUMAN", 1, (DecisionType.INITIAL,), "h_human", "HUMAN"),
                                  ActionSpec("WAIT", 2, (DecisionType.INITIAL,), "h_wait", "WAIT")))

    class TenD:
        dim, version = 10, "mock"

        def encode(self, obs):
            return np.full(10, obs.ai_resolution_score, dtype=np.float32)

    contract = ObservationContract("MOCK", "MOCK_10D", 10, DecisionType.INITIAL, "mock")

    class MockPolicy:
        def load(self):
            pass

        def reset(self):
            pass

        def predict(self, state, obs, mask):
            assert state.vector.shape == (10,)
            return "HUMAN"

        def get_policy_metadata(self):
            from callverse_workforce.agent.routing_policy import PolicyMetadata

            return PolicyMetadata("mock", "MOCK", None, "MOCK_10D", (DecisionType.INITIAL,), "MOCK", "mock")

    class MockMasks:
        def get(self, version, dt, obs):
            assert version == "MOCK"
            return pct.action_mask

    live = EV._config(BASE, NORMAL_DAY)
    from callverse_workforce.agent.perception import public_context_from
    from callverse_workforce.twin.environment import RoutingEnvironment

    env = RoutingEnvironment(live)
    dp = env.reset(NORMAL_DAY.seed)
    per = SimulationPerception(env, public_context_from(live, AE._base(BASE)))
    agent = WorkforceManagerAgent(MockPolicy(), RulePolicyAdapter(RULE), MONITOR, per.public_context(), AE._base(BASE),
                                  SimulatedManager("primary", 5.0), registry=reg, state_builder=StateBuilder(contract, TenD()),
                                  mask_provider=MockMasks())
    for _ in range(20):
        pct = per.percept(dp)
        o = agent.route(pct)
        assert o.action == "HUMAN" and o.policy_used == "PPO"
        dp = env.step(__import__("callverse_workforce.domain.enums", fromlist=["RoutingAction"]).RoutingAction(o.action)).next_decision
    assert agent.memory.records("AgentStarted")[0]["policy"]["policy_version"] == "MOCK"


def test_40_no_v2_behaviour_enabled():
    from callverse_workforce.agent.types import ENABLED_DECISION_TYPES

    assert ENABLED_DECISION_TYPES == (DecisionType.INITIAL,)
    assert not {"HUMAN_NORMAL", "HUMAN_HIGH"} & set(V1_1_REGISTRY.action_ids)
    assert not (PROJECT_ROOT / "src" / "callverse_workforce" / "v2").exists()

    class P:
        observation = None

    pct = Percept("D", "C", 0.0, DecisionType.POST_AI_FAILURE, None, None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        StateBuilder().build(pct)


# extra structural checks ----------------------------------------------------------------------------------
def test_state_machine_transitions_complete():
    assert ALLOWED[AgentMode.NORMAL_PPO] == {AgentMode.EMERGENCY_ANALYSIS, AgentMode.TECHNICAL_FALLBACK}
    assert AgentMode.TEMPORARY_PRUDENCE not in ALLOWED[AgentMode.NORMAL_PPO]  # no prudence without approval
    assert ALLOWED[AgentMode.TECHNICAL_FALLBACK] == {AgentMode.NORMAL_PPO}


def test_frozen_v1_1_artifacts_unchanged():
    import importlib.util

    pre = OUT / "frozen_state_pre_dev.json"
    if not pre.exists():
        pytest.skip("DEV not started")
    spec = importlib.util.spec_from_file_location("rpf2", PROJECT_ROOT / "scripts" / "run_phase_f.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.frozen_state() == json.loads(pre.read_text(encoding="utf-8"))
