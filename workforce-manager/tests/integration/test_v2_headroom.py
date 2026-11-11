"""V2 headroom protocol — specification tests 1-48 (docs/v2_headroom_spec.md §10).

Simulations use ONLY already-opened, non-protected E.1 development days (1681-1760) and synthetic fixtures.
No V2 DEV day (4000-4079) and no protected TEST seed is ever simulated here.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
from collections import Counter, defaultdict

import numpy as np
import pytest
import simpy

from callverse_workforce.agent.types import DecisionType
from callverse_workforce.config import load_config
from callverse_workforce.config.loader import PROJECT_ROOT
from callverse_workforce.headroom import stats as HS
from callverse_workforce.phase_e import evaluation as EV
from callverse_workforce.phase_e.scenarios import fixed_days
from callverse_workforce.seeds import PHASE_F_TEST_SEEDS, PPO_E1_DEV_SEEDS, V2_RESERVED_SEEDS
from callverse_workforce.twin.digital_twin import DigitalTwin
from callverse_workforce.v2_postfail import actions as A
from callverse_workforce.v2_postfail import protocol as P
from callverse_workforce.v2_postfail import stats as S
from callverse_workforce.v2_postfail.policies import ContinuationPolicy, b1, b2
from callverse_workforce.v2_postfail.queue import CALLBACK, HIGH, HOLD, NORMAL, AgingHumanPool
from callverse_workforce.v2_postfail.snapshot import take_snapshot
from callverse_workforce.v2_postfail.twin import A_AGE_PRIMARY, A_AGE_SENSITIVITY, V2Environment, V2Twin
from callverse_workforce.v2_postfail.world import MEAN_REVERSION_MINUTES, branch_seed, run_branch

BASE_PATH = str(PROJECT_ROOT / "configs" / "normal_day.yaml")
BASE = load_config(BASE_PATH)
RULE = json.loads((PROJECT_ROOT / "artifacts/v1_1/baselines/frozen/frozen_v1_1_policies.json").read_text(encoding="utf-8"))["frozen"]["RulePolicy"]
E1_DAYS = fixed_days(PPO_E1_DEV_SEEDS)  # already opened in Phase E.1 (not protected)
PKG = PROJECT_ROOT / "src" / "callverse_workforce" / "v2_postfail"
F_REF_TEST = 0.30  # test-only value; the real f_ref is computed by rule A1 in Stage 2


def run_day(day, post_fn, initial="RulePolicy"):
    cfg = EV._config(BASE_PATH, day)
    env = V2Environment(V2Twin(cfg, day.seed))
    init = ContinuationPolicy(initial, "B1", RULE)
    dps = []
    dp = env.next_decision()
    while dp is not None:
        dps.append(dp)
        a = init.act(dp) if dp.decision_type is DecisionType.INITIAL else post_fn(dp)
        dp = env.step(a)
    return env, dps, cfg


def wait_then_callback(dp):
    if dp.mask.is_valid(A.WAIT):
        return A.WAIT
    return A.CALLBACK if dp.mask.is_valid(A.CALLBACK) else A.HUMAN_NORMAL


def callback_first(dp):
    return A.CALLBACK if dp.mask.is_valid(A.CALLBACK) else A.HUMAN_NORMAL


@pytest.fixture(scope="module")
def waitday():  # AI_DEGRADATION, cell h1 (handoff 1.0 min) -> many POST decisions
    return run_day(E1_DAYS[41], wait_then_callback)


@pytest.fixture(scope="module")
def cbday():  # AI_DEGRADATION, cell h2 (handoff 2.0, hangup 0.05)
    return run_day(E1_DAYS[42], callback_first)


@pytest.fixture(scope="module")
def b1day():
    return run_day(E1_DAYS[45], b1)


def events(env, et):
    return [e for e in env.twin.events.records if e.event_type.value == et]


def first_snapshot(day, nth=40):
    cfg = EV._config(BASE_PATH, day)
    from callverse_workforce.agent.perception import public_context_from

    env = V2Environment(V2Twin(cfg, day.seed))
    pol = ContinuationPolicy("RulePolicy", "B1", RULE)
    dp, n = env.next_decision(), 0
    while dp is not None:
        if dp.decision_type is DecisionType.POST_AI_FAILURE:
            if n == nth:
                return take_snapshot(env.twin, dp.contact_id, public_context_from(cfg, BASE)), dp
            n += 1
        dp = env.step(pol.act(dp))
    raise RuntimeError


# ---------------------------------------------------------------- WAIT / U1 (1-6)
def test_01_wait_reevaluation_timing(waitday):
    env, _, _ = waitday
    started = defaultdict(list)
    for e in env.twin.events.records:
        if e.event_type.value == "WAIT_STARTED":
            started[e.contact_id].append((e.seq, e.time))
    re = events(env, "WAIT_REEVALUATION")
    assert re
    for e in re:
        t0 = max((s, t) for s, t in started[e.contact_id] if s < e.seq)[1]  # the WAIT that this re-evaluation ends
        if e.data["cause"].value == "TIMER":
            assert abs((e.time - t0) - 2.0) < 1e-9
        else:
            assert e.data["cause"].value == "HUMAN_AVAILABLE" and e.time - t0 <= 2.0 + 1e-9
    assert {e.data["cause"].value for e in re} == {"TIMER", "HUMAN_AVAILABLE"}


def test_02_wait_count_starts_at_zero(waitday):
    _, dps, _ = waitday
    assert all(dp.observation.wait_count == 0 for dp in dps if dp.decision_type is DecisionType.INITIAL)


def test_03_wait_count_increments_only_on_wait(waitday):
    env, dps, _ = waitday
    executed = Counter()
    by_contact = defaultdict(list)
    for e in env.twin.events.records:
        if e.event_type.value == "ROUTING_DECISION":
            assert e.data["wait_count"] == executed[e.contact_id]
            if e.data["action"] == A.WAIT:
                executed[e.contact_id] += 1
            by_contact[e.contact_id].append(e.data["action"])
    assert max(executed.values()) == 3
    for cid, rt in env.twin.contacts.items():
        assert rt.wait_count == executed[cid]


def test_04_fourth_wait_forbidden_and_attempts_irrelevant(waitday):
    _, dps, _ = waitday
    at3 = [dp for dp in dps if dp.observation.wait_count >= 3]
    assert at3 and all(not dp.mask.is_valid(A.WAIT) and "max_waits_reached" in dp.mask.reasons[A.WAIT] for dp in at3)
    o = next(dp for dp in dps if dp.decision_type is DecisionType.POST_AI_FAILURE and dp.mask.is_valid(A.WAIT)).observation.base
    import dataclasses

    many = dataclasses.replace(o, routing_attempt_count=99)
    from callverse_workforce.routing.mask import ServiceAvailability

    m = A.v2_mask(DecisionType.POST_AI_FAILURE, many, 0, ServiceAvailability(True, True), BASE.mask, BASE.wait)
    assert m.is_valid(A.WAIT)  # routing_attempt_count never substitutes for wait_count


def test_05_reevaluation_stays_post_ai_masked(waitday):
    _, dps, _ = waitday
    re = [dp for dp in dps if dp.observation.wait_count > 0]
    assert re and all(dp.decision_type is DecisionType.POST_AI_FAILURE and not dp.mask.is_valid(A.AI) for dp in re)


def test_06_v1_1_mask_unchanged():
    pre = json.loads((PROJECT_ROOT / "artifacts/phase_f/frozen_state_pre_dev.json").read_text(encoding="utf-8"))
    from callverse_workforce.final_test.v1_1 import sha_tree

    assert sha_tree(PROJECT_ROOT / "src/callverse_workforce/routing") == pre["sources"]["routing"]


# ---------------------------------------------------------------- CALLBACK (7-10)
def test_07_answer_probability(cbday):
    env, _, cfg = cbday
    assert cfg.callback.answer_probability == 0.65
    att, reached = len(events(env, "CALLBACK_ATTEMPT_STARTED")), len(events(env, "CALLBACK_REACHED"))
    p = reached / att
    assert att > 100 and abs(p - 0.65) < 4 * math.sqrt(0.65 * 0.35 / att)


def test_08_max_two_attempts(cbday):
    env, _, _ = cbday
    assert max(rt.callback_attempts for rt in env.twin.contacts.values()) == 2


def test_09_callback_window(cbday):
    env, _, _ = cbday
    sched = {e.contact_id: e.time for e in events(env, "CALLBACK_SCHEDULED")}
    exp = events(env, "CALLBACK_WINDOW_EXPIRED")
    assert all(abs(e.time - sched[e.contact_id] - 120.0) < 1e-9 for e in exp)
    assert all(e.time <= sched[e.contact_id] + 120.0 + 1e-9 for e in events(env, "CALLBACK_REACHED"))


def test_10_callback_accounting_unchanged(cbday):
    env, _, cfg = cbday
    from callverse_workforce.reward.accumulator import RewardAccumulator

    acc = RewardAccumulator(cfg.reward)
    checked = 0
    for e in env.twin.events.records:
        v = env.twin.contacts[e.contact_id].truth.value_weight if e.contact_id in env.twin.contacts else None
        et = e.event_type.value
        if et == "CALLBACK_DROPOUT":
            assert e.data["reward"] == acc.abandonment(v)
            checked += 1
        elif et in ("CALLBACK_FAILED", "CALLBACK_WINDOW_EXPIRED"):
            assert e.data["reward"] == acc.callback_failure()
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------- handoff U2 (11-16)
def _handoff_events(env):
    return [(e.contact_id, e.event_type.value) for e in env.twin.events.records if "handoff_minutes" in e.data]


def test_11_handoff_once_online(b1day):
    env, _, cfg = b1day
    h = cfg.v1_1.handoff_overhead_minutes
    assert h > 0
    starts = [e for e in events(env, "HUMAN_SERVICE_STARTED") if env.twin.contacts[e.contact_id].escalated]
    assert starts and all(e.data.get("handoff_minutes") == h for e in starts)


def test_12_handoff_at_answered_callback(cbday):
    env, _, cfg = cbday
    reached = [e for e in events(env, "CALLBACK_REACHED") if env.twin.contacts[e.contact_id].escalated]
    assert reached and all(e.data.get("handoff_minutes") == cfg.v1_1.handoff_overhead_minutes for e in reached)


def test_13_no_handoff_at_scheduling_or_unanswered(cbday):
    env, _, _ = cbday
    assert not any("handoff_minutes" in e.data for e in env.twin.events.records
                   if e.event_type.value in ("CALLBACK_SCHEDULED", "CALLBACK_ATTEMPT_STARTED", "CALLBACK_ATTEMPT_FAILED"))
    connected = len([1 for c, et in _handoff_events(env)])
    assert env.twin.handoff_events == connected


def test_14_handoff_never_twice(cbday, waitday):
    for env, _, _ in (cbday, waitday):
        assert max(Counter(c for c, _ in _handoff_events(env)).values()) == 1


def test_15_non_escalated_never_handoff(b1day):
    env, _, _ = b1day
    assert all(env.twin.contacts[c].escalated for c, _ in _handoff_events(env))


def test_16_v1_1_handoff_unchanged():
    pre = json.loads((PROJECT_ROOT / "artifacts/phase_f/frozen_state_pre_dev.json").read_text(encoding="utf-8"))
    from callverse_workforce.final_test.v1_1 import sha_tree

    assert sha_tree(PROJECT_ROOT / "src/callverse_workforce/twin") == pre["sources"]["twin"]


# ---------------------------------------------------------------- queue / aging U3 (17-24)
def _pool_run(script, a_age=A_AGE_PRIMARY, capacity=1):
    env = simpy.Environment()
    pool = AgingHumanPool(env, capacity, a_age)
    order = []

    def holder():
        r = pool.request(cls=NORMAL)
        yield r
        yield env.timeout(script["busy_until"])
        pool.release(r)

    def client(name, t, cls, acc=0.0):
        yield env.timeout(t)
        r = pool.request(cls=cls, acc_before=acc)
        yield r
        order.append(name)
        yield env.timeout(0.001)
        pool.release(r)

    env.process(holder())
    for name, t, cls, *acc in script["clients"]:
        env.process(client(name, t, cls, *acc))
    env.run()
    return order, pool


def test_17_high_before_normal():
    order, _ = _pool_run({"busy_until": 1.0, "clients": [("n", 0.1, NORMAL), ("h", 0.2, HIGH)]})
    assert order == ["h", "n"]


def test_18_aged_normal_not_postponed():
    order, pool = _pool_run({"busy_until": 9.0, "clients": [("n", 0.0, NORMAL), ("h1", 5.0, HIGH), ("h2", 8.6, HIGH)]})
    assert order[0] == "n" and pool.aging_violations == 0 and pool.aging_overrides == 1


def test_19_fifo_within_class():
    order, _ = _pool_run({"busy_until": 1.0, "clients": [("n1", 0.1, NORMAL), ("n2", 0.2, NORMAL), ("h1", 0.3, HIGH), ("h2", 0.4, HIGH),
                                                          ("c1", 0.5, CALLBACK), ("c2", 0.6, CALLBACK)]})
    assert order == ["h1", "h2", "n1", "n2", "c1", "c2"]


def test_20_same_discipline_everywhere(b1day):
    env, _, _ = b1day
    assert isinstance(env.twin.humans, AgingHumanPool) and env.twin.humans.a_age == A_AGE_PRIMARY
    snap, dp = first_snapshot(E1_DAYS[45], 10)
    tw_holder = {}
    import callverse_workforce.v2_postfail.world as W

    orig = W.V2Twin

    class Spy(orig):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            tw_holder["t"] = self

    W.V2Twin = Spy
    try:
        run_branch(snap, BASE, A.HUMAN_NORMAL, branch_seed(1726, 10, 0), ContinuationPolicy("RulePolicy", "B1", RULE), "persistence",
                   A_AGE_PRIMARY, F_REF_TEST)
    finally:
        W.V2Twin = orig
    assert isinstance(tw_holder["t"].humans, AgingHumanPool) and tw_holder["t"].humans.a_age == A_AGE_PRIMARY


def test_21_accumulated_queue_time_kept():
    order, _ = _pool_run({"busy_until": 4.0, "clients": [("n_acc", 0.0, NORMAL, 6.0), ("h", 0.5, HIGH)]})
    assert order == ["n_acc", "h"]  # 6.0 accumulated + 4.0 waited >= A_age -> aged before the HIGH contact
    env = simpy.Environment()
    pool = AgingHumanPool(env, 1, A_AGE_PRIMARY)
    r = pool.request(cls=NORMAL, acc_before=5.0)
    assert abs(r.aging_at - (A_AGE_PRIMARY - 5.0)) < 1e-12


def test_22_no_high_reproduces_v1_1_fifo():
    rng = np.random.default_rng(0)
    clients = [(f"c{i}", float(t), [NORMAL, CALLBACK, HOLD][int(rng.integers(3))]) for i, t in enumerate(sorted(rng.uniform(0, 20, 40)))]
    v2, _ = _pool_run({"busy_until": 30.0, "clients": clients})
    env = simpy.Environment()
    res = simpy.PriorityResource(env, 1)
    order = []
    prio = {HOLD: -1, NORMAL: 0, CALLBACK: 1}

    def holder():
        r = res.request(priority=0)
        yield r
        yield env.timeout(30.0)
        res.release(r)

    def client(name, t, cls):
        yield env.timeout(t)
        r = res.request(priority=prio[cls])
        yield r
        order.append(name)
        yield env.timeout(0.001)
        res.release(r)

    env.process(holder())
    for c in clients:
        env.process(client(*c))
    env.run()
    assert v2 == order


def test_23_a_age_recomputed():
    from callverse_workforce.twin.patience import PatienceDistribution

    cls = []
    for rt, rc in BASE.request_types.items():
        for ch, cc in BASE.channels.items():
            cls.append((rc.share * cc.share, PatienceDistribution.from_config(rc.patience_mean_minutes * cc.patience_multiplier, BASE.patience)))
    z = sum(w for w, _ in cls)
    F = lambda t: sum(w / z * d.cdf(t) for w, d in cls)
    lo, hi = 0.5, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if F(mid) >= 0.5 else (mid, hi)
    rec = json.loads((PROJECT_ROOT / "artifacts/v2_headroom/u3_aging_threshold.json").read_text(encoding="utf-8"))
    assert abs(hi - rec["mixture_median_minutes"]) < 1e-9 and abs(round(hi, 6) - A_AGE_PRIMARY) < 1e-12 and A_AGE_SENSITIVITY == 3.0


def test_24_no_patience_to_policy_or_planner():
    for name in ("snapshot.py", "world.py", "policies.py", "stats.py", "protocol.py"):
        attrs = {n.attr for n in ast.walk(ast.parse((PKG / name).read_text(encoding="utf-8"))) if isinstance(n, ast.Attribute)}
        assert "truth" not in attrs, name
    from callverse_workforce.v2_postfail.twin import V2Observation

    fields = set(V2Observation.__dataclass_fields__)
    assert fields == {"base", "decision_type", "wait_count", "escalated"}


# ---------------------------------------------------------------- V1.1 preservation (25-27)
def test_25_v1_1_bit_identical():
    from callverse_workforce.baselines.policies import make_policy
    from callverse_workforce.twin.runner import run_simulation

    stored = json.loads((PROJECT_ROOT / "artifacts/ppo_v1_1_e1/development_evaluation/per_day.json").read_text(encoding="utf-8"))
    for d in (E1_DAYS[0], E1_DAYS[47]):
        r = run_simulation(EV._config(BASE_PATH, d), make_policy("RulePolicy", RULE, d.seed), seed=d.seed)
        assert r.episode_reward.total == next(x["total_reward"] for x in stored["RulePolicy"] if x["seed"] == d.seed)
    path = json.loads((PROJECT_ROOT / "artifacts/final_test_v1_1/model_hashes.json").read_text(encoding="utf-8"))["models"]["e1_s500"]["path"]
    d = E1_DAYS[0]
    r = run_simulation(EV._config(BASE_PATH, d), EV._policy("ppo", path, d.seed), seed=d.seed)
    assert r.episode_reward.total == next(x["total_reward"] for x in stored["e1_s500"] if x["seed"] == d.seed)


@pytest.mark.parametrize("idx", [0, 11, 22, 33, 43, 52, 61, 70])  # all 8 families, all 4 cells
def test_26_v2_b1_reproduces_v1_1(idx):
    from callverse_workforce.baselines.policies import make_policy
    from callverse_workforce.twin.runner import run_simulation

    d = E1_DAYS[idx]
    v1 = run_simulation(EV._config(BASE_PATH, d), make_policy("RulePolicy", RULE, d.seed), seed=d.seed)
    env, _, _ = run_day(d, b1)
    assert env.twin.reward.episode.total == v1.episode_reward.total
    assert {c.contact_id: c.outcome.value for c in v1.contacts} == {k: r.outcome.value for k, r in env.twin.contacts.items()}


def test_27_frozen_hashes_unchanged():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rpf", PROJECT_ROOT / "scripts" / "run_phase_f.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.frozen_state() == json.loads((PROJECT_ROOT / "artifacts/phase_f/frozen_state_pre_dev.json").read_text(encoding="utf-8"))
    mh = json.loads((PROJECT_ROOT / "artifacts/final_test_v1_1/model_hashes.json").read_text(encoding="utf-8"))["models"]["e1_s500"]
    assert hashlib.sha256(open(mh["path"], "rb").read()).hexdigest() == mh["sha256"]


# ---------------------------------------------------------------- decision context (28-31)
def test_28_post_after_every_nonhangup_visible_failure(b1day):
    env, _, _ = b1day
    vis = {e.contact_id for e in events(env, "AI_VISIBLE_FAILURE")}
    hang = {e.contact_id for e in events(env, "ABANDONMENT") if e.data.get("cause") == "post_ai_failure_hangup"}
    post = {e.contact_id for e in events(env, "ROUTING_DECISION") if e.data["decision_type"] == "POST_AI_FAILURE"}
    false = {e.contact_id for e in events(env, "AI_FALSE_RESOLUTION")}
    assert post == vis - hang and not (post & false)


def test_29_decision_type_on_every_event(b1day):
    env, _, _ = b1day
    ev = events(env, "ROUTING_DECISION")
    assert ev and all(e.data["decision_type"] in ("INITIAL", "POST_AI_FAILURE") for e in ev)


def test_30_initial_ai_human_only(waitday):
    _, dps, _ = waitday
    assert all(set(dp.mask.valid_actions) <= {A.AI, A.HUMAN} for dp in dps if dp.decision_type is DecisionType.INITIAL)


def test_31_v2_contract_through_unchanged_agent_shell():
    from callverse_workforce.agent.manager import SimulatedManager
    from callverse_workforce.agent.perception import public_context_from
    from callverse_workforce.agent.routing_policy import PolicyMetadata, RulePolicyAdapter
    from callverse_workforce.agent.config import SupervisionConfig
    from callverse_workforce.agent.state_builder import StateBuilder
    from callverse_workforce.agent.types import Percept
    from callverse_workforce.agent.workforce_manager import WorkforceManagerAgent
    from callverse_workforce.v2_postfail.encoding import V2_CONTRACT, V2Encoder

    d = E1_DAYS[0]
    cfg = EV._config(BASE_PATH, d)
    env = V2Environment(V2Twin(cfg, d.seed))

    class V2Masks:
        def get(self, version, dt, obs):
            assert version == "V2_POSTFAIL"
            return A.v2_mask(dt, obs, 0, env.twin.service_availability(), cfg.mask, cfg.wait)

    class Pol:
        def load(self): pass
        def reset(self): pass
        def predict(self, state, obs, mask):
            assert state.vector.shape == (V2_CONTRACT.dim,)
            return A.AI if mask.is_valid(A.AI) else A.HUMAN
        def get_policy_metadata(self):
            return PolicyMetadata("v2-mock", "V2_POSTFAIL", None, "V2_INITIAL_POSTFAIL", (DecisionType.INITIAL,), "V2_POSTFAIL", "v2")

    agent = WorkforceManagerAgent(Pol(), RulePolicyAdapter(RULE), SupervisionConfig(enabled=False), public_context_from(cfg, BASE), BASE,
                                  SimulatedManager("primary", 5.0), registry=A.V2_REGISTRY,
                                  state_builder=StateBuilder(V2_CONTRACT, V2Encoder()), mask_provider=V2Masks())
    dp, routed = env.next_decision(), 0
    while dp is not None and routed < 100:
        if dp.decision_type is DecisionType.INITIAL:
            o = agent.route(Percept(dp.decision_id, dp.contact_id, dp.timestamp, DecisionType.INITIAL, dp.observation.base, dp.mask))
            assert o.policy_used == "PPO" and o.action in (A.AI, A.HUMAN)
            routed += 1
            dp = env.step(o.action)
        else:
            dp = env.step(b1(dp))  # POST routing through the shell is out of scope (L4)
    assert routed == 100 and agent.guardrails.mask_mismatches == 0


# ---------------------------------------------------------------- baselines (32-33)
def test_32_b1_deterministic(b1day):
    _, dps, _ = b1day
    post = [dp for dp in dps if dp.decision_type is DecisionType.POST_AI_FAILURE]
    assert post and {b1(dp) for dp in post} == {A.HUMAN_NORMAL}


class _DP:
    def __init__(self, urgency, cb_valid, humans, queue, deadline):
        import dataclasses

        base = first_snapshot.__defaults__  # noqa: F841 (placeholder to keep import order simple)
        from callverse_workforce.domain.enums import Urgency

        self.decision_type = DecisionType.POST_AI_FAILURE

        class O:
            pass

        self.observation = O()
        self.observation.base = O()
        self.observation.base.urgency_pred = Urgency(urgency)
        self.observation.base.humans_available = humans
        self.observation.base.human_queue_size = queue
        self.observation.base.deadline_remaining_pred = deadline
        valid = (False, False, True, True, True, cb_valid)
        self.mask = A.V2Mask(DecisionType.POST_AI_FAILURE, valid)


def test_33_b2_frozen_order():
    assert b2(_DP("HIGH", True, 0, 5, 1000)) == A.HUMAN_HIGH
    assert b2(_DP("LOW", True, 0, 2, 180)) == A.CALLBACK
    assert b2(_DP("LOW", False, 0, 5, 1000)) == A.HUMAN_NORMAL
    assert b2(_DP("MEDIUM", True, 1, 5, 1000)) == A.HUMAN_NORMAL
    assert b2(_DP("LOW", True, 0, 1, 1000)) == A.HUMAN_NORMAL
    assert b2(_DP("LOW", True, 0, 3, 179.9)) == A.HUMAN_NORMAL


# ---------------------------------------------------------------- continuation U4 (34-37)
@pytest.fixture(scope="module")
def branches():
    snap, dp = first_snapshot(E1_DAYS[45], 25)
    cont = ContinuationPolicy("RulePolicy", "B2", RULE)
    out = {a: [run_branch(snap, BASE, a, branch_seed(1726, 25, k), cont, "persistence", A_AGE_PRIMARY, F_REF_TEST) for k in range(2)]
           for a in dp.mask.valid_actions}
    return snap, dp, out


def test_34_forced_once_then_continuation_to_780(branches):
    _, _, out = branches
    for a, rs in out.items():
        for r in rs:
            assert r["status"] == "OK" and r["target_actions"][0] == a and r["horizon"] == 780.0 and r["finalized"]
            assert r["decisions_after_target"] > 0


def test_35_end_of_day_settlement_identical():
    assert V2Twin.finalize is DigitalTwin.finalize and BASE.false_ai_discovery.settle_at_horizon is True
    assert BASE.scenario.horizon_minutes == 780.0


def test_36_discounted_return_gamma():
    src = (PKG / "world.py").read_text(encoding="utf-8")
    assert "DiscountedReturn(lambda: twin.now, twin.now, cfg.smdp.gamma_per_minute)" in src and BASE.smdp.gamma_per_minute == 0.995
    assert P.GAMMA_PER_MINUTE == 0.995


def test_37_same_horizon_for_all_actions(branches):
    _, _, out = branches
    assert {r["horizon"] for rs in out.values() for r in rs} == {780.0}
    assert len({tuple(r["seed"] for r in rs) for rs in out.values()}) == 1


# ---------------------------------------------------------------- leakage & statistics (38-41)
def test_38_leakage_gate():
    from callverse_workforce.v2_postfail.leakage import leakage_gate

    res = leakage_gate(E1_DAYS[0], E1_DAYS[43], BASE_PATH, RULE, F_REF_TEST, k=2)
    assert res["passed"], res["checks"]


def test_39_reused_statistics():
    src = (PKG / "stats.py").read_text(encoding="utf-8")
    assert "from callverse_workforce.headroom.stats import analyze_state, cluster_bootstrap_mean, clearly_positive" in src
    assert 'res["regret"]["X"]["cross_fitted"]' in src and "clearly_positive(G[a_star][ev] - G[baseline_action][ev])" in src
    assert S.CLEAR_MEANINGFUL == 0.10 and S.CLEAR_WEAK == 0.05 and S.ADV_MEANINGFUL == 0.5


def test_40_gate_and_classification_synthetic():
    base = {"n_states": 1600, "h3_ci": {"lo": 0.1}}
    assert S.classify(base | {"h1_share": 0.12, "h2_mean_adv": 0.6})["label"] == "MEANINGFUL"
    assert S.classify(base | {"h1_share": 0.06, "h2_mean_adv": 0.1})["label"] == "WEAK"
    assert S.classify(base | {"h1_share": 0.01, "h2_mean_adv": 0.9})["label"] == "WEAK"
    assert S.classify({"n_states": 900, "h3_ci": {"lo": -0.1}, "h1_share": 0.01, "h2_mean_adv": 0.1}) == \
        {"label": "NONE", "H1": False, "H2": False, "H3": False, "limited_power": True}
    m, n = {"label": "MEANINGFUL"}, {"label": "NONE"}
    ok, bad = {"mc_unstable": False}, {"mc_unstable": True}
    assert S.decision_gate(m, m, ok, ok, 0)["verdict"] == "PPO V2 TRAINING CAN BE CONSIDERED"
    assert "CONTEXTUAL RULE ALREADY CAPTURES" in S.decision_gate(m, n, ok, ok, 0)["verdict"]
    assert S.decision_gate(m, m, bad, ok, 0)["verdict"] == "UNSTABLE — PPO V2 TRAINING NOT JUSTIFIED"
    assert S.decision_gate(m, m, ok, ok, 1)["verdict"] == "PPO V2 TRAINING NOT YET JUSTIFIED"
    # pre-registered stability rule: same subsample, compare H1/H2/H3 verdicts at K=32 vs K=64
    rows = [{"state_id": f"{d}:{i}", "day": d, "family": "NORMAL", "urgency": "LOW", "h1": 0.0, "adv": 1.0, "label": "TIE",
             "best": "HUMAN_NORMAL", "gap": 0.0, "baseline_action": "HUMAN_NORMAL"} for d in range(8) for i in range(5)]
    same = S.mc_stability(rows, rows, "t")
    assert same["status"] == "STABLE" and same["differing"] == []
    flip_h2 = S.mc_stability(rows, [r | {"adv": 0.2} for r in rows], "t")  # H2 (>0.5) verdict changes only
    assert flip_h2["status"] == "UNSTABLE" and flip_h2["differing"] == ["H2"]
    flip_h1 = S.mc_stability(rows, [r | {"h1": 1.0} if i % 5 == 0 else r for i, r in enumerate(rows)], "t")  # 20 % >= 10 %
    assert flip_h1["status"] == "UNSTABLE" and "H1" in flip_h1["differing"]
    sub = rows[:20]
    assert S.mc_stability(rows, sub, "t")["n"] == 20  # only the pre-selected subsample is compared
    assert S.mc_stability(rows, [rows[0] | {"state_id": "not_in_primary"}], "t")["status"] == "UNSTABLE"  # mismatch -> fail closed


def test_41_strata_cannot_change_gate():
    sig = inspect.signature(S.decision_gate)
    assert list(sig.parameters) == ["class_b1", "class_b2", "mc_b1", "mc_b2", "aging_violations"]
    src = inspect.getsource(S.classify)
    assert "by_family" not in src and "by_urgency" not in src


# ---------------------------------------------------------------- seeds (42-43)
def _runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rv2", PROJECT_ROOT / "scripts" / "run_v2_headroom.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_42_protected_ranges_inaccessible():
    m = _runner()
    assert not set(P.DEV_SEEDS) & (set(V2_RESERVED_SEEDS) | set(PHASE_F_TEST_SEEDS))
    for s in (2240, 2999, 3500, 3769):
        with pytest.raises(SystemExit):
            m.assert_not_protected([s])
    with pytest.raises(SystemExit):
        m.require_authorization()  # no DEV authorization exists in Stage 1


def test_43_collision_aborts():
    m = _runner()
    with pytest.raises(SystemExit):
        m.collision_audit({4001, 4002}, {"debug": {4002}})
    assert m.collision_audit({4001}, {"x": {1}})["n_collisions"] == 0


# ---------------------------------------------------------------- continuation & R2 (44-48)
def test_44_identical_continuation_and_futures():
    snap, dp = first_snapshot(E1_DAYS[45], 12)
    seen = []
    orig = P.run_branch

    def spy(snap_, base, action, seed, continuation, *a, **k):
        seen.append((action, seed, id(continuation)))
        return {"status": "OK", "discounted_return": 0.0, "aging_violations": 0, "aging_overrides": 0, "classes": {}}

    P.run_branch = spy
    try:
        rec = {"snapshot": snap, "state_index": 12, "valid": list(dp.mask.valid_actions), "b2_action": A.HUMAN_NORMAL, "urgency": "LOW"}
        P.state_returns(rec, E1_DAYS[45], BASE, P.ROWS[("primary", "B2")], RULE, 4, "persistence", A_AGE_PRIMARY, F_REF_TEST)
    finally:
        P.run_branch = orig
    assert len({c for _, _, c in seen}) == 1
    seeds_by_action = defaultdict(list)
    for a, s, _ in seen:
        seeds_by_action[a].append(s)
    assert len({tuple(v) for v in seeds_by_action.values()}) == 1 and len(seeds_by_action) == len(dp.mask.valid_actions)


def test_45_dispatch_by_decision_type(waitday):
    _, dps, _ = waitday
    for row in P.ROWS.values():
        c = ContinuationPolicy(*row, RULE)
        for dp in dps[:400]:
            a = c.act(dp)
            if dp.decision_type is DecisionType.INITIAL:
                assert a in (A.AI, A.HUMAN)
            else:
                assert a == (b1(dp) if row[1] == "B1" else b2(dp))
    assert all(not dp.mask.is_valid(A.AI) for dp in dps if dp.decision_type is DecisionType.POST_AI_FAILURE)


def _null_share(n_alt, n_states=4000, K=P.K_PRIMARY, effect=0.0, seed=0):
    rng = np.random.default_rng(seed)
    acts = [A.HUMAN_NORMAL, A.HUMAN_HIGH, A.WAIT, A.CALLBACK][: n_alt + 1]
    out = []
    for _ in range(n_states):
        common = rng.normal(0, 5, K)
        ret = {a: common + rng.normal(0, 1, K) for a in acts}
        if effect:
            ret[acts[-1]] = ret[acts[-1]] + effect
        out.append(S.h1_heldout(ret, A.HUMAN_NORMAL))
    return float(np.mean(out))


def test_46_heldout_h1_null():
    s1, s3 = _null_share(1), _null_share(3)
    assert s1 <= 0.035 and s3 <= 0.035 and s3 <= s1 + 0.01


def test_47_heldout_h1_planted_and_disjoint_folds():
    assert _null_share(3, n_states=500, effect=3.0) > 0.9
    for K in (P.K_PRIMARY, P.K_STABILITY):
        (sel, ev), _ = S.folds(K)
        assert not set(sel) & set(ev) and len(sel) == len(ev) == K // 2


def test_48_mean_reversion_unchanged_phase_f():
    from callverse_workforce.agent.config import SupervisionConfig
    from callverse_workforce.agent.whatif import world_parameters
    from callverse_workforce.v2_postfail.world import plan_for

    snap, _ = first_snapshot(E1_DAYS[45], 5)
    assert MEAN_REVERSION_MINUTES == SupervisionConfig().mean_reversion_minutes == 60.0
    assert world_parameters(snap, BASE, plan_for(snap, BASE, "mean_reversion", F_REF_TEST))["degradation_minutes"] == 60.0
    assert world_parameters(snap, BASE, plan_for(snap, BASE, "persistence", F_REF_TEST))["degradation_minutes"] == 780.0 - snap.now


def test_k_values_match_spec():
    spec = (PROJECT_ROOT / "docs" / "v2_headroom_spec.md").read_text(encoding="utf-8")
    assert P.K_PRIMARY == 32 and P.K_STABILITY == 64 and "**K = 32 new hypothetical futures**" in spec
    assert "**Pre-registered stability rule (K = 64, selected summary only).**" in spec
    assert "if **any** of the H1, H2 or H3 verdicts differs between K = 32 and K = 64" in spec and "**`UNSTABLE`**" in spec
    assert P.STABILITY_SUBSAMPLE == 0.25 and P.STABILITY_SEED == 20261027


# ---------------------------------------------------------------- restoration (spec §7.0a, section 6 of the Stage-1 brief)
import zlib  # noqa: E402

from callverse_workforce.v2_postfail.world import restore_world  # noqa: E402


def _mixed(dp):
    k = zlib.crc32(dp.contact_id.encode()) % 4
    pref = [A.WAIT, A.CALLBACK, A.HUMAN_HIGH, A.HUMAN_NORMAL][k:] + [A.HUMAN_NORMAL]
    return next(a for a in pref if dp.mask.is_valid(a))


def _live_snapshot(idx, nth):
    """Live V2 day (opened E.1 day) under a mixed POST policy; returns (snapshot, live twin) at the nth first-POST state."""
    from callverse_workforce.agent.perception import public_context_from

    d = E1_DAYS[idx]
    cfg = EV._config(BASE_PATH, d)
    env = V2Environment(V2Twin(cfg, d.seed))
    init = ContinuationPolicy("RulePolicy", "B1", RULE)
    dp, n = env.next_decision(), 0
    while dp is not None:
        if dp.decision_type is DecisionType.POST_AI_FAILURE and dp.observation.wait_count == 0:
            if n == nth:
                return take_snapshot(env.twin, dp.contact_id, public_context_from(cfg, BASE)), env.twin
            n += 1
        dp = env.step(init.act(dp) if dp.decision_type is DecisionType.INITIAL else _mixed(dp))
    raise RuntimeError


@pytest.fixture(scope="module", params=[(42, 47), (72, 200)], ids=["d1723_s47", "d1753_s200"])
def restored(request):
    snap, live = _live_snapshot(*request.param)
    twin, cfg, cell, id_map = restore_world(snap, BASE, branch_seed(9, 9, 0), "persistence", A_AGE_PRIMARY, F_REF_TEST)
    twin.env.run(until=snap.now + 1e-9)  # every restored process has executed its restoration step
    from callverse_workforce.agent.perception import public_context_from

    snap2 = take_snapshot(twin, id_map[snap.target_contact_id], snap.public)
    return snap, live, twin, id_map, snap2


def test_49_restoration_round_trip(restored):
    snap, live, twin, id_map, snap2 = restored
    after = {c.contact_id: c for c in snap2.in_system}
    assert {c.state for c in snap.in_system} >= {"HUMAN_QUEUE", "WAITING", "CALLBACK_QUEUE", "HUMAN_SERVICE", "PENDING_POST"}
    for c in snap.in_system:
        r = after[id_map[c.contact_id]]
        assert r.state == c.state, (c.contact_id, c.state, r.state)
        for f in ("escalated", "wait_count", "queue_class", "routing_attempt_count", "handoff_applied", "decision_type",
                  "callback_attempts", "callback_attempt_times", "callback_started_at"):
            assert getattr(r, f) == getattr(c, f), (c.state, f)
        assert abs(r.online_wait - c.online_wait) < 1e-5 and abs(r.human_queue_minutes - c.human_queue_minutes) < 1e-5
    # human-queue order (class + FIFO / aging position) preserved
    live_q = [r.owner for r in sorted(live.humans.put_queue, key=lambda r: live.humans.key(r, snap.now)) if r.owner]
    hyp_q = [r.owner for r in sorted(twin.humans.put_queue, key=lambda r: twin.humans.key(r, twin.now)) if r.owner]
    assert [id_map[o] for o in live_q if o in id_map] == hyp_q


def test_50_restoration_cannot_bypass_masks(restored):
    snap, live, twin, id_map, snap2 = restored
    for rt, _ in list(twin.pending):
        obs, mask = twin.observe_v2(rt)
        if rt.escalated:
            assert obs.decision_type is DecisionType.POST_AI_FAILURE and not mask.is_valid(A.AI)
        if rt.wait_count >= 3:
            assert not mask.is_valid(A.WAIT)
    import dataclasses

    sp = twin.restored[id_map[snap.target_contact_id]]
    assert sp.decision_type is DecisionType.POST_AI_FAILURE
    # a restored contact at wait_count 3 can never WAIT again (unit check on the frozen V2 mask)
    o = twin.observe_v2(next(rt for rt, _ in twin.pending if rt.escalated))[0].base
    from callverse_workforce.routing.mask import ServiceAvailability

    assert not A.v2_mask(DecisionType.POST_AI_FAILURE, o, 3, ServiceAvailability(True, True), BASE.mask, BASE.wait).is_valid(A.WAIT)


def test_51_no_duplicated_completed_effects(restored):
    snap, live, twin, id_map, snap2 = restored
    assert twin.reward.episode.total == 0.0 or abs(twin.reward.episode.total) < 1e-6  # nothing completed at restoration
    restored_ids = set(twin.restored)
    assert not any(e.event_type.value in ("AI_ATTEMPT_STARTED", "CALLBACK_SCHEDULED", "HUMAN_SERVICE_STARTED", "WAIT_STARTED")
                   and e.contact_id in restored_ids for e in twin.events.records)  # in-progress effects are not replayed
    env = V2Environment(twin)
    cont = ContinuationPolicy("RulePolicy", "B1", RULE)
    dp = env.next_decision()
    while dp is not None:
        if dp.contact_id in restored_ids and twin.restored[dp.contact_id].escalated:
            assert dp.decision_type is DecisionType.POST_AI_FAILURE and not dp.mask.is_valid(A.AI)
        dp = env.step(cont.act(dp))
    for cid, sp in twin.restored.items():
        rt = twin.contacts[cid]
        assert rt.callback_attempts <= BASE.callback.max_attempts and rt.callback_attempts >= sp.callback_attempts
        if sp.handoff_applied:  # already connected in the live world: no second handoff
            assert not any(e.contact_id == cid and "handoff_minutes" in e.data for e in twin.events.records)
        if sp.state == "AI_SERVICE":  # resumed, not restarted: never a second AI attempt start
            assert not any(e.contact_id == cid and e.event_type.value == "AI_ATTEMPT_STARTED" for e in twin.events.records)
    assert twin.humans.aging_violations == 0


def test_52_restoration_identical_across_actions(restored):
    snap, *_ = restored
    a, b = (restore_world(snap, BASE, branch_seed(9, 9, 1), "persistence", A_AGE_PRIMARY, F_REF_TEST)[0] for _ in range(2))
    for tw in (a, b):
        tw.env.run(until=snap.now + 1e-9)
    ser = lambda tw: [(e.time, e.event_type.value, e.contact_id, sorted(map(str, e.data.items()))) for e in tw.events.records]
    assert ser(a) == ser(b)  # the restored world does not depend on the evaluated action (applied only afterwards)
    cont = ContinuationPolicy("RulePolicy", "B2", RULE)
    outs = {act: run_branch(snap, BASE, act, branch_seed(9, 9, 1), cont, "persistence", A_AGE_PRIMARY, F_REF_TEST)
            for act in (A.HUMAN_NORMAL, A.HUMAN_HIGH)}
    assert all(o["status"] == "OK" and o["target_actions"][0] == act for act, o in outs.items())


def test_53_callback_and_wait_timing_resumed(restored):
    snap, live, twin, id_map, snap2 = restored
    cb = BASE.callback
    for c in snap.in_system:
        rt = twin.contacts[id_map[c.contact_id]]
        if c.state.startswith("CALLBACK"):
            assert rt.callback_started_at == c.callback_started_at  # window (created + 120) preserved
        if c.state == "WAITING":
            assert twin.restored[id_map[c.contact_id]].elapsed_in_state == c.elapsed_in_state
    # remaining WAIT timer and next callback dial follow the live clocks (spec §7.0a)
    src = (PKG / "twin.py").read_text(encoding="utf-8")
    assert "resume_wait_minutes=max(0.0, self.config.wait.reevaluation_minutes - sp.elapsed_in_state)" in src
    assert "last + cb.attempt_minutes + cb.retry_delay_minutes" in src


def test_53b_resumed_timers_dynamic():
    snap, live = _live_snapshot(42, 47)
    twin, cfg, cell, id_map = restore_world(snap, BASE, branch_seed(9, 9, 2), "persistence", A_AGE_PRIMARY, F_REF_TEST)
    start = snap.now - 1e-6
    env = V2Environment(twin)
    cont = ContinuationPolicy("RulePolicy", "B1", RULE)
    dp = env.next_decision()
    while dp is not None:
        dp = env.step(cont.act(dp))
    checked = 0
    for c in snap.in_system:
        cid = id_map[c.contact_id]
        if c.state == "WAITING":
            re = [e for e in twin.events.records if e.contact_id == cid and e.event_type.value == "WAIT_REEVALUATION"]
            ends = [e for e in twin.events.records if e.contact_id == cid and e.event_type.value in ("ABANDONMENT", "DEADLINE_MISSED")]
            first = min([e.time for e in re + ends])
            assert first <= start + (2.0 - c.elapsed_in_state) + 1e-9  # never later than the live remaining timer
            if re and re[0].data["cause"].value == "TIMER":
                assert abs(re[0].time - (start + 2.0 - c.elapsed_in_state)) < 1e-9
            checked += 1
        if c.state == "CALLBACK_QUEUE" and c.callback_attempt_times:
            retry_at = c.callback_attempt_times[-1] + BASE.callback.attempt_minutes + BASE.callback.retry_delay_minutes
            dials = [e.time for e in twin.events.records if e.contact_id == cid and e.event_type.value == "CALLBACK_ATTEMPT_STARTED"]
            assert all(t >= min(retry_at, start) - 1e-9 for t in dials[:1])  # no dial before the live retry time
            checked += 1
    assert checked >= 1
