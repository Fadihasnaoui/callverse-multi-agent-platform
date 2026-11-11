"""Phase C: nested loads, state sampling, counterfactual branching, paired statistics."""

from __future__ import annotations

import itertools
import json
import math

import numpy as np
import pytest

from callverse_workforce.baselines.policies import AIFirstPolicy, make_policy
from callverse_workforce.config import apply_overrides, load_config
from callverse_workforce.domain.enums import EventType
from callverse_workforce.headroom import experiment as H
from callverse_workforce.headroom.branch import (
    BranchJob,
    DiscountedReturn,

    policy_spec,
    run_branch,
    simulate_branch,
)
from callverse_workforce.headroom.loads import NOISE_LEVELS, headroom_config
from callverse_workforce.headroom.sampling import QUOTA, STRATA, link_states, sample_states, stratum_of
from callverse_workforce.headroom.stats import analyze_state, paired_interval, t_quantile
from callverse_workforce.reward.breakdown import RewardBreakdown
from callverse_workforce.seeds import (
    FINAL_TEST_RESERVED,
    HEADROOM_EXTENSION_SEEDS,
    HEADROOM_PRIMARY_SEEDS,
    SEED_SETS,
    SeedPolicyError,
)
from callverse_workforce.twin.digital_twin import DigitalTwin
from callverse_workforce.twin.generator import ContactGenerator
from callverse_workforce.utils.rng import RngManager, Stream

BASE = load_config()
RULE = {"theta": 0.5, "queue_max": 2, "congested_action": "AI"}  # frozen B2 RulePolicy
SEED = HEADROOM_PRIMARY_SEEDS[0]
SMALL_QUOTA = {"calm": 1, "rising": 1, "peak": 2, "post_peak": 1, "late": 1}


@pytest.fixture(scope="module")
def states():
    st, meta = sample_states(BASE, "NORMAL", SEED, RULE, SMALL_QUOTA)
    return st


def _job(state, action, variant=0, continuation=("RulePolicy", RULE), cfg=BASE):
    return BranchJob(state["state_id"], cfg, state["seed"], H.state_key(state), action, variant,
                     policy_spec("RulePolicy", RULE), policy_spec(*continuation))


def _state_with(states, n_valid=4):
    return next(s for s in states if s["n_valid"] >= n_valid) if any(s["n_valid"] >= n_valid for s in states) else states[0]


# --------------------------------------------------------------------------- seeds
def test_headroom_seeds_disjoint_and_final_test_untouched():  # C-1, C-2
    assert HEADROOM_PRIMARY_SEEDS == tuple(range(1301, 1311))
    assert HEADROOM_EXTENSION_SEEDS == tuple(range(1311, 1321))
    for (a, sa), (b, sb) in itertools.combinations(SEED_SETS.items(), 2):
        assert not set(sa) & set(sb), (a, b)
    with pytest.raises(SeedPolicyError):
        H.build_samples(BASE, (2000,), RULE)
    with pytest.raises(SeedPolicyError):
        H.build_samples(BASE, (1101,), RULE)  # B2 evaluation seeds are not headroom seeds
    assert FINAL_TEST_RESERVED == range(2000, 3000)


def test_frozen_b2_policies_loaded_unchanged():
    frozen = H.load_frozen()
    assert frozen["RulePolicy"] == RULE
    assert frozen["ThresholdPolicy"]["theta"] == 0.30


# --------------------------------------------------------------------------- nested loads
def test_load_levels_nested_and_shared_contacts_identical():  # C-3..C-5
    m = H.nested_manifest(BASE, SEED)
    assert m["normal_subset_of_x1_3"] and m["x1_3_subset_of_x1_5"] and m["shared_contacts_identical"]
    c = m["counts"]
    assert c["NORMAL"] < c["NORMAL_X1_3"] < c["NORMAL_X1_5"]
    assert 1.15 < c["NORMAL_X1_3"] / c["NORMAL"] < 1.45 and 1.3 < c["NORMAL_X1_5"] / c["NORMAL"] < 1.7


def test_only_arrival_volume_changes_between_loads():
    x15 = headroom_config(BASE, "NORMAL_X1_5")
    a, b = BASE.model_dump(), x15.model_dump()
    for d in (a, b):
        d["scenario"].pop("extra_load_layers"), d["scenario"].pop("name"), d["scenario"].pop("purpose")
    assert a == b
    assert x15.humans.n_advisors == 8 and x15.scenario.purpose == "HEADROOM_DIAGNOSTIC_ONLY"


def test_additional_load_rng_does_not_alter_base_contacts():  # C-6
    base = ContactGenerator(BASE, RngManager(SEED)).generate_day()
    heavy = ContactGenerator(apply_overrides(BASE, {"scenario.extra_load_layers": [0.9, 0.7]}), RngManager(SEED)).generate_day()
    heavy_by_id = {c.contact_id: c for c in heavy}
    assert all(heavy_by_id[c.contact_id] == c for c in base)
    extra = [c for c in heavy if c.contact_id not in {x.contact_id for x in base}]
    assert extra and min(c.index for c in extra) >= 100_000


def test_noise_levels_change_only_observed_score():  # C-28, C-29
    days = {n: ContactGenerator(headroom_config(BASE, "NORMAL", n), RngManager(SEED)).generate_day() for n in NOISE_LEVELS}
    assert days["LOW"] == days["MEDIUM"] == days["HIGH"]  # hidden truth (incl. AI success prob) unchanged
    stds = {n: headroom_config(BASE, "NORMAL", n).intake_noise.ai_score_noise_std for n in NOISE_LEVELS}
    assert stds == pytest.approx({"LOW": 0.05, "MEDIUM": 0.10, "HIGH": 0.15})
    from callverse_workforce.intake.analyzer import SimulatedIntakeAnalyzer

    truth = days["MEDIUM"][5]
    obs = {n: SimulatedIntakeAnalyzer(headroom_config(BASE, "NORMAL", n).intake_noise).analyze_request(
        truth, RngManager(SEED).contact_stream(Stream.INTAKE_NOISE, truth.index), 0.0) for n in NOISE_LEVELS}
    strip = lambda o: {k: v for k, v in vars(o).items() if k != "ai_resolution_score"}
    assert strip(obs["LOW"]) == strip(obs["MEDIUM"]) == strip(obs["HIGH"])
    assert len({obs[n].ai_resolution_score for n in NOISE_LEVELS}) > 1 or truth.ai_correct_prob in (0.0, 1.0)


# --------------------------------------------------------------------------- sampling
def test_state_sampling_deterministic_stratified_and_eligible(states):  # C-7..C-9
    again, meta = sample_states(BASE, "NORMAL", SEED, RULE, SMALL_QUOTA)
    assert again == states
    assert all(s["n_valid"] >= 2 for s in states)
    assert {s["stratum"] for s in states} == {name for name, _ in STRATA}
    assert {k: sum(s["stratum"] == k for s in states) for k in SMALL_QUOTA} == SMALL_QUOTA
    assert meta["sampled"] == len(states) and sum(QUOTA.values()) == 30
    assert stratum_of(0.0) == "calm" and stratum_of(119.9) == "calm" and stratum_of(120.0) == "rising"
    assert stratum_of(200) == "peak" and stratum_of(450) == "peak" and stratum_of(350) == "post_peak"
    assert stratum_of(700) == "late"
    for s in states:  # policy actions recorded from the observation only and valid under the mask
        for k in ("rule_action", "aifirst_action", "threshold_action"):
            assert s["mask"][s[k]]
        assert len(s["encoded"]) == 43


def test_sample_not_conditioned_on_disagreement(states):
    assert any(s["rule_action"] == s["aifirst_action"] for s in states)


# --------------------------------------------------------------------------- branching
def test_reconstruction_and_forced_action_once(states):  # C-10, C-11, C-32, C-33
    st = _state_with(states)
    r = run_branch(_job(st, "HUMAN"))
    assert r["status"] == "OK" and r["forced_action_applied"]
    assert r["reward_reconciles"] and r["terminal_reconciles"]
    bad = dict(st, timestamp=st["timestamp"] + 1.0)
    assert run_branch(_job(bad, "HUMAN"))["status"] == "RECONSTRUCTION_FAILED"
    bad2 = dict(st, observation=dict(st["observation"], human_queue_size=st["observation"]["human_queue_size"] + 7))
    assert run_branch(_job(bad2, "HUMAN"))["status"] == "RECONSTRUCTION_FAILED"


def test_invalid_action_never_forced(states):  # C-12
    masked = next((s for s in states if s["n_valid"] < 4), None)
    if masked is None:
        pytest.skip("no masked state in fixture")
    bad_action = next(a for a, ok in masked["mask"].items() if not ok)
    assert run_branch(_job(masked, bad_action))["status"] == "INVALID_FORCED_ACTION"
    jobs = H.make_jobs(BASE, [masked], RULE, ("RulePolicy", RULE), k=2)
    assert {j.action for j in jobs} == {a for a, ok in masked["mask"].items() if ok} and len(jobs) == 2 * masked["n_valid"]


def _run(st, action, variant, continuation=("RulePolicy", RULE)):
    out, env, tracker, d = simulate_branch(_job(st, action, variant, continuation))
    assert out["status"] == "OK"
    return env, tracker, d


def test_continuation_policy_after_forced_action(states):  # C-13, C-34
    st = _state_with(states)
    for cont in (("RulePolicy", RULE), ("AIFirstPolicy", {})):
        env, _, _ = _run(st, "WAIT" if st["mask"]["WAIT"] else "HUMAN", 0, cont)
        policy = make_policy(cont[0], cont[1], SEED)
        after = env.transitions[st["target_index"] + 1:]
        assert after and all(t.action is policy.predict(t.state, t.action_mask) for t in after)


def test_past_fixed_future_varies_and_is_shared_across_actions(states):  # C-14..C-16
    st = _state_with(states)
    envs = {(a, k): _run(st, a, k) for a in ("AI", "HUMAN") for k in (0, 1)}
    before = {key: [e.to_dict() for e in env.twin.events.records[:d["events_before"]]] for key, (env, _, d) in envs.items()}
    assert len({json.dumps(v, sort_keys=True) for v in before.values()}) == 1  # identical past
    decisions_before = {key: [x.to_dict() for x in env.decisions[:st["target_index"]]] for key, (env, _, _) in envs.items()}
    assert len({json.dumps(v, sort_keys=True) for v in decisions_before.values()}) == 1

    def future_arrivals(env, d):
        return [(e.contact_id, e.time, json.dumps(e.to_dict()["data"], sort_keys=True))
                for e in env.twin.events.records[d["events_before"]:] if e.event_type is EventType.ARRIVAL
                and e.contact_id.startswith("V")]

    fa = {key: future_arrivals(env, d) for key, (env, _, d) in envs.items()}
    assert fa[("AI", 0)] == fa[("HUMAN", 0)] and fa[("AI", 1)] == fa[("HUMAN", 1)]  # shared within variant
    assert fa[("AI", 0)] != fa[("AI", 1)]  # differs across variants


def test_future_rng_streams_paired_and_variant_keyed():  # C-17
    truths = ContactGenerator(BASE, RngManager(SEED)).generate_day()
    t = truths[3]
    twins = [DigitalTwin(BASE, SEED, contacts=truths) for _ in range(3)]
    before = twins[0]._stream(Stream.AI_SUCCESS, t).random()
    twins[0].activate_future_variant(2, [])
    twins[1].activate_future_variant(2, [])
    twins[2].activate_future_variant(3, [])
    a, b, c = (tw._stream(Stream.AI_SUCCESS, t).random() for tw in twins)
    assert a == b != c and a != before
    with pytest.raises(RuntimeError):
        twins[0].activate_future_variant(4, [])


def test_branch_values_reproducible_and_worker_independent(states):  # C-18, C-30, C-31
    sub = [_state_with(states)]
    jobs = H.make_jobs(BASE, sub, RULE, ("RulePolicy", RULE), k=2)
    serial = H.execute(jobs, workers=1)
    assert serial == H.execute(jobs, workers=1)
    parallel = H.execute(jobs, workers=2)
    assert parallel == serial
    assert H.execute(jobs, workers=3, chunksize=1) == serial


def test_discounted_return_uses_gamma_power_delta_t():  # C-19
    clock = {"t": 10.0}
    d = DiscountedReturn(lambda: clock["t"], origin=10.0, gamma=0.995)
    clock["t"] = 30.0
    d(RewardBreakdown(resolution=3.0), 0.0)  # event 20 min after the branch point
    assert d.components["resolution"] == pytest.approx(3.0 * 0.995 ** 20)
    d(RewardBreakdown(online_waiting=-1.0), 20.0)  # constant cost rate over [10, 30]
    rate = -1.0 / 20.0
    grid = np.linspace(0, 20, 200001)
    integral = np.trapezoid(rate * 0.995 ** grid, grid)
    assert d.components["online_waiting"] == pytest.approx(integral, rel=1e-6)
    assert d.total == pytest.approx(3.0 * 0.995 ** 20 + integral)
    with pytest.raises(AssertionError):
        d(RewardBreakdown(ai_busy=-1.0), 25.0)  # interval would start before the branch point


def test_branch_return_matches_discounted_transitions(states):
    st = _state_with(states)
    env, tracker, d = _run(st, "AI", 0)
    t0 = d["t_target"]
    approx = sum(t.reward * 0.995 ** (t.timestamp - t0) for t in env.transitions[st["target_index"]:])
    assert tracker.total == pytest.approx(approx, rel=0.02, abs=2.0)  # start-of-interval approximation
    undiscounted = sum(t.reward for t in env.transitions[st["target_index"]:])
    assert sum(tracker.undiscounted.values()) == pytest.approx(undiscounted, abs=1e-6)


# --------------------------------------------------------------------------- statistics
def test_action_values_and_paired_interval():  # C-20, C-21
    rets = {"AI": [10, 12, 11, 13, 9, 10, 12, 11], "HUMAN": [8, 9, 9, 10, 7, 8, 9, 9]}
    res = analyze_state(rets, {"RulePolicy": "HUMAN", "AIFirstPolicy": "AI", "ThresholdPolicy": "AI"}, "x")
    assert res["J"]["AI"] == pytest.approx(11.0) and res["J"]["HUMAN"] == pytest.approx(8.625)
    diff = np.subtract(rets["AI"], rets["HUMAN"])
    half = t_quantile(7) * diff.std(ddof=1) / math.sqrt(8)
    iv = paired_interval(diff)
    assert iv["t_lo"] == pytest.approx(diff.mean() - half) and iv["t_hi"] == pytest.approx(diff.mean() + half)
    assert t_quantile(7) == 2.365
    assert iv["boot_lo"] <= iv["mean"] <= iv["boot_hi"]


def test_best_and_tie_classification_and_near_ties():  # C-22..C-24
    clear = analyze_state({"AI": [10] * 4 + [11] * 4, "HUMAN": [5] * 8}, {"RulePolicy": "AI"}, "a")
    assert clear["label"] == "AI_BEST" and clear["best"] == "AI"
    noisy = analyze_state({"AI": [10, -10, 10, -10, 10, -10, 10, -9], "HUMAN": [0] * 8}, {"RulePolicy": "AI"}, "b")
    assert noisy["label"] == "TIE"  # mean 0.125 higher but not distinguishable
    wait = analyze_state({"AI": [0.0] * 8, "WAIT": [3.0, 3.1, 2.9, 3.0, 3.2, 2.8, 3.0, 3.0]}, {"RulePolicy": "AI"}, "c")
    assert wait["label"] == "WAIT_BEST"
    assert wait["near_tie"] == {"lt_5": True, "lt_10": True, "lt_25": True}
    big = analyze_state({"AI": [0.0] * 8, "HUMAN": [30.0 + 0.1 * i for i in range(8)]}, {"RulePolicy": "AI"}, "d")
    assert big["near_tie"] == {"lt_5": False, "lt_10": False, "lt_25": False}


def test_regret_raw_clear_and_cross_fitted():  # C-25, C-26
    rets = {"AI": [10.0] * 8, "HUMAN": [12.0, 12.5, 11.5, 12.0, 12.2, 11.8, 12.0, 12.0], "WAIT": [9.0] * 8}
    res = analyze_state(rets, {"RulePolicy": "HUMAN", "AIFirstPolicy": "AI", "ThresholdPolicy": "WAIT"}, "r")
    assert res["regret"]["RulePolicy"] == {"action": "HUMAN", "raw": 0.0, "clear": 0.0, "cross_fitted": 0.0, "is_best": True}
    ai = res["regret"]["AIFirstPolicy"]
    assert ai["raw"] == pytest.approx(2.0) and ai["clear"] == pytest.approx(2.0)
    assert ai["cross_fitted"] == pytest.approx(2.0, abs=0.2)
    tie = analyze_state({"AI": [1, -1, 1, -1, 1, -1, 1, -1], "HUMAN": [-1, 1, -1, 1, -1, 1, -1, 1.2]}, {"RulePolicy": "AI"}, "t")
    assert tie["regret"]["RulePolicy"]["clear"] == 0.0  # raw > 0 but not distinguishable -> filtered to 0
    assert tie["regret"]["RulePolicy"]["raw"] >= 0.0


def test_summary_disagreement_extension_and_rough_label():  # C-27, C-35
    states = [{"state_id": f"s{i}", "seed": 1301 + i % 2, "stratum": "peak", "mask": {"AI": True, "HUMAN": True, "WAIT": True, "CALLBACK": False},
               "n_valid": 3, "rule_action": "HUMAN" if i % 2 else "AI", "aifirst_action": "AI", "threshold_action": "AI",
               "ai_resolution_score": 0.5, "human_utilization": 1.0, "human_queue_size": 3, "urgency_pred": "LOW",
               "waiting_time_so_far": 0.0, "request_type": "FAQ"} for i in range(6)]
    results = []
    for st in states:
        for a, base in (("AI", 10.0), ("HUMAN", 12.0), ("WAIT", 9.0)):
            for k in range(8):
                results.append({"state_id": st["state_id"], "action": a, "variant": k, "status": "OK",
                                "discounted_return": base + 0.1 * ((k * 7) % 5)})
    an = H.analyze(states, results)
    s = H.summarize(states, an, [{"eligible_decisions": 800}], results, "t")
    assert s["label_distribution"]["HUMAN_BEST"] == 1.0
    assert "not additive" in s["regret"]["RulePolicy"]["rough_daily_label"]
    assert s["regret"]["AIFirstPolicy"]["raw"]["mean"] == pytest.approx(2.0)
    assert s["regret"]["AIFirstPolicy"]["rough_daily_raw"] == pytest.approx(2.0 * 800)
    dis = s["disagreement"]
    assert dis["n"] == 3 and dis["verdicts"] == {"RULE_CLEARLY_BETTER": 3} and dis["mean_aifirst_regret"] == pytest.approx(2.0)
    assert s["advantage_vs_ai"]["HUMAN"]["clearly_positive_fraction"] == 1.0
    assert s["advantage_vs_ai"]["WAIT"]["clearly_negative_fraction"] == 1.0
    ext = H.extension_needed({"L": {"regret": {"RulePolicy": {"cross_fitted_ci": {"lo": 0.05, "hi": 0.6}}}}})
    assert ext["extend"]
    assert not H.extension_needed({"L": {"regret": {"RulePolicy": {"cross_fitted_ci": {"lo": 0.0, "hi": 0.2}}}}})["extend"]
    failed = H.analyze(states[:1], [dict(r, status="RECONSTRUCTION_FAILED") for r in results[:24]])
    assert failed[0]["status"] == "FAILED"


def test_subset_and_linked_states_deterministic(states):
    sub = H.subset_states(states, {"calm": 1, "rising": 0, "peak": 1, "post_peak": 0, "late": 0})
    assert [s["stratum"] for s in sub] == ["calm", "peak"]
    cfg = headroom_config(BASE, "NORMAL", "HIGH")
    linked = link_states(sub, cfg, "HIGH", RULE)
    assert linked == link_states(sub, cfg, "HIGH", RULE)
    assert all(l["linked_primary_state"] == p["state_id"] and l["noise"] == "HIGH" for l, p in zip(linked, sub))
    assert all(l["n_valid"] >= 2 for l in linked)


def test_posthoc_threshold_does_not_modify_frozen_theta(tmp_path):  # C-36
    from callverse_workforce.baselines.experiments import run_policy_days

    before = H.FROZEN_PATH.read_text(encoding="utf-8")
    run_policy_days(BASE, "ThresholdPolicy", {"theta": 0.0}, (1011,))
    assert H.FROZEN_PATH.read_text(encoding="utf-8") == before
    assert H.load_frozen()["ThresholdPolicy"]["theta"] == 0.30
    # theta = 0 is behaviourally AI-first (score >= 0 always)
    from callverse_workforce.twin import run_simulation

    a = run_simulation(BASE, make_policy("ThresholdPolicy", {"theta": 0.0}, 1011), seed=1011)
    b = run_simulation(BASE, AIFirstPolicy(), seed=1011)
    assert a.total_reward == b.total_reward
