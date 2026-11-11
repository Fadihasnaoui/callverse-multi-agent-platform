"""Scripted scenarios exercising each routing path of the Digital Twin."""

from __future__ import annotations

import pytest

from callverse_workforce.domain.enums import EventType as E
from callverse_workforce.domain.enums import Outcome, RoutingAction, Urgency, WakeCause
from callverse_workforce.twin import InvalidActionError, RoutingEnvironment
from helpers import by_contact, contact, events_of, fixed, make_config, make_contact, run_scripted

A = RoutingAction
C0, C1, C2, C3, C4 = (f"C{i:05d}" for i in range(5))


def test_empty_day_with_scripted_contacts():  # smoke 1
    env, result, decisions = run_scripted(make_config(), [], fixed(A.HUMAN))
    assert decisions == [] and result.transitions == [] and result.contacts == []
    assert result.total_reward == 0.0
    assert env.done


def test_empty_day_with_zero_arrival_rate():  # smoke 1 (generated)
    cfg = make_config(scenario__hourly_arrival_rates=[0.0] * 12)
    _, result, decisions = run_scripted(cfg, None, fixed(A.HUMAN))
    assert decisions == [] and result.metrics["generated_contacts"] == 0
    assert result.total_reward == 0.0


def test_ai_success_resolves_contact():  # smoke 4
    c = make_contact(0, 1.0, ai_correct_prob=1.0, ai_service_minutes=3.0, request_value=0.5)
    _, result, _ = run_scripted(make_config(), [c], fixed(A.AI))
    s = contact(result, C0)
    assert s.outcome is Outcome.RESOLVED_AI
    assert s.terminal_time == pytest.approx(4.0)
    assert len(events_of(result, E.AI_ATTEMPT_STARTED)) == 1
    assert len(events_of(result, E.AI_RESOLVED)) == 1
    assert result.episode_reward.resolution == pytest.approx(2 * 1.5)
    assert result.episode_reward.ai_busy_minutes == pytest.approx(3.0)
    assert result.episode_reward.human_busy_minutes == 0.0


def test_visible_ai_failure_escalates_to_human_queue():  # smoke 5
    c = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=0.0, ai_service_minutes=2.0,
                     human_service_minutes=5.0)
    _, result, decisions = run_scripted(make_config(), [c], fixed(A.AI))
    types = [e.event_type for e in result.events if e.contact_id == C0]
    i_fail = types.index(E.AI_VISIBLE_FAILURE)
    assert types[i_fail + 1] is E.AI_ESCALATION
    assert types[i_fail + 2] is E.HUMAN_QUEUE_ENTERED
    assert events_of(result, E.HUMAN_QUEUE_ENTERED, C0)[0].data["escalated"] is True
    s = contact(result, C0)
    assert s.outcome is Outcome.RESOLVED_HUMAN and s.escalated
    assert s.terminal_time == pytest.approx(7.0)
    assert len(decisions) == 1  # escalation is automatic, not a new routing decision


def test_false_ai_resolution_is_discovered_later():  # smoke 6
    c = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=1.0, ai_service_minutes=2.0,
                     false_discovery_delay_minutes=30.0)
    _, result, _ = run_scripted(make_config(), [c], fixed(A.AI))
    false_ev = events_of(result, E.AI_FALSE_RESOLUTION, C0)
    found = events_of(result, E.FALSE_AI_DISCOVERED, C0)
    assert len(false_ev) == 1 and false_ev[0].time == pytest.approx(2.0)
    assert len(found) == 1 and found[0].time == pytest.approx(32.0)
    assert contact(result, C0).outcome is Outcome.FALSE_AI_DISCOVERED
    assert not events_of(result, E.AI_ESCALATION)


def test_false_ai_penalty_cancels_earlier_credit():  # smoke 18
    c = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=1.0, request_value=0.25,
                     false_discovery_delay_minutes=30.0)
    _, result, _ = run_scripted(make_config(), [c], fixed(A.AI))
    ep = result.episode_reward
    v = 1.25
    assert ep.resolution == pytest.approx(2 * v)  # credited once when it looked resolved
    assert ep.false_ai == pytest.approx(-(2 * v + 4))
    assert ep.resolution + ep.false_ai == pytest.approx(-4.0)  # net effect: credit removed + 4
    assert ep.n_resolutions == 1 and ep.n_false_ai == 1 and ep.unfinished == 0.0


def test_undiscovered_false_resolution_is_settled_at_horizon():
    cfg = make_config(scenario__arrival_window_minutes=60, scenario__drain_minutes=0)
    c = make_contact(0, 0.0, ai_correct_prob=0.0, ai_false_resolution_prob=1.0,
                     false_discovery_delay_minutes=500.0)
    _, result, _ = run_scripted(cfg, [c], fixed(A.AI))
    assert len(events_of(result, E.FALSE_AI_SETTLED_AT_HORIZON, C0)) == 1
    assert contact(result, C0).outcome is Outcome.FALSE_AI_DISCOVERED
    assert result.episode_reward.unfinished == 0.0


def test_human_with_no_free_advisor_enters_queue():  # smoke 7
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=10.0), make_contact(1, 1.0, human_service_minutes=4.0)]
    _, result, decisions = run_scripted(cfg, contacts, fixed(A.HUMAN))
    assert decisions[1].observation.humans_available == 0
    q = events_of(result, E.HUMAN_QUEUE_ENTERED, C1)[0]
    assert q.time == pytest.approx(1.0) and q.data["humans_idle"] == 0
    start = events_of(result, E.HUMAN_SERVICE_STARTED, C1)[0]
    assert start.time == pytest.approx(10.0)
    assert contact(result, C1).online_wait_minutes == pytest.approx(9.0)
    assert contact(result, C1).outcome is Outcome.RESOLVED_HUMAN


def test_wait_reevaluates_after_two_minutes():  # smoke 8
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=100.0), make_contact(1, 1.0)]
    _, result, decisions = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], C1: [A.WAIT, A.HUMAN]}))
    c1_decisions = [d for d in decisions if d.contact_id == C1]
    assert [d.timestamp for d in c1_decisions] == pytest.approx([1.0, 3.0])
    reeval = events_of(result, E.WAIT_REEVALUATION, C1)
    assert len(reeval) == 1 and reeval[0].data["cause"] is WakeCause.TIMER
    assert c1_decisions[1].observation.routing_attempt_count == 1
    assert c1_decisions[1].observation.waiting_time_so_far == pytest.approx(2.0)


def test_wait_reevaluates_early_when_human_becomes_free():  # smoke 9
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=1.0), make_contact(1, 0.5)]
    _, result, decisions = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], C1: [A.WAIT, A.HUMAN]}))
    reeval = events_of(result, E.WAIT_REEVALUATION, C1)
    assert len(reeval) == 1  # no duplicate re-evaluation from the later 2-minute timer
    assert reeval[0].time == pytest.approx(1.0)
    assert reeval[0].data["cause"] is WakeCause.HUMAN_AVAILABLE
    c1_decisions = [d for d in decisions if d.contact_id == C1]
    assert [d.timestamp for d in c1_decisions] == pytest.approx([0.5, 1.0])
    assert c1_decisions[1].observation.humans_available == 1
    assert contact(result, C1).outcome is Outcome.RESOLVED_HUMAN


def test_one_free_human_wakes_only_the_oldest_wait_contact():
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=1.0), make_contact(1, 0.2), make_contact(2, 0.4)]
    _, result, _ = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], C1: [A.WAIT], C2: [A.WAIT]}))
    first_c1 = events_of(result, E.WAIT_REEVALUATION, C1)[0]
    first_c2 = events_of(result, E.WAIT_REEVALUATION, C2)[0]
    assert first_c1.data["cause"] is WakeCause.HUMAN_AVAILABLE and first_c1.time == pytest.approx(1.0)
    assert first_c2.data["cause"] is WakeCause.TIMER and first_c2.time == pytest.approx(2.4)


def test_wait_unavailable_after_three_waits():  # smoke 10
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=500.0), make_contact(1, 1.0)]
    plan = by_contact({C0: [A.HUMAN]})

    def chooser(dp):
        if dp.contact_id == C1 and dp.action_mask.is_valid(A.WAIT):
            return A.WAIT
        return plan(dp)

    _, result, decisions = run_scripted(cfg, contacts, chooser)
    c1 = [d for d in decisions if d.contact_id == C1]
    assert len(c1) == 4
    assert [d.action_mask.is_valid(A.WAIT) for d in c1] == [True, True, True, False]
    assert c1[3].action_mask.reasons["WAIT"] == ("max_waits_reached",)
    assert contact(result, C1).n_waits == 3
    assert len(events_of(result, E.WAIT_STARTED, C1)) == 3


def test_masked_action_is_rejected():
    env = RoutingEnvironment(make_config(), contacts=[make_contact(0, 0.0, callback_allowed=False)])
    env.reset()
    with pytest.raises(InvalidActionError):
        env.step(A.CALLBACK)


def test_abandonment_penalty_occurs_once_and_blocks_missed_deadline():  # smoke 16 + 17
    cfg = make_config(humans__n_advisors=1)
    contacts = [
        make_contact(0, 0.0, human_service_minutes=100.0),
        make_contact(1, 0.0, patience_minutes=5.0, deadline_offset_minutes=8.0, request_value=0.5),
    ]
    _, result, _ = run_scripted(cfg, contacts, fixed(A.HUMAN))
    assert len(events_of(result, E.ABANDONMENT, C1)) == 1
    assert events_of(result, E.ABANDONMENT, C1)[0].time == pytest.approx(5.0)
    assert not events_of(result, E.DEADLINE_MISSED)
    ep = result.episode_reward
    assert ep.n_abandonments == 1 and ep.abandonment == pytest.approx(-3 * 1.5)
    assert ep.n_missed_deadlines == 0 and ep.missed_deadline == 0.0
    assert contact(result, C1).outcome is Outcome.ABANDONED


def test_patience_is_cumulative_across_wait_and_queue():
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=100.0), make_contact(1, 0.0, patience_minutes=3.0)]
    _, result, _ = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], C1: [A.WAIT, A.HUMAN]}))
    ab = events_of(result, E.ABANDONMENT, C1)
    assert len(ab) == 1 and ab[0].time == pytest.approx(3.0)  # 2 min WAIT + 1 min in queue


def test_missed_deadline_while_queued_is_terminal_and_charged_once():
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=100.0),
                make_contact(1, 0.0, patience_minutes=1000.0, deadline_offset_minutes=20.0, request_value=0.0)]
    _, result, _ = run_scripted(cfg, contacts, fixed(A.HUMAN))
    missed = events_of(result, E.DEADLINE_MISSED, C1)
    assert len(missed) == 1 and missed[0].time == pytest.approx(20.0)
    assert result.episode_reward.missed_deadline == pytest.approx(-4.0)
    assert not events_of(result, E.ABANDONMENT)
    assert contact(result, C1).outcome is Outcome.MISSED_DEADLINE


def test_callback_pending_time_creates_cost():  # smoke 19
    cfg = make_config(humans__n_advisors=1, callback__model="b1", callback__b1__reach_probability=1.0,
                      callback__attempt_minutes=1.0)
    contacts = [make_contact(0, 0.0, human_service_minutes=10.0), make_contact(1, 0.0, human_service_minutes=4.0)]
    _, result, _ = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], C1: [A.CALLBACK]}))
    s = contact(result, C1)
    assert s.outcome is Outcome.RESOLVED_CALLBACK
    assert s.callback_pending_minutes == pytest.approx(11.0)  # 10 min queued + 1 min dial attempt
    ep = result.episode_reward
    assert ep.callback_pending_contact_minutes == pytest.approx(11.0)
    assert ep.callback_pending == pytest.approx(-0.02 * 11.0)
    assert ep.online_waiting_contact_minutes == 0.0  # callbacks are not online waiting


def test_callback_failure_after_max_attempts():
    cfg = make_config(callback__model="b1", callback__b1__reach_probability=0.0, callback__b1__max_attempts=2,
                      callback__retry_delay_minutes=5.0)
    _, result, _ = run_scripted(cfg, [make_contact(0, 0.0)], fixed(A.CALLBACK))
    assert len(events_of(result, E.CALLBACK_ATTEMPT_FAILED, C0)) == 2
    assert contact(result, C0).outcome is Outcome.CALLBACK_FAILED
    assert result.episode_reward.callback_failure == pytest.approx(-1.0)


def test_unfinished_callback_penalized_at_end_of_day():  # smoke 20
    cfg = make_config(humans__n_advisors=1, scenario__arrival_window_minutes=60, scenario__drain_minutes=0,
                      callback__model="b1")
    contacts = [make_contact(0, 0.0, human_service_minutes=100.0, request_value=0.0),
                make_contact(1, 1.0, request_value=0.5)]
    _, result, _ = run_scripted(cfg, contacts, by_contact({C0: [A.HUMAN], C1: [A.CALLBACK]}))
    s = contact(result, C1)
    assert s.outcome is Outcome.UNFINISHED and s.terminal_time == pytest.approx(60.0)
    assert s.callback_pending_minutes == pytest.approx(59.0)
    unfinished = events_of(result, E.UNFINISHED_AT_HORIZON, C1)
    assert len(unfinished) == 1 and unfinished[0].data["reward"] == pytest.approx(-2 * 1.5)
    ep = result.episode_reward
    assert ep.n_unfinished == 2  # the in-service contact C0 is unfinished too
    assert ep.unfinished == pytest.approx(-2 * 1.0 - 2 * 1.5)
    assert result.transitions[-1].reward_breakdown.unfinished == pytest.approx(ep.unfinished)


def test_online_waiting_cost_counts_all_waiting_contacts():  # smoke 21
    cfg = make_config(humans__n_advisors=1)
    contacts = [make_contact(0, 0.0, human_service_minutes=100.0)]
    contacts += [make_contact(i, 0.0) for i in (1, 2, 3)]
    contacts += [make_contact(4, 10.0)]
    _, result, _ = run_scripted(cfg, contacts, fixed(A.HUMAN))
    tr = next(t for t in result.transitions if t.contact_id == C3)
    assert tr.timestamp == 0.0 and tr.delta_t_minutes == pytest.approx(10.0)
    assert tr.reward_breakdown.online_waiting_contact_minutes == pytest.approx(30.0)  # 3 contacts x 10 min
    assert tr.reward_breakdown.online_waiting == pytest.approx(-0.05 * 30.0)
    assert tr.reward_breakdown.human_busy_minutes == pytest.approx(10.0)


def test_delta_t_between_decision_events():  # smoke 22
    contacts = [make_contact(0, 0.0), make_contact(1, 5.0), make_contact(2, 12.0)]
    _, result, _ = run_scripted(make_config(), contacts, fixed(A.HUMAN))
    assert [t.timestamp for t in result.transitions] == pytest.approx([0.0, 5.0, 12.0])
    assert [t.delta_t_minutes for t in result.transitions] == pytest.approx([5.0, 7.0, 780.0 - 12.0])
    assert [t.done for t in result.transitions] == [False, False, True]
    assert result.transitions[0].next_state == result.transitions[1].state
    assert result.transitions[-1].next_state is None


def test_simultaneous_decisions_have_zero_delta_t():
    contacts = [make_contact(0, 3.0), make_contact(1, 3.0)]
    _, result, _ = run_scripted(make_config(), contacts, fixed(A.HUMAN))
    assert result.transitions[0].delta_t_minutes == 0.0


def test_urgent_contact_cannot_be_deferred_in_the_twin():
    c = make_contact(0, 0.0, urgency=Urgency.HIGH, deadline_offset_minutes=60.0, human_requested=True)
    env = RoutingEnvironment(make_config(), contacts=[c])
    dp = env.reset()
    assert dp.action_mask.valid_actions == (A.HUMAN,)
