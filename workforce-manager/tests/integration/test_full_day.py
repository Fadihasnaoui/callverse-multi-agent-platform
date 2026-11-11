"""Full normal-day runs: reproducibility, common random numbers and accounting invariants."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from callverse_workforce.config import load_config
from callverse_workforce.domain.enums import EventType as E
from callverse_workforce.domain.enums import Outcome, RoutingAction
from callverse_workforce.routing import DevelopmentPolicy
from callverse_workforce.tracing.io import save_result
from callverse_workforce.twin import run_simulation

TERMINAL_EVENTS = {
    E.CONTACT_RESOLVED, E.FALSE_AI_DISCOVERED, E.FALSE_AI_SETTLED_AT_HORIZON, E.ABANDONMENT,
    E.DEADLINE_MISSED, E.CALLBACK_FAILED, E.CALLBACK_WINDOW_EXPIRED, E.CALLBACK_DROPOUT, E.UNFINISHED_AT_HORIZON,
}


class AlwaysHuman:
    def reset(self) -> None:
        pass

    def predict(self, observation, action_mask):
        return RoutingAction.HUMAN


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def day(config):
    return run_simulation(config, DevelopmentPolicy(), seed=123)


def _arrivals(result):
    """Exogenous (depth-0) arrivals and their intake predictions; recontacts are endogenous."""
    exogenous = {c.contact_id for c in result.contacts if c.recontact_depth == 0}
    return [(e.contact_id, e.time, json.dumps(e.to_dict()["data"], sort_keys=True))
            for e in result.events
            if e.event_type in (E.ARRIVAL, E.INTAKE_PREDICTION) and e.contact_id in exogenous]


def test_deterministic_replay_with_same_seed(config, day):  # smoke 2
    again = run_simulation(config, DevelopmentPolicy(), seed=123)
    assert [e.to_dict() for e in again.events] == [e.to_dict() for e in day.events]
    assert [t.to_dict() for t in again.transitions] == [t.to_dict() for t in day.transitions]
    assert again.total_reward == day.total_reward


def test_different_seed_changes_generated_contacts(config, day):  # smoke 3
    other = run_simulation(config, DevelopmentPolicy(), seed=124)
    assert _arrivals(other) != _arrivals(day)


def test_routing_choices_do_not_change_arrivals(config, day):
    """Common random numbers: a different policy sees the exact same demand and intake."""
    human_only = run_simulation(config, AlwaysHuman(), seed=123)
    assert _arrivals(human_only) == _arrivals(day)
    assert human_only.total_reward != day.total_reward


def test_all_routing_paths_exercised(day):
    m = day.metrics
    assert all(n > 0 for n in m["action_distribution"].values())
    assert m["ai_visible_failures"] > 0 and m["false_ai_resolutions"] > 0
    assert m["wait_reevaluations"] > 0 and m["callbacks_scheduled"] > 0


def test_every_contact_has_exactly_one_terminal_outcome(day):
    terminal = Counter(e.contact_id for e in day.events if e.event_type in TERMINAL_EVENTS)
    assert all(c.outcome is not None for c in day.contacts)
    assert set(terminal) == {c.contact_id for c in day.contacts}
    assert set(terminal.values()) == {1}
    ab = {e.contact_id for e in day.events if e.event_type is E.ABANDONMENT}
    missed = {e.contact_id for e in day.events if e.event_type is E.DEADLINE_MISSED}
    assert not ab & missed


def test_transition_rewards_sum_to_episode_reward(day):
    total = sum(t.reward for t in day.transitions) + day.pre_decision_reward.total
    assert total == pytest.approx(day.episode_reward.total)
    for t in day.transitions:
        assert t.reward == pytest.approx(t.reward_breakdown.total)


def test_exposures_match_per_contact_accounting(config, day):
    ep = day.episode_reward
    assert ep.online_waiting_contact_minutes == pytest.approx(sum(c.online_wait_minutes for c in day.contacts))
    assert ep.callback_pending_contact_minutes == pytest.approx(sum(c.callback_pending_minutes for c in day.contacts))
    assert ep.human_busy_minutes <= config.humans.n_advisors * config.scenario.horizon_minutes + 1e-6
    assert ep.n_resolutions == day.metrics["resolved_contacts"] + day.metrics["false_ai_resolutions"]


def test_smdp_timing_metadata(day):
    tr = day.transitions
    assert len(tr) == len(day.decisions)
    assert len({t.decision_id for t in tr}) == len(tr)
    for cur, nxt in zip(tr, tr[1:]):
        assert cur.delta_t_minutes == pytest.approx(nxt.timestamp - cur.timestamp)
        assert cur.delta_t_minutes >= 0 and not cur.done
        assert cur.next_state == nxt.state and cur.next_action_mask == nxt.action_mask
    assert tr[-1].done and tr[-1].next_state is None
    assert tr[-1].timestamp + tr[-1].delta_t_minutes == pytest.approx(day.horizon_minutes)


def test_masks_and_actions_are_consistent(day):  # smoke 24 (full day)
    for d in day.decisions:
        assert d.n_valid_actions >= 1
        assert d.action_mask.is_valid(d.action)
        assert d.action_mask.is_valid(RoutingAction.HUMAN)


def test_wait_bookkeeping_has_no_duplicate_reevaluations(config, day):
    reevals = Counter(e.contact_id for e in day.events if e.event_type is E.WAIT_REEVALUATION)
    for c in day.contacts:
        assert c.n_waits <= config.wait.max_waits
        assert reevals.get(c.contact_id, 0) == c.n_decisions - 1
        assert reevals.get(c.contact_id, 0) <= c.n_waits


def test_false_ai_resolutions_are_all_settled(day):
    false_ids = {e.contact_id for e in day.events if e.event_type is E.AI_FALSE_RESOLUTION}
    outcomes = {c.contact_id: c.outcome for c in day.contacts}
    assert false_ids and all(outcomes[cid] is Outcome.FALSE_AI_DISCOVERED for cid in false_ids)


def test_results_are_saved_as_structured_files(day, tmp_path):
    paths = save_result(day, tmp_path)
    summary = json.loads(paths["summary"].read_text())
    assert summary["generated_contacts"] == len(day.contacts)
    first = json.loads(paths["transitions"].read_text().splitlines()[0])
    for key in ("decision_id", "contact_id", "timestamp", "state", "action", "action_mask", "reward",
                "reward_breakdown", "next_state", "delta_t_minutes", "done"):
        assert key in first
    assert len(paths["events"].read_text().splitlines()) == len(day.events)
