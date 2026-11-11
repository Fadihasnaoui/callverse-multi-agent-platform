"""Reward accumulator arithmetic and intake analyzer."""

from __future__ import annotations

import numpy as np
import pytest

from callverse_workforce.config import load_config
from callverse_workforce.domain.enums import Urgency
from callverse_workforce.intake.analyzer import SimulatedIntakeAnalyzer
from callverse_workforce.reward.accumulator import RewardAccumulator
from callverse_workforce.reward.breakdown import REWARD_COMPONENTS, RewardBreakdown
from helpers import make_config, make_contact

R = load_config().reward


def test_false_ai_removes_credit_and_adds_penalty():  # smoke 18 (unit)
    acc = RewardAccumulator(R)
    v = 1.6
    acc.resolution(v)
    acc.false_ai_detected(v)
    ep = acc.episode
    assert ep.resolution == pytest.approx(2 * v)
    assert ep.false_ai == pytest.approx(-(2 * v + 4))
    assert ep.total == pytest.approx(-4.0)


def test_time_accrual_uses_all_contacts():
    acc = RewardAccumulator(R)
    acc.accrue_time(2.0, n_human_busy=3, n_ai_busy=4, n_online_waiting=10, n_callback_pending=5)
    b = acc.drain()
    assert b.online_waiting_contact_minutes == 20
    assert b.online_waiting == pytest.approx(-0.05 * 20)
    assert b.human_busy == pytest.approx(-0.08 * 6)
    assert b.ai_busy == pytest.approx(-0.01 * 8)
    assert b.callback_pending == pytest.approx(-0.02 * 10)
    assert acc.drain().total == 0.0  # drained


def test_event_coefficients():
    acc = RewardAccumulator(R)
    v = 1.25
    assert acc.abandonment(v) == pytest.approx(-3 * v)
    assert acc.missed_deadline(v) == pytest.approx(-4 * v)
    assert acc.callback_failure() == pytest.approx(-1.0)
    assert acc.unfinished(v) == pytest.approx(-2 * v)


def test_breakdown_total_sums_only_reward_components():
    b = RewardBreakdown(resolution=3.0, abandonment=-1.0, online_waiting_contact_minutes=500.0, n_resolutions=7)
    assert b.total == pytest.approx(2.0)
    assert set(REWARD_COMPONENTS) <= set(b.to_dict())


def test_noiseless_intake_reproduces_observable_truth():
    truth = make_contact(3, 12.0, urgency=Urgency.HIGH, deadline_offset_minutes=90.0, ai_correct_prob=0.4)
    obs = SimulatedIntakeAnalyzer(make_config().intake_noise).analyze_request(truth, np.random.default_rng(0), 12.0)
    assert obs.urgency_pred is Urgency.HIGH
    assert obs.request_type is truth.request_type
    assert obs.predicted_deadline_at == pytest.approx(truth.deadline_at)
    assert obs.ai_resolution_score == pytest.approx(0.4)


def test_noisy_intake_makes_errors_at_configured_rate():
    cfg = make_config(intake_noise__urgency_error_prob=0.5)
    analyzer = SimulatedIntakeAnalyzer(cfg.intake_noise)
    truth = make_contact(0, 0.0, urgency=Urgency.MEDIUM)
    wrong = sum(
        analyzer.analyze_request(truth, np.random.default_rng(i), 0.0).urgency_pred is not Urgency.MEDIUM
        for i in range(2000)
    )
    assert 0.45 < wrong / 2000 < 0.55

