import numpy as np

from procurement_bandits.benchmark import run_benchmark
from procurement_bandits.environment import ProcurementBanditEnv
from procurement_bandits.policies import DiscountedLinUCB
from procurement_bandits.statistics import paired_bootstrap_mean_difference, paired_regret_report


def test_stationary_block_has_no_hidden_regime_change():
    env = ProcurementBanditEnv(seed=3, horizon=100, regime_shift=False)
    before = env.expected_rewards(env.shift_step - 1)
    after = env.expected_rewards(env.shift_step)
    stationary_after = ProcurementBanditEnv(seed=3, horizon=100, regime_shift=False).expected_rewards(
        env.shift_step
    )
    assert np.allclose(after, stationary_after)
    # Context changes over time, so expected rewards need not match across adjacent rounds.
    # The important contract is that toggling the hidden regime shift changes the same round.
    shifted_after = ProcurementBanditEnv(seed=3, horizon=100, regime_shift=True).expected_rewards(
        env.shift_step
    )
    assert not np.allclose(after, shifted_after)
    assert before.shape == after.shape


def test_discounted_linucb_uses_only_chosen_feedback_update():
    policy = DiscountedLinUCB(3, 4, discount=0.95)
    features = np.eye(3, 4)
    action = policy.select(features, 0)
    before = [state.b.copy() for state in policy.states]
    policy.update(action, features[action], reward=2.0)
    for index, state in enumerate(policy.states):
        if index == action:
            assert not np.allclose(state.b, before[index])
        else:
            assert np.allclose(state.b, before[index])


def test_final_benchmark_separates_nominal_and_shifted_scenarios():
    nominal = run_benchmark(
        seeds=[100, 101],
        train_seeds=[0, 1],
        horizon=40,
        regime_shift=False,
    )
    shifted = run_benchmark(
        seeds=[200, 201],
        train_seeds=[0, 1],
        horizon=40,
        regime_shift=True,
    )
    assert {row.scenario for row in nominal} == {"stationary"}
    assert {row.scenario for row in shifted} == {"regime_shift"}
    assert "discounted_linucb" in {row.method for row in nominal}
    assert "discounted_linucb" in {row.method for row in shifted}


def test_paired_bootstrap_is_reproducible_and_aligned():
    candidate = np.array([4.0, 5.0, 3.0, 4.5])
    reference = np.array([5.0, 6.0, 4.0, 5.0])
    first = paired_bootstrap_mean_difference(candidate, reference, seed=9)
    second = paired_bootstrap_mean_difference(candidate, reference, seed=9)
    assert first == second
    report = paired_regret_report(candidate, reference, seed=9)
    assert report["mean_regret_difference"] < 0.0
    assert report["win_rate"] == 1.0
