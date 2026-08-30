import numpy as np

from procurement_bandits.environment import ProcurementBanditEnv


def test_contexts_are_reproducible():
    first = ProcurementBanditEnv(seed=7, horizon=40)
    second = ProcurementBanditEnv(seed=7, horizon=40)
    assert np.allclose(first.context(12), second.context(12))
    assert np.allclose(first.action_features(12), second.action_features(12))


def test_oracle_selects_max_expected_reward():
    env = ProcurementBanditEnv(seed=5, horizon=50)
    for t in [0, 10, env.shift_step, 49]:
        rewards = env.expected_rewards(t)
        assert env.oracle_action(t) == int(np.argmax(rewards))


def test_regime_shift_changes_hidden_supplier_economics():
    env = ProcurementBanditEnv(seed=11, horizon=100)
    before = env.expected_rewards(env.shift_step - 1)
    after = env.expected_rewards(env.shift_step)
    assert not np.allclose(before, after)


def test_outcome_is_deterministic_for_same_seed_time_action():
    first = ProcurementBanditEnv(seed=17, horizon=80).step(30, 2)
    second = ProcurementBanditEnv(seed=17, horizon=80).step(30, 2)
    assert first == second
