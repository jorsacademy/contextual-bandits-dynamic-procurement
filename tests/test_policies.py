import numpy as np

from procurement_bandits.benchmark import run_benchmark, select_static_supplier
from procurement_bandits.environment import ProcurementBanditEnv
from procurement_bandits.policies import (
    DiscountedLinUCB,
    EpsilonGreedyLinear,
    LinearThompson,
    LinUCB,
)


def test_linear_policies_return_valid_actions_after_updates():
    env = ProcurementBanditEnv(seed=3, horizon=20)
    policies = [
        EpsilonGreedyLinear(env.n_actions, env.feature_dim, seed=1),
        LinUCB(env.n_actions, env.feature_dim),
        DiscountedLinUCB(env.n_actions, env.feature_dim),
        LinearThompson(env.n_actions, env.feature_dim, seed=2),
    ]
    for policy in policies:
        for t in range(8):
            features = env.action_features(t)
            action = policy.select(features, t)
            assert 0 <= action < env.n_actions
            reward = env.step(t, action).reward
            policy.update(action, features[action], reward)
        estimates = [np.linalg.norm(state.b) for state in policy.states]
        assert max(estimates) > 0.0


def test_static_supplier_selection_is_reproducible():
    assert select_static_supplier([0, 1, 2], 60) == select_static_supplier([0, 1, 2], 60)


def test_benchmark_reports_all_methods_and_nonnegative_regret():
    rows = run_benchmark(seeds=[10, 11], train_seeds=[0, 1], horizon=40)
    methods = {row.method for row in rows}
    assert methods == {
        "random",
        "static_best_train",
        "epsilon_greedy",
        "linucb",
        "discounted_linucb",
        "linear_thompson",
    }
    assert all(row.cumulative_regret >= -1e-10 for row in rows)
    assert all(0.0 <= row.service_failure_rate <= 1.0 for row in rows)
    assert all(abs(sum(row.action_shares) - 1.0) < 1e-10 for row in rows)
    assert all(row.mean_decision_latency_ms >= 0.0 for row in rows)
