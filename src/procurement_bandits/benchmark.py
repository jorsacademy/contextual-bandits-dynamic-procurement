from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .environment import ProcurementBanditEnv
from .policies import (
    BanditPolicy,
    EpsilonGreedyLinear,
    LinUCB,
    LinearThompson,
    RandomPolicy,
    StaticSupplierPolicy,
)


@dataclass(frozen=True)
class EpisodeSummary:
    method: str
    seed: int
    cumulative_regret: float
    pre_shift_regret: float
    post_shift_regret: float
    mean_reward: float
    mean_realized_cost: float
    service_failure_rate: float
    action_shares: tuple[float, ...]


def select_static_supplier(train_seeds: list[int], horizon: int) -> int:
    """Choose one supplier using training-period average expected utility only."""
    totals = None
    count = 0
    for seed in train_seeds:
        env = ProcurementBanditEnv(seed=seed, horizon=horizon)
        for t in range(env.shift_step):
            rewards = env.expected_rewards(t)
            totals = rewards.copy() if totals is None else totals + rewards
            count += 1
    if totals is None or count == 0:
        raise ValueError("train_seeds must not be empty")
    return int(np.argmax(totals / count))


def make_policy(name: str, env: ProcurementBanditEnv, *, seed: int, static_action: int) -> BanditPolicy:
    if name == "random":
        return RandomPolicy(env.n_actions, seed=seed)
    if name == "static_best_train":
        return StaticSupplierPolicy(static_action)
    if name == "epsilon_greedy":
        return EpsilonGreedyLinear(env.n_actions, env.feature_dim, epsilon=0.1, seed=seed)
    if name == "linucb":
        return LinUCB(env.n_actions, env.feature_dim, alpha=1.0)
    if name == "linear_thompson":
        return LinearThompson(env.n_actions, env.feature_dim, exploration_scale=0.6, seed=seed)
    raise ValueError(f"unknown policy: {name}")


def run_episode(
    method: str,
    *,
    seed: int,
    horizon: int,
    static_action: int,
) -> EpisodeSummary:
    env = ProcurementBanditEnv(seed=seed, horizon=horizon)
    policy = make_policy(method, env, seed=10_000 + seed, static_action=static_action)
    regrets = []
    rewards = []
    costs = []
    failures = []
    actions = []

    for t in range(horizon):
        features = env.action_features(t)
        action = policy.select(features, t)
        expected = env.expected_rewards(t)
        oracle_expected = float(np.max(expected))
        outcome = env.step(t, action)
        policy.update(action, features[action], outcome.reward)

        regrets.append(oracle_expected - float(expected[action]))
        rewards.append(outcome.reward)
        costs.append(outcome.realized_cost)
        failures.append(outcome.service_failure)
        actions.append(action)

    regrets_array = np.asarray(regrets, dtype=float)
    shares = tuple(float(np.mean(np.asarray(actions) == a)) for a in range(env.n_actions))
    return EpisodeSummary(
        method=method,
        seed=seed,
        cumulative_regret=float(np.sum(regrets_array)),
        pre_shift_regret=float(np.sum(regrets_array[: env.shift_step])),
        post_shift_regret=float(np.sum(regrets_array[env.shift_step :])),
        mean_reward=float(np.mean(rewards)),
        mean_realized_cost=float(np.mean(costs)),
        service_failure_rate=float(np.mean(failures)),
        action_shares=shares,
    )


def run_benchmark(
    *,
    seeds: list[int],
    train_seeds: list[int],
    horizon: int = 400,
) -> list[EpisodeSummary]:
    static_action = select_static_supplier(train_seeds, horizon)
    methods = ["random", "static_best_train", "epsilon_greedy", "linucb", "linear_thompson"]
    return [
        run_episode(method, seed=seed, horizon=horizon, static_action=static_action)
        for seed in seeds
        for method in methods
    ]


def summarize(rows: list[EpisodeSummary]) -> list[dict[str, float | str]]:
    output = []
    for method in sorted({row.method for row in rows}):
        selected = [row for row in rows if row.method == method]
        output.append(
            {
                "method": method,
                "mean_cumulative_regret": float(np.mean([r.cumulative_regret for r in selected])),
                "mean_pre_shift_regret": float(np.mean([r.pre_shift_regret for r in selected])),
                "mean_post_shift_regret": float(np.mean([r.post_shift_regret for r in selected])),
                "mean_reward": float(np.mean([r.mean_reward for r in selected])),
                "mean_realized_cost": float(np.mean([r.mean_realized_cost for r in selected])),
                "service_failure_rate": float(np.mean([r.service_failure_rate for r in selected])),
            }
        )
    return output
