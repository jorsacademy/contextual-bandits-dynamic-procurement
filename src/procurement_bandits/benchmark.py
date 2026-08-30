from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .environment import ProcurementBanditEnv
from .policies import (
    BanditPolicy,
    DiscountedLinUCB,
    EpsilonGreedyLinear,
    LinearThompson,
    LinUCB,
    RandomPolicy,
    StaticSupplierPolicy,
)


@dataclass(frozen=True)
class EpisodeSummary:
    method: str
    scenario: str
    seed: int
    cumulative_regret: float
    pre_shift_regret: float
    post_shift_regret: float
    mean_reward: float
    mean_realized_cost: float
    service_failure_rate: float
    mean_decision_latency_ms: float
    action_shares: tuple[float, ...]


def select_static_supplier(train_seeds: list[int], horizon: int) -> int:
    """Choose one supplier using stationary training-period expected utility only."""
    totals = None
    count = 0
    for seed in train_seeds:
        env = ProcurementBanditEnv(seed=seed, horizon=horizon, regime_shift=False)
        for t in range(horizon):
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
    if name == "discounted_linucb":
        return DiscountedLinUCB(env.n_actions, env.feature_dim, alpha=1.0, discount=0.97)
    if name == "linear_thompson":
        return LinearThompson(env.n_actions, env.feature_dim, exploration_scale=0.6, seed=seed)
    raise ValueError(f"unknown policy: {name}")


def run_episode(
    method: str,
    *,
    seed: int,
    horizon: int,
    static_action: int,
    regime_shift: bool,
) -> EpisodeSummary:
    env = ProcurementBanditEnv(seed=seed, horizon=horizon, regime_shift=regime_shift)
    scenario = "regime_shift" if regime_shift else "stationary"
    policy = make_policy(method, env, seed=10_000 + seed, static_action=static_action)
    regrets = []
    rewards = []
    costs = []
    failures = []
    actions = []
    latencies = []

    for t in range(horizon):
        features = env.action_features(t)
        start = perf_counter()
        action = policy.select(features, t)
        latencies.append((perf_counter() - start) * 1000.0)
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
    split = env.shift_step if regime_shift else horizon
    return EpisodeSummary(
        method=method,
        scenario=scenario,
        seed=seed,
        cumulative_regret=float(np.sum(regrets_array)),
        pre_shift_regret=float(np.sum(regrets_array[:split])),
        post_shift_regret=float(np.sum(regrets_array[split:])),
        mean_reward=float(np.mean(rewards)),
        mean_realized_cost=float(np.mean(costs)),
        service_failure_rate=float(np.mean(failures)),
        mean_decision_latency_ms=float(np.mean(latencies)),
        action_shares=shares,
    )


def run_benchmark(
    *,
    seeds: list[int],
    train_seeds: list[int],
    horizon: int = 400,
    regime_shift: bool = True,
) -> list[EpisodeSummary]:
    static_action = select_static_supplier(train_seeds, horizon)
    methods = [
        "random",
        "static_best_train",
        "epsilon_greedy",
        "linucb",
        "discounted_linucb",
        "linear_thompson",
    ]
    return [
        run_episode(
            method,
            seed=seed,
            horizon=horizon,
            static_action=static_action,
            regime_shift=regime_shift,
        )
        for seed in seeds
        for method in methods
    ]


def summarize(rows: list[EpisodeSummary]) -> list[dict[str, float | str]]:
    output = []
    keys = sorted({(row.scenario, row.method) for row in rows})
    for scenario, method in keys:
        selected = [row for row in rows if row.scenario == scenario and row.method == method]
        output.append(
            {
                "scenario": scenario,
                "method": method,
                "mean_cumulative_regret": float(np.mean([r.cumulative_regret for r in selected])),
                "mean_pre_shift_regret": float(np.mean([r.pre_shift_regret for r in selected])),
                "mean_post_shift_regret": float(np.mean([r.post_shift_regret for r in selected])),
                "mean_reward": float(np.mean([r.mean_reward for r in selected])),
                "mean_realized_cost": float(np.mean([r.mean_realized_cost for r in selected])),
                "service_failure_rate": float(np.mean([r.service_failure_rate for r in selected])),
                "mean_decision_latency_ms": float(
                    np.mean([r.mean_decision_latency_ms for r in selected])
                ),
            }
        )
    return output
