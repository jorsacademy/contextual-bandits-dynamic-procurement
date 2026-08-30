from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BanditPolicy:
    def select(self, features: np.ndarray, t: int) -> int:
        raise NotImplementedError

    def update(self, action: int, feature: np.ndarray, reward: float) -> None:
        return None


class RandomPolicy(BanditPolicy):
    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def select(self, features: np.ndarray, t: int) -> int:
        del features, t
        return int(self.rng.integers(self.n_actions))


class StaticSupplierPolicy(BanditPolicy):
    def __init__(self, action: int) -> None:
        self.action = int(action)

    def select(self, features: np.ndarray, t: int) -> int:
        del features, t
        return self.action


@dataclass
class _LinearArmState:
    a: np.ndarray
    b: np.ndarray


class DisjointLinearPolicy(BanditPolicy):
    def __init__(self, n_actions: int, feature_dim: int, ridge: float = 1.0) -> None:
        self.n_actions = n_actions
        self.feature_dim = feature_dim
        self.states = [
            _LinearArmState(np.eye(feature_dim) * ridge, np.zeros(feature_dim))
            for _ in range(n_actions)
        ]

    def _theta(self, action: int) -> np.ndarray:
        state = self.states[action]
        return np.linalg.solve(state.a, state.b)

    def update(self, action: int, feature: np.ndarray, reward: float) -> None:
        x = np.asarray(feature, dtype=float)
        state = self.states[action]
        state.a += np.outer(x, x)
        state.b += reward * x


class EpsilonGreedyLinear(DisjointLinearPolicy):
    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        *,
        epsilon: float = 0.1,
        seed: int = 0,
        ridge: float = 1.0,
    ) -> None:
        super().__init__(n_actions, feature_dim, ridge=ridge)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def select(self, features: np.ndarray, t: int) -> int:
        del t
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        values = [float(features[a] @ self._theta(a)) for a in range(self.n_actions)]
        return int(np.argmax(values))


class LinUCB(DisjointLinearPolicy):
    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        *,
        alpha: float = 1.0,
        ridge: float = 1.0,
    ) -> None:
        super().__init__(n_actions, feature_dim, ridge=ridge)
        self.alpha = alpha

    def select(self, features: np.ndarray, t: int) -> int:
        del t
        scores = []
        for action, state in enumerate(self.states):
            x = features[action]
            theta = np.linalg.solve(state.a, state.b)
            variance = float(x @ np.linalg.solve(state.a, x))
            scores.append(float(x @ theta + self.alpha * np.sqrt(max(variance, 0.0))))
        return int(np.argmax(scores))


class LinearThompson(DisjointLinearPolicy):
    def __init__(
        self,
        n_actions: int,
        feature_dim: int,
        *,
        exploration_scale: float = 0.5,
        seed: int = 0,
        ridge: float = 1.0,
    ) -> None:
        super().__init__(n_actions, feature_dim, ridge=ridge)
        self.exploration_scale = exploration_scale
        self.rng = np.random.default_rng(seed)

    def select(self, features: np.ndarray, t: int) -> int:
        del t
        scores = []
        for action, state in enumerate(self.states):
            mean = np.linalg.solve(state.a, state.b)
            cov = self.exploration_scale**2 * np.linalg.inv(state.a)
            sample = self.rng.multivariate_normal(mean, cov)
            scores.append(float(features[action] @ sample))
        return int(np.argmax(scores))
