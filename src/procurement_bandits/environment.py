from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Supplier:
    name: str
    base_price: float
    lead_time: float
    defect_rate: float
    reliability: float


@dataclass(frozen=True)
class ProcurementOutcome:
    reward: float
    realized_cost: float
    service_failure: bool
    expected_reward: float


class ProcurementBanditEnv:
    """Synthetic contextual supplier-selection environment.

    ``regime_shift=False`` is a stationary nominal block. When ``regime_shift=True``,
    hidden supplier utility and selected operational failure probabilities change at
    ``shift_step``. Policies never observe the hidden parameters directly.
    """

    def __init__(
        self,
        *,
        seed: int,
        horizon: int = 400,
        shift_fraction: float = 0.5,
        reward_noise: float = 0.25,
        regime_shift: bool = True,
    ) -> None:
        if horizon < 2:
            raise ValueError("horizon must be at least 2")
        if not 0.0 < shift_fraction < 1.0:
            raise ValueError("shift_fraction must lie in (0, 1)")
        self.seed = seed
        self.horizon = horizon
        self.shift_step = round(horizon * shift_fraction)
        self.reward_noise = reward_noise
        self.regime_shift = regime_shift
        self.suppliers = (
            Supplier("alpha", 10.0, 2.0, 0.020, 0.96),
            Supplier("beta", 9.4, 3.0, 0.035, 0.93),
            Supplier("gamma", 10.8, 1.4, 0.015, 0.985),
            Supplier("delta", 9.8, 2.4, 0.025, 0.95),
        )
        self._base_theta = self._build_theta()
        self._shift_theta = self._base_theta.copy()
        self._shift_theta[0, 3] -= 0.9
        self._shift_theta[1, 1] += 0.7
        self._shift_theta[2, 2] += 0.6
        self._shift_theta[3, 3] += 0.5

    @property
    def n_actions(self) -> int:
        return len(self.suppliers)

    @property
    def feature_dim(self) -> int:
        return 6

    def _build_theta(self) -> np.ndarray:
        # Features: intercept, demand_pressure, urgency, market_price,
        # reliability_advantage, lead_time_advantage.
        return np.array(
            [
                [-9.7, -0.4, 0.5, -0.8, 1.3, 0.7],
                [-9.2, -0.2, 0.1, -0.5, 0.8, 0.2],
                [-10.2, -0.1, 1.0, -0.6, 1.6, 1.1],
                [-9.5, -0.3, 0.4, -0.7, 1.0, 0.5],
            ],
            dtype=float,
        )

    def context(self, t: int) -> np.ndarray:
        if not 0 <= t < self.horizon:
            raise ValueError("t outside horizon")
        local = np.random.default_rng(self.seed * 100_003 + t)
        demand_pressure = float(np.clip(local.normal(0.0, 1.0), -2.5, 2.5))
        urgency = float(local.beta(2.0, 2.5))
        market_price = float(np.clip(local.normal(0.0, 0.8), -2.0, 2.0))
        return np.array([demand_pressure, urgency, market_price], dtype=float)

    def action_features(self, t: int) -> np.ndarray:
        z = self.context(t)
        rows = []
        max_lead = max(s.lead_time for s in self.suppliers)
        for supplier in self.suppliers:
            rows.append(
                [
                    1.0,
                    z[0],
                    z[1],
                    z[2],
                    supplier.reliability - 0.90,
                    (max_lead - supplier.lead_time) / max_lead,
                ]
            )
        return np.asarray(rows, dtype=float)

    def expected_rewards(self, t: int) -> np.ndarray:
        shifted = self.regime_shift and t >= self.shift_step
        theta = self._shift_theta if shifted else self._base_theta
        return np.einsum("ad,ad->a", self.action_features(t), theta)

    def oracle_action(self, t: int) -> int:
        return int(np.argmax(self.expected_rewards(t)))

    def _operational_probabilities(self, t: int, action: int) -> tuple[float, float]:
        supplier = self.suppliers[action]
        reliability = supplier.reliability
        defect_rate = supplier.defect_rate
        if self.regime_shift and t >= self.shift_step:
            if action == 0:
                reliability = max(0.0, reliability - 0.12)
                defect_rate = min(1.0, defect_rate + 0.035)
            elif action == 1:
                reliability = min(1.0, reliability + 0.025)
            elif action == 2:
                defect_rate = min(1.0, defect_rate + 0.020)
        return reliability, defect_rate

    def step(self, t: int, action: int) -> ProcurementOutcome:
        if not 0 <= action < self.n_actions:
            raise ValueError("invalid supplier action")
        supplier = self.suppliers[action]
        expected = float(self.expected_rewards(t)[action])
        local = np.random.default_rng(self.seed * 1_000_003 + t * 101 + action)
        reliability, defect_rate = self._operational_probabilities(t, action)
        late = bool(local.random() > reliability)
        defect = bool(local.random() < defect_rate)
        z = self.context(t)
        base_cost = supplier.base_price * (1.0 + 0.04 * z[2])
        late_penalty = 2.5 * (1.0 + z[1]) if late else 0.0
        defect_penalty = 4.0 if defect else 0.0
        realized_cost = float(base_cost + late_penalty + defect_penalty)
        noise = float(local.normal(0.0, self.reward_noise))
        reward = expected - 0.15 * (realized_cost - supplier.base_price) + noise
        return ProcurementOutcome(
            reward=float(reward),
            realized_cost=realized_cost,
            service_failure=late or defect,
            expected_reward=expected,
        )
