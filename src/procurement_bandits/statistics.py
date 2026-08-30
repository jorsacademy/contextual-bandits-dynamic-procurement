from __future__ import annotations

from math import comb

import numpy as np


def paired_bootstrap_mean_difference(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 4000,
    seed: int = 0,
) -> tuple[float, float, float]:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("candidate and reference must be aligned one-dimensional arrays")
    if candidate.size == 0:
        raise ValueError("paired arrays must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be at least 100")

    differences = candidate - reference
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, differences.size, size=(n_bootstrap, differences.size))
    means = differences[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(differences.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def exact_sign_test(candidate: np.ndarray, reference: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("candidate and reference must be aligned one-dimensional arrays")
    differences = candidate - reference
    nonzero = differences[differences != 0.0]
    n = int(nonzero.size)
    if n == 0:
        return 1.0
    wins = int(np.sum(nonzero < 0.0))
    tail = min(wins, n - wins)
    probability = sum(comb(n, k) for k in range(tail + 1)) / (2**n)
    return float(min(1.0, 2.0 * probability))


def paired_regret_report(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float | int]:
    candidate = np.asarray(candidate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    mean_difference, ci_low, ci_high = paired_bootstrap_mean_difference(
        candidate,
        reference,
        seed=seed,
    )
    differences = candidate - reference
    return {
        "mean_regret_difference": mean_difference,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "median_regret_difference": float(np.median(differences)),
        "win_rate": float(np.mean(differences < 0.0)),
        "sign_test_pvalue": exact_sign_test(candidate, reference),
        "n": int(candidate.size),
    }
