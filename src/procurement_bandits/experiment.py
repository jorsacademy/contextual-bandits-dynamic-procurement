from __future__ import annotations

import numpy as np

from .benchmark import EpisodeSummary, run_benchmark, summarize
from .statistics import paired_regret_report


def _paired(rows: list[EpisodeSummary], method: str, reference: str) -> dict[str, float | int]:
    candidate_rows = sorted(
        [row for row in rows if row.method == method],
        key=lambda row: row.seed,
    )
    reference_rows = sorted(
        [row for row in rows if row.method == reference],
        key=lambda row: row.seed,
    )
    if [row.seed for row in candidate_rows] != [row.seed for row in reference_rows]:
        raise ValueError("paired methods must share identical environment seeds")
    return paired_regret_report(
        np.asarray([row.cumulative_regret for row in candidate_rows]),
        np.asarray([row.cumulative_regret for row in reference_rows]),
        seed=23,
    )


def _print_block(name: str, rows: list[EpisodeSummary]) -> None:
    print(f"split={name}")
    for row in summarize(rows):
        print(
            f"{row['method']},regret={row['mean_cumulative_regret']:.3f},"
            f"pre={row['mean_pre_shift_regret']:.3f},"
            f"post={row['mean_post_shift_regret']:.3f},"
            f"cost={row['mean_realized_cost']:.3f},"
            f"failure={row['service_failure_rate']:.3f},"
            f"latency_ms={row['mean_decision_latency_ms']:.4f}"
        )
    for method in ["epsilon_greedy", "linucb", "discounted_linucb", "linear_thompson"]:
        report = _paired(rows, method, "static_best_train")
        print(
            f"paired,{method}-static_best_train,"
            f"mean_diff={report['mean_regret_difference']:.3f},"
            f"ci95=[{report['ci95_low']:.3f},{report['ci95_high']:.3f}],"
            f"win_rate={report['win_rate']:.3f},"
            f"p={report['sign_test_pvalue']:.4f}"
        )


def main() -> None:
    train_seeds = list(range(20))
    nominal_rows = run_benchmark(
        seeds=list(range(100, 110)),
        train_seeds=train_seeds,
        horizon=400,
        regime_shift=False,
    )
    shifted_rows = run_benchmark(
        seeds=list(range(200, 210)),
        train_seeds=train_seeds,
        horizon=400,
        regime_shift=True,
    )
    _print_block("nominal_final", nominal_rows)
    _print_block("supplier_regime_shift", shifted_rows)


if __name__ == "__main__":
    main()
