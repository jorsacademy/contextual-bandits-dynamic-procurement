from __future__ import annotations

from .benchmark import run_benchmark, summarize


def main() -> None:
    train_seeds = list(range(10))
    test_rows = run_benchmark(seeds=[100, 101, 102, 103, 104], train_seeds=train_seeds, horizon=300)
    shifted_rows = run_benchmark(
        seeds=[200, 201, 202, 203, 204],
        train_seeds=train_seeds,
        horizon=500,
    )

    print("split=test")
    for row in summarize(test_rows):
        print(
            f"{row['method']},regret={row['mean_cumulative_regret']:.3f},"
            f"pre={row['mean_pre_shift_regret']:.3f},"
            f"post={row['mean_post_shift_regret']:.3f},"
            f"cost={row['mean_realized_cost']:.3f},"
            f"failure={row['service_failure_rate']:.3f}"
        )

    print("split=long_horizon_shift")
    for row in summarize(shifted_rows):
        print(
            f"{row['method']},regret={row['mean_cumulative_regret']:.3f},"
            f"pre={row['mean_pre_shift_regret']:.3f},"
            f"post={row['mean_post_shift_regret']:.3f},"
            f"cost={row['mean_realized_cost']:.3f},"
            f"failure={row['service_failure_rate']:.3f}"
        )


if __name__ == "__main__":
    main()
