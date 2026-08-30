# Contextual Bandits for Dynamic Procurement

Research-oriented Industrial Engineering / Operations Research benchmark for **sequential supplier selection under contextual uncertainty, bandit feedback and non-stationarity**.

## Research question

Can contextual bandit policies learn supplier-specific procurement economics quickly enough to reduce cumulative regret relative to static purchasing rules, and does explicit forgetting improve adaptation when supplier quality changes?

## Current status

**Feature-complete research benchmark.**

The repository implements:

- seeded synthetic procurement contexts;
- heterogeneous supplier price, quality, lead-time and reliability profiles;
- supplier-specific latent linear reward models;
- operational realized procurement cost and service-failure simulation;
- a truly stationary nominal final block;
- a midpoint supplier-regime-shift block with hidden utility and operational reliability changes;
- a clairvoyant contextual oracle used only for evaluation;
- random and static-supplier baselines;
- linear epsilon-greedy;
- disjoint-arm LinUCB;
- discounted LinUCB for nonstationary adaptation;
- linear Thompson Sampling;
- cumulative pseudo-regret, realized cost, reliability and supplier-share reporting;
- decision-latency measurement;
- paired bootstrap confidence intervals and exact sign tests;
- frozen train/final seed blocks, tests, final report and CI across Python 3.10–3.12.

## Bandit-feedback contract

At epoch `t`, the buyer observes context and supplier-specific action features, selects exactly one supplier, and receives only that supplier's realized feedback.

The environment internally knows all conditional expected rewards so it can calculate oracle pseudo-regret after the decision. Those counterfactual expectations are **evaluation-only** and are never used by policy updates.

## Policies

- `random`;
- `static_best_train`;
- `epsilon_greedy`;
- `linucb`;
- `discounted_linucb`;
- `linear_thompson`;
- contextual clairvoyant oracle for regret accounting only.

Discounted LinUCB exponentially forgets old sufficient statistics while still updating from only the chosen supplier's feedback. This creates an auditable adaptation baseline for the regime-shift experiment.

## Frozen final evaluation

`configs/experiment.json` freezes:

- training seeds `0-19`;
- nominal-final seeds `100-109`;
- supplier-regime-shift seeds `200-209`;
- horizon `400`;
- midpoint shift at 50% of the horizon.

The two final blocks are intentionally separate. Final seeds cannot be used for tuning policy parameters.

Primary metric: **cumulative pseudo-regret** against the contextual oracle.

Secondary metrics:

- pre/post-shift regret;
- mean realized procurement cost;
- service-failure rate;
- supplier-selection proportions;
- mean action-selection latency.

Learned policies are paired by environment seed against `static_best_train` and report a paired 95% bootstrap interval, win rate and exact two-sided sign-test p-value.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m procurement_bandits.experiment
```

## Repository map

```text
src/procurement_bandits/
  environment.py
  policies.py
  benchmark.py
  statistics.py
  experiment.py
tests/
  test_environment.py
  test_policies.py
  test_completion.py
configs/
  experiment.json
docs/
  experimental_protocol.md
  final_report.md
.github/workflows/
  ci.yml
```

## Scientific acceptance rules

1. policies receive only pre-decision context;
2. policy updates use only the chosen action's feedback;
3. oracle counterfactual expectations are evaluation-only;
4. all methods face identical seeded realizations;
5. stationary and regime-shift results remain separate;
6. paired inference is performed by environment seed;
7. regret, procurement cost, service reliability and latency are interpreted jointly;
8. negative/null results are retained.

See `docs/final_report.md` for the complete methodological contract.

## Scope boundary

This repository studies one supplier-selection decision per epoch. Inventory carryover, order quantities, pipeline inventory, delayed rewards and fleet/inventory state transitions are deliberately excluded; those belong in the separate approximate-dynamic-programming project.

## License

MIT
