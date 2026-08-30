# Contextual Bandits for Dynamic Procurement

Research-oriented Industrial Engineering / Operations Research benchmark for **sequential supplier selection under contextual uncertainty and non-stationarity**.

## Research question

Can contextual bandit policies learn supplier-specific procurement economics quickly enough to reduce cumulative regret relative to static purchasing rules when demand urgency, market price, supplier reliability and delivery risk vary over time?

## Current status

**Phase 1 implemented: contextual procurement environment + oracle/bandit baselines + frozen regret benchmark.**

The benchmark includes:

- seeded synthetic procurement contexts;
- multiple suppliers with heterogeneous price, quality and lead-time profiles;
- supplier-specific linear latent utility models;
- stochastic realized procurement cost and service penalties;
- a deterministic mid-horizon supplier regime shift;
- a clairvoyant contextual oracle;
- random and static-supplier baselines;
- epsilon-greedy linear value estimation;
- disjoint-arm LinUCB;
- linear Thompson Sampling;
- cumulative regret, realized cost, service-failure and supplier-selection reporting;
- frozen in-distribution and shifted final seed blocks;
- tests and GitHub Actions CI.

## Decision model

At decision epoch `t`, the buyer observes context `z_t` containing demand pressure, spot-market conditions and urgency. For each supplier `a`, the environment builds an action feature vector `x_{t,a}` by combining the common context with supplier attributes.

The buyer chooses exactly one supplier:

```text
A_t in {1, ..., K}
```

The latent conditional expected utility is supplier-specific and linear:

```text
mu_t(a) = x_{t,a}^T theta_a
```

The environment separately simulates operational procurement economics from supplier base price, market conditions, late delivery and defects. Realized bandit reward is the latent supplier utility adjusted by realized cost shocks and zero-mean noise. This separation keeps pseudo-regret auditable while retaining operational cost and service KPIs.

The hidden parameters change for selected suppliers after a fixed regime-shift point, creating a controlled nonstationary test.

The **clairvoyant oracle** knows the current hidden expected utility parameters and chooses the best supplier for each observed context. Learning policies do not have access to these parameters.

## Policies

- `random`: uniform supplier selection;
- `static_best_train`: one supplier chosen from training-period average expected utility;
- `epsilon_greedy`: per-supplier ridge regressions with explicit exploration;
- `linucb`: disjoint-arm upper-confidence linear bandit;
- `linear_thompson`: Gaussian posterior-style linear Thompson Sampling;
- `oracle`: contextual clairvoyant reference, used only for regret computation.

## Evaluation contract

Primary metric: **cumulative pseudo-regret** against the contextual oracle.

Secondary metrics:

- mean realized procurement cost;
- service-failure rate;
- mean reward;
- supplier-selection proportions;
- pre-shift and post-shift regret;
- adaptation after regime shift.

A contextual policy is not promoted simply because it is more complex. If a static supplier or simple epsilon-greedy rule is competitive, that result is retained.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
ruff check src tests
pytest -q
python -m procurement_bandits.experiment
```

## Repository layout

```text
src/procurement_bandits/
  environment.py
  policies.py
  benchmark.py
  experiment.py
tests/
  test_environment.py
  test_policies.py
configs/
  experiment.json
docs/
  experimental_protocol.md
.github/workflows/
  ci.yml
```

## Scope boundary

This repository studies contextual supplier selection with one procurement action per epoch. Multi-item joint ordering, inventory carryover, combinatorial supplier allocation, contracts and full reinforcement learning are separate extensions rather than hidden changes to this benchmark.

## License

MIT
