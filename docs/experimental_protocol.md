# Experimental Protocol

## Objective

Evaluate whether contextual bandit policies adapt supplier selection to procurement context and a controlled mid-horizon regime shift better than static or random rules.

## Information structure

At each epoch, every policy observes the same action-feature matrix containing demand pressure, urgency, market conditions and supplier attributes. Policies observe only the reward of the supplier they select. They never observe hidden supplier coefficients or counterfactual realized rewards.

The oracle observes hidden expected rewards and is used only to compute pseudo-regret.

## Non-stationarity

The environment changes selected hidden supplier coefficients at a deterministic midpoint. The shift is not announced to the policy. Pre-shift and post-shift regret are retained separately.

## Data separation

Training seeds are used only to select the static supplier baseline and, in future phases, tune policy hyperparameters. Test and stress seeds are frozen and must not be used for tuning.

## Metrics

Primary: cumulative pseudo-regret against the contextual oracle.

Secondary: realized procurement cost, mean reward, service-failure rate, supplier-selection shares, pre-shift regret and post-shift regret.

Pseudo-regret is preferred for learning-quality comparisons because stochastic realized reward noise is shared only through the environment, while realized cost and service metrics preserve operational interpretability.

## Acceptance rules

1. Every action must be a valid supplier index.
2. The oracle must select the maximum current hidden expected reward.
3. Regret must be computed from expected rewards, not hindsight realized noise.
4. Final test and stress seeds must remain outside tuning.
5. Static and random baselines must be retained.
6. Negative/null results must not be removed.
7. A contextual method is not considered superior without both lower regret and acceptable operational cost/service behavior.
