# Phase 1 Annotated Reading List

Recommended order: 1 → 6 → 3 → 7, then acquisition papers 4–5, then stopping
papers 8–10. The objective is implementation-level understanding; proofs can be
revisited when a method is implemented.

## 1. Frazier (2018), *A Tutorial on Bayesian Optimization*

- **Problem:** introduces sequential optimization of expensive black boxes.
- **Contribution:** connects GP regression, acquisition functions, and the BO
  loop in one readable treatment.
- **Assumptions:** relatively small evaluation budgets and a surrogate capable
  of representing the objective.
- **Key idea:** expected improvement combines predicted improvement with
  uncertainty; neither posterior mean nor variance alone is sufficient.
- **Use here:** conceptual starting point for `bo_primer.md` and the toy loop.
- **Limitation/question:** idealized GP assumptions do not guarantee calibrated
  uncertainty after hyperparameter fitting.
- [Paper](https://arxiv.org/abs/1807.02811)

## 2. Shahriari et al. (2016), *Taking the Human Out of the Loop*

- **Problem:** unifies a broad BO literature and application vocabulary.
- **Contribution:** survey of models, acquisition functions, and extensions.
- **Key figure:** the model–acquisition–observation feedback loop.
- **Use here:** terminology and failure-mode checklist.
- **Limitation:** broad survey; software APIs and some practice have evolved.
- [Paper](https://ieeexplore.ieee.org/document/7352306)

## 3. Jones, Schonlau & Welch (1998), *Efficient Global Optimization*

- **Problem:** global optimization when each deterministic evaluation is costly.
- **Contribution:** classical Efficient Global Optimization algorithm using EI.
- **Key equation:** `EI=(μ-f_best)Φ(z)+σφ(z)` for maximization.
- **Use here:** defines the baseline acquisition and incumbent logic.
- **Limitation:** deterministic setting and classical modeling choices.
- [Paper](https://doi.org/10.1023/A:1008306431147)

## 4. Srinivas et al. (2010), *GP Optimization in the Bandit Setting*

- **Problem:** balance cumulative reward and exploration with theoretical
  guarantees.
- **Contribution:** GP-UCB and regret bounds using information gain.
- **Key rule:** select `argmax μ_t(x)+sqrt(β_t)σ_t(x)`.
- **Use here:** motivates UCB and shows that exploration weight is consequential.
- **Limitation:** guarantees require assumptions that fitted real-world GPs may
  not satisfy; cumulative and simple regret are different targets.
- [Paper](https://arxiv.org/abs/0912.3995)

## 5. Wang & Jegelka (2017), *Max-value Entropy Search*

- **Problem:** choose evaluations that directly reduce uncertainty about the
  optimum.
- **Contribution:** targets information about the unknown maximum value rather
  than improvement at a single incumbent.
- **Key idea:** maximize mutual information between a new observation and `f*`.
- **Use here:** fifth acquisition family and a contrast with EI/UCB.
- **Limitation:** additional sampling and approximation choices affect cost.
- [Paper](https://proceedings.mlr.press/v70/wang17e.html)

## 6. Balandat et al. (2020), *BoTorch*

- **Problem:** make modern, composable Monte Carlo BO practical in PyTorch.
- **Contribution:** differentiable acquisition computation and modular models.
- **Key algorithm:** reparameterized Monte Carlo samples permit gradient-based
  acquisition optimization.
- **Use here:** implementation framework and reproducibility conventions.
- **Limitation:** a flexible library does not decide scientifically appropriate
  priors, stopping thresholds, or evaluation metrics.
- [Paper](https://arxiv.org/abs/1910.06403)

## 7. Ament et al. (2023), *Unexpected Improvements to Expected Improvement*

- **Problem:** ordinary EI can underflow and produce unusable gradients.
- **Contribution:** numerically stable logarithmic EI variants.
- **Key idea:** optimize `log(EI)` without changing the ideal maximizer.
- **Use here:** use `LogExpectedImprovement` in the example and pilot.
- **Limitation:** numerical stability fixes computation, not surrogate
  misspecification.
- [Paper](https://arxiv.org/abs/2310.20708)

## 8. Makarova et al. (2022), *Automatic Termination for HPO*

- **Problem:** decide when further hyperparameter trials are not worth their
  cost.
- **Contribution:** probabilistic bounds on remaining optimization potential,
  with statistical-error control.
- **Use here:** motivates probability-aware stopping and explicit error budgets.
- **Limitation:** transfer from HPO assumptions to controlled function studies
  must be stated rather than assumed.
- [Paper](https://proceedings.mlr.press/v188/makarova22a.html)

## 9. Ishibashi et al. (2023), *A Stopping Criterion for BO*

- **Problem:** stop using the gap between successive expected minimum simple
  regrets.
- **Contribution:** a regret-oriented alternative to raw incumbent stagnation.
- **Use here:** clarifies that stopping should target decision quality.
- **Limitation:** posterior quantities inherit model assumptions; implementation
  approximation must be measured.
- [Paper](https://proceedings.mlr.press/v206/ishibashi23a.html)

## 10. Wilson (2024), *Stopping BO with Probabilistic Regret Bounds*

- **Problem:** stop when the recommendation is probably within a chosen
  tolerance of the optimum.
- **Contribution:** frames stopping through probabilistic regret guarantees.
- **Key condition:** stop when a posterior statement such as
  `P(r_t ≤ ε | D_t) ≥ 1-δ` is satisfied.
- **Use here:** direct conceptual basis for S5 and for reporting false stops.
- **Limitation:** posterior confidence is not frequentist truth when the model is
  misspecified; repeated checks and Monte Carlo error require care.
- [Paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/b204de7078301292a8876a762eed3dcb-Abstract-Conference.html)

## Two papers to study first

1. **Frazier (2018)** for the whole loop and vocabulary.
2. **Wilson (2024)** for the project's distinguishing question: when a BO
   recommendation is good enough to stop.

Read BoTorch (2020) alongside implementation, and LogEI (2023) before trusting
small acquisition values as stopping evidence.

