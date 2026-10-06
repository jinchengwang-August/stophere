# Bayesian Optimization: a Practical Primer

## 1. The problem

Bayesian optimization (BO) searches for an input

$$
x^*\in\arg\max_{x\in\mathcal X} f(x)
$$

when evaluating the unknown objective `f` is expensive. An evaluation may be a
physical experiment, simulation, or model-training run. BO is useful when we
can afford tens or hundreds of evaluations, not millions.

An observation is

$$
y_i=f(x_i)+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma_n^2).
$$

The objective is the real input-output relationship. The surrogate is a model
of that relationship; it is not the objective itself.

## 2. The BO loop

1. Evaluate a small initial design.
2. Fit a probabilistic surrogate to `(x, y)`.
3. Use an acquisition function to score candidate inputs.
4. Evaluate the best-scoring candidate on the real objective.
5. Refit and repeat until a stopping rule fires or the budget is exhausted.
6. Return an input using a separately defined recommendation rule.

These roles must remain separate:

| Object | Question answered |
|---|---|
| Objective `f(x)` | What result does the real experiment produce? |
| GP surrogate | What functions remain plausible after the observations? |
| Acquisition `a(x)` | Which `x` should be evaluated next? |
| Stopping rule | Is another evaluation worth performing? |
| Recommendation | Which `x` should be returned at the end? |

## 3. Gaussian-process surrogate

A Gaussian process (GP) places a probability distribution over functions:

$$
f\sim\mathcal{GP}(m(x),k(x,x')).
$$

Conditioning on observations produces, for any candidate `x`, a posterior mean
and variance:

$$
\mu(x)=k_{xX}(K+\sigma_n^2I)^{-1}y,
$$

$$
\sigma^2(x)=k(x,x)-k_{xX}(K+\sigma_n^2I)^{-1}k_{Xx}.
$$

The mean is the model's current prediction. The variance records uncertainty
under the model assumptions. It is usually small near observations and large
far from them.

### Kernel and length scale

A kernel defines how information transfers between locations. For the RBF
kernel,

$$
k(x,x')=\sigma_f^2\exp\left[-\frac{(x-x')^2}{2\ell^2}\right].
$$

- Small `lengthscale` `ℓ`: the model permits rapid changes and observations
  influence only nearby points.
- Large `ℓ`: the model assumes a smooth function and information travels far.
- `outputscale` `σ_f²`: vertical variation the model expects.
- `noise` `σ_n²`: variation attributed to measurement error rather than the
  latent function.

In multiple dimensions, ARD uses one length scale per coordinate. A fitted
large length scale suggests that coordinate changes the response slowly, but
this interpretation is conditional on scaling and model correctness.

### How fitting works

GP hyperparameters are commonly fitted by maximizing the log marginal
likelihood:

$$
\log p(y\mid X,\theta)=
-\tfrac12y^T K_\theta^{-1}y
-\tfrac12\log|K_\theta|
-\tfrac n2\log(2\pi).
$$

The first term rewards data fit; the log-determinant penalizes overly flexible
covariance explanations. Numerical optimization selects `θ` (length scales,
output scale, noise, and sometimes mean parameters). A local optimum or a poor
kernel can make the model confidently wrong, so hyperparameters and predictive
coverage should be recorded rather than treated as unquestionable truth.

## 4. Exploration and exploitation

An acquisition function is cheap to evaluate because it uses the surrogate.
It balances high predicted value (exploitation) and high uncertainty
(exploration).

| Strategy | Intuition | Important parameter or risk |
|---|---|---|
| EI / LogEI | Expected amount by which `x` beats the incumbent | Can become tiny numerically; use LogEI |
| PI / LogPI | Probability of beating a threshold | May prefer safe, tiny improvements |
| UCB | `μ(x) + sqrt(β)σ(x)` | `β` controls exploration |
| Thompson sampling | Sample one plausible function and maximize it | Must preserve joint posterior correlation |
| MES | Learn the unknown maximum value | More computation and approximation |

For noiseless maximization, expected improvement over best value `f_best` is

$$
EI(x)=(\mu-f_{best})\Phi(z)+\sigma\phi(z),\quad
z=\frac{\mu-f_{best}}{\sigma}.
$$

The first term rewards predicted improvement; the second rewards uncertainty.

## 5. Recommendation and regret

In the noiseless pilot, a transparent recommendation is the best evaluated
point:

$$
\hat x_t\in\arg\max_{x_i\in D_t} f(x_i).
$$

Simple regret is

$$
r_t=f^*-f(\hat x_t).
$$

The recommendation is acceptable for tolerance `ε` when `r_t≤ε`. In a real
experiment `f*` is unknown; benchmark truth is used only by the evaluator, not
by the optimizer or stopping rule.

## 6. Stopping rules

- **Fixed budget:** reproducible cost control, but no quality claim.
- **Stagnation:** stop after little recent improvement; easily fooled by a
  missed region.
- **Negligible EI:** stop when maximum diagnostic EI is small; inherits GP
  assumptions and numerical sensitivity.
- **Confidence gap:** compare the largest plausible value with a lower bound at
  the recommended point. Pointwise intervals do not automatically give global
  or repeated-check coverage.
- **Posterior acceptable-regret probability:** stop when the joint posterior
  assigns high probability to regret at most `ε`; Monte Carlo error and model
  misspecification must be reported.

No recent improvement is evidence about the sampled trajectory, not proof that
the global optimum has been found. A narrow, remote peak is the canonical
counterexample.

## 7. Tool roles

- **BoTorch:** acquisition functions and Bayesian-optimization building blocks.
- **GPyTorch:** scalable GP models, kernels, likelihoods, and fitting machinery.
- **Ax:** higher-level experiment management built around BoTorch.
- **Optuna:** general hyperparameter optimization with multiple samplers.
- **SciPy optimizers:** deterministic numerical optimization; useful for
  maximizing a fitted acquisition function, not a replacement for uncertainty.

## 8. Project-specific validity checks

1. Keep benchmark truth hidden from the optimizer and stopper.
2. Distinguish field seed, optimizer seed, fitting seed, sampling seed, and
   noise seed.
3. Record fitted length scales, noise, warnings, and failures each iteration.
4. Score early stopping by regret and false-stop rate, not stopping time alone.
5. Pair methods on the same initial designs and candidate catalogues.
6. Treat fitted-GP probability statements as conditional on the model.

## Glossary

**Posterior:** updated distribution after observing data. **Kernel:** covariance
rule connecting inputs. **Hyperparameter:** a model parameter fitted outside the
latent function values. **Incumbent:** current best reference value.
**Acquisition value:** utility of evaluating a candidate next. **Simple regret:**
value lost by the final recommendation. **Calibration:** whether stated
uncertainties match observed frequencies.
