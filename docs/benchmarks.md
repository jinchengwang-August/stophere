# Phase 2 benchmark bank

## Why these functions

Phase 2 builds a controlled set of landscapes for the project question: do Bayesian optimization stopping rules remain dependable when the objective changes, or do they stop too early on specific shapes? This phase constructs and validates the test problems; comparison of acquisition and stopping policies is later work.

The core continuous bank has 24 cases: eleven landscape families in both 2D and 3D, plus Branin (2D) and Hartmann-3 (3D). Every core objective is a noise-free maximization function on the unit cube [0, 1]^d. Measurement noise should be added by a separate observation wrapper, never by changing the latent function. A separate 3x3 binary Ising Markov random field (MRF) is included as an **auxiliary discrete system**, not counted among the 24 core continuous cases.

| Name | Main property tested | Best-known value |
| --- | --- | --- |
| quadratic_peak | Smooth broad unimodal peak | 1, analytic |
| rotated_ellipsoid | Coupled coordinates and strong anisotropy | 1, analytic |
| rippled_peak | Broad trend with local oscillations | Numerical reference |
| competing_gaussian_peaks | Several near-tied separated modes | 1, analytic |
| hidden_narrow_peak | Attractive broad basin plus a better narrow remote peak | 1, analytic |
| rosenbrock | Narrow curved valley | 0 after negation/scaling |
| ackley | Flat outer area and structure near the center | 0 after negation |
| rastrigin | Repeated competing local optima | 0 after negation/scaling |
| grf_long | Fixed, broad smooth random field | Numerical reference |
| grf_short | Fixed field with short spatial scale | Numerical reference |
| grf_rough | Lower-smoothness Matérn-like field | Numerical reference |
| branin | Standard 2D multimodal anchor | -0.397887, analytic |
| hartmann3 | Standard 3D multimodal anchor | Numerical reference |

## Auxiliary structured-discrete system: Ising MRF

`get_benchmark("mrf_ising_3x3", 9)` creates a fixed 3x3 binary pairwise MRF. Its nine variables are spins `s_i ∈ {-1,+1}` (encoded as inputs `x_i ∈ {0,1}`), and its log-potential score is

```text
score(s) = Σ_(i,j) J_ij s_i s_j + Σ_i h_i s_i.
```

The pair terms encode graph edges; mixed-sign couplings include frustrated loops, and small node fields break some symmetries. The corresponding probability model is proportional to `exp(score(s))`; its MAP state is therefore the maximizer of this score. This benchmark is deterministic once constructed. All `2^9 = 512` configurations are enumerable, so the MAP reference is exact and regret can be scored without giving unqueried labels to the optimizer.

An MRF is a probability model with conditional-independence structure: a node is independent of non-neighbors once its neighbors are known. BO is not being used to infer the MRF distribution here; it is used to search the expensive configuration-to-score function. This is a discrete, structured extension and should be reported separately from the 2D/3D continuous cases. In particular, an ordinary RBF kernel on the binary vector does not automatically exploit the MRF graph topology.

The MRF checks live in `tests/test_mrf_benchmark.py`; the broader GP/stopping-rule screen is documented in `docs/phase3_characterization.md` and implemented by `experiments/characterization_matrix.py`.

## API and design decisions

Use get_benchmark(name, dimension, field_seed=..., n_features=...) or benchmark_suite(). A Benchmark accepts one point of shape (d,) or a batch (n, d), and provides its name, dimension, description, and any known optimum. Invalid shapes, non-finite inputs, and points outside [0, 1]^d raise ValueError. All functions are deterministic after construction.

Native domains are mapped into the unit cube: Rosenbrock uses z = 4x - 2; Ackley uses z = 10x - 5; Rastrigin uses z = 10.24x - 5.12; Branin uses u = 15x1 - 5 and v = 15x2. The minimization functions are negated; Rosenbrock and Rastrigin additionally use -L/(1+L), placing their best value at 0. Comparable raw objective values do not imply comparable difficulty, so scoring should use each function's declared tolerance and report simple regret.

## Random fields

The grf cases use a finite random Fourier approximation with 256 spectral components by default. Frequencies follow a Student-t representation associated with a Matérn spectrum; the length scale and smoothness change by family. Frequencies, phases, and weights are sampled once at construction. The field seed names the landscape; optimizer seeds must be separate. The output is bounded in [-1, 1] by dividing by the sum of absolute weights. This is a Matérn-like finite approximation, not an exact Matérn sample path. Resampling the field at each query would incorrectly turn the objective into noise.

## Gallery and validation

From the repository root, run python -m examples.phase2_gallery. It saves a 2D heatmap gallery and 3D slices with the third coordinate fixed at 0.5 under results/phase2/gallery. These plots explain geometry; BO policies must only receive the queried points and values.

Run pytest -q. Tests cover all 24 dimensions, scalar and batch evaluation, invalid inputs, known optima, unique registry entries, and fixed-field reproducibility. The gallery is generated from a script, not manually edited.

Random-field and other numerical optima are references, not proofs of the continuous-domain maximum. Use a dense or scrambled Sobol catalogue plus independent searches to construct and save numerical references. State whether claims are over that finite catalogue or the continuous domain: a finite catalogue can miss a narrow peak entirely.

## Phase sequence

Phase 2 completes the benchmark bank. Phase 3 adds acquisition policies and a common run loop. Phase 4 implements stopping rules as observers of the same observed trajectory. Phase 5 runs the pilot matrix. Phase 6 evaluates regret, early-stop failures, cost, and GP diagnostics. Testing different benchmark examples and hyperparameters belongs in that later controlled reliability evaluation; vary one factor at a time first, then confirm with multiple optimizer and field seeds.
