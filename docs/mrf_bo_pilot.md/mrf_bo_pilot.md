# Auxiliary MRF benchmark and BO pilot

## Why add an MRF?

The project succeeds by **characterizing where methods work and fail**, not by inventing a novel optimizer. The core Phase 2 bank contains 24 continuous 2D/3D landscapes. This adds one small, explicitly separate discrete system to test whether conclusions transfer to structured configuration spaces.

A Markov random field (MRF) is a probability model over variables on an undirected graph. Its defining idea is conditional independence: given a node's neighbors, that node is independent of non-neighbors. A pairwise MRF factorizes into node and edge potentials:

`p(s) = (1/Z) ∏ᵢ ψᵢ(sᵢ) ∏_(i,j) ψᵢⱼ(sᵢ,sⱼ)`

For the Ising case, spins are `sᵢ ∈ {-1,+1}`; the score is

`g(s) = Σ_(i,j∈E) Jᵢⱼ sᵢ sⱼ + Σᵢ hᵢ sᵢ`.

Positive coupling `Jᵢⱼ` rewards equal neighboring spins; negative coupling rewards opposite spins; `hᵢ` biases one node. Mixed-sign loops create frustration, so all local preferences cannot necessarily be satisfied at once. Since `p(s) ∝ exp(g(s))`, maximizing `g` gives the MAP configuration. This benchmark is deterministic once its graph, couplings, and fields are fixed. A 3×3 grid has 9 binary variables and only `2⁹ = 512` states, so we can enumerate the exact MAP value for scoring.

BO searches the configuration-to-score function; it is **not** being used to infer the MRF probability model. The generic RBF GP in this pilot sees a flattened 9-bit vector and does not use the MRF graph topology. That is a limitation to characterize, not a claim that every MRF-aware optimizer behaves this way.

## Pilot protocol

Run from repository root with `MPLBACKEND=Agg python -m experiments.mrf_bo_pilot`. It compares finite-catalogue GP-UCB with uniform random search on four systems: smooth 2D quadratic peak, 2D competing peaks, 2D hidden narrow peak, and the 3×3 binary Ising MRF. Each policy uses the same six starting points for a given seed; there are 10 paired seeds and 30 total objective evaluations. GP-UCB uses a fixed RBF kernel (length scale 0.18 for the continuous cases, 1.35 for the MRF vector), `β=2`, standardized observations, and `1e-6` jitter. Observations are noiseless. These are deliberately fixed settings—not fitted or sensitivity-tested hyperparameters.

The evaluator computes the optimum and range of the finite candidate catalogue, but the acquisition only sees queried labels. Normalized regret is `(catalogue_best - best_queried)/(catalogue_best - catalogue_worst)`; success means normalized regret ≤ 0.05 at the 30-evaluation budget. Every run continued to the full budget, so first-hit time is **not** stopping-rule savings.

## Preliminary results

Mean final normalized regret (standard deviation), 10 seeds:

| System | GP-UCB | Random | Success GP-UCB / Random |
|---|---:|---:|---:|
| Quadratic peak | 0.00002 (0.00008) | 0.01935 (0.01199) | 10/10 / 10/10 |
| Competing peaks | 0.02132 (0.03765) | 0.17837 (0.13747) | 8/10 / 1/10 |
| Hidden narrow peak | 0.32991 (0.00000) | 0.28911 (0.17346) | 0/10 / 2/10 |
| Ising MRF | 0.00743 (0.00753) | 0.11497 (0.06782) | 10/10 / 2/10 |

Files in the local pilot output are `results/phase3/mrf_bo_pilot/summary.csv`, `trajectories.csv`, and `regret_curves.png`.

## What this suggests—and does not

In this small pilot, GP-UCB did well on the smooth peak, competing peaks, and this one MRF; the hidden narrow peak exposed a clear failure condition for the fixed surrogate scale. Identical GP-UCB final regret across the 10 hidden-peak seeds suggests the current model/acquisition repeatedly settled on the broad basin rather than discovering the narrow one. This is a useful characterization result, not a universal ranking.

This is a NumPy exact-GP prototype, not the planned BoTorch implementation. Ten seeds, one fixed MRF, no observation noise, one fixed GP configuration, and a finite catalogue are exploratory only. No stopping rule has been tested. Next: reproduce the pilot in BoTorch; vary only length scale on paired runs; then evaluate stopping rules as observers of complete shared trajectories using regret, success, false-stop rate, and actual evaluation savings. Continuous-grid conclusions apply only to that grid; the MRF catalogue is exhaustive.
