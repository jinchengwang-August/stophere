# Phase 3 — BO and stopping-rule characterization

## Goal and scope

Project success means **reliable characterization**, not novelty or maximum difficulty: identify when BO works, when it fails, and what quality/cost trade-off a stopping rule creates. This is an exploratory, fixed-hyperparameter NumPy exact-GP screen, not a final BoTorch study.

Run locally with MPLBACKEND=Agg python -m experiments.characterization_matrix from the repository root.

## Design

The matrix contains nine finite search systems: smooth 2D, anisotropic 2D, rippled 2D, competing peaks 2D, hidden narrow peak 2D, Rosenbrock 3D, Ackley 3D, rough random field 2D, and a 3x3 binary Ising MRF. The continuous candidates are fixed grids (21 points per 2D axis; 9 per 3D axis); the MRF evaluates all 512 configurations. Each method uses five paired seeds, five common starting points, and a budget of 24 evaluations: 9 systems × 12 configurations × 5 seeds = 540 trajectories.

The single nominal setup (RBF GP-UCB, length scale multiplier 1, β=2) is the baseline. One-factor contrasts test Matérn 3/2 and 5/2 kernels, length scales 0.5× and 2×, UCB β=0.5 and 8, EI and PI, plus random search. One extra Matérn-3/2 ×2 length-scale interaction checks a combined setting. A separate condition adds Gaussian measurement noise with SD 3% of that catalogue's objective range and supplies the GP the matching noise variance. This is exact-GP Python/NumPy code; parameters are fixed, not fitted.

Four stopping rules are replayed on each completed trajectory, including random search; a fixed RBF GP is refit to observed points for model-based stop diagnostics: (1) full 24-evaluation budget; (2) patience-5, no improvement exceeding 1% of recent observed response SD; (3) stop when maximum catalogue posterior 95% UCB is within 0.05 observed SD of the incumbent; (4) stop when the maximum marginal probability of beating the incumbent by 0.01 observed SD is ≤0.05. Thresholds use only observations/model, not the oracle optimum. The last two are diagnostics, not guarantees. False stop means normalized regret >0.05 at termination.

## Results

Means below average the per-system five-seed means; they are descriptive, not significance tests.

| Configuration | Mean normalized regret | Success ≤0.05 |
|---|---:|---:|
| Nominal RBF/UCB β=2 | 0.1314 | 71.1% |
| Matérn 3/2 | 0.1171 | 71.1% |
| Matérn 5/2 | 0.1015 | 75.6% |
| RBF length scale ×0.5 | 0.1104 | 73.3% |
| RBF length scale ×2 | 0.1536 | 64.4% |
| UCB β=0.5 | 0.1106 | 73.3% |
| UCB β=8 | 0.1209 | 68.9% |
| EI | 0.1179 | 73.3% |
| PI | 0.1298 | 73.3% |
| Matérn 3/2 × length scale 2 | 0.0958 | 77.8% |
| 3% observation noise (matched model) | 0.1507 | 64.4% |
| Random search | 0.2105 | 44.4% |

Per-system nominal regret: smooth 2D 0.000; anisotropic 2D 0.000; competing modes 0.000; rippled 0.038; hidden narrow 0.158; rough field 0.089; Rosenbrock 3D 0.859; Ackley 3D 0.000; MRF 0.039. The Rosenbrock result is a clear low-budget failure case. Some grids are easy enough that methods tie at zero.

| Stopping rule | Mean stopped regret | Success | False stops | Calls saved (of 24) |
|---|---:|---:|---:|---:|
| Fixed budget | 0.129 | 69.3% | 0.0% | 0.00 |
| Patience-5 | 0.165 | 56.1% | 43.9% | 10.57 |
| Global UCB gap | 0.142 | 66.1% | 33.9% | 3.84 |
| Max marginal P(improve) | 0.145 | 65.2% | 34.8% | 4.80 |

Patience saves the most evaluations but false-stops often. Posterior-based rules save less and still false-stop about one third of runs. On the MRF specifically, patience saves 9.9 calls but false-stops 73%; UCB-gap saves 3.1 calls and false-stops 27%. These thresholds need calibration, not adoption as defaults.

## Interpretation and caveats

Across this small pilot, Matérn-5/2, shorter RBF length scale, lower β, and EI did slightly better on average than nominal; twice-longer RBF scale and the noisy condition did worse. But the best setting varies by system. The same-run best configurations are selection-biased and require fresh-seed confirmation. Five seeds, small fixed grids, unfit hyperparameters, one fixed MRF, and matched Gaussian noise are not enough for broad claims. A generic Euclidean GP on the MRF encoding also ignores graph structure.

Next: reproduce selected cases in BoTorch; spend more seeds/budget on Rosenbrock, hidden-narrow, rough-field, and MRF; sweep stopping thresholds to draw regret-versus-savings/false-stop frontiers; test noise misspecification and fitted GP hyperparameters. Use the CSVs for every per-system/configuration value rather than pooled averages.
