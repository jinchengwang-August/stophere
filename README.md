# Bayesian Optimization Stopping Study

This project studies when Bayesian optimization can stop while still returning an acceptably good solution. The repository is organized around the project phases.

## Start here

- [Workplan](workplan.md) — project question, scope, milestones, and planned experiments.
- [Phase 1 overview](PHASE1_README.md) — core concepts, reading list, worked BO example, results, and presentation.
- [Phase 2 overview](PHASE2_README.md) — benchmark bank, checks, and landscape gallery.
- [Phase 3 characterization report](docs/phase3_characterization.md) — controlled pilot of GP, acquisition, hyperparameter, and stopping-rule sensitivity.
- [Documentation index](docs/README.md) — all written materials grouped by phase.

## Code map

- `examples/bo_intro.py` and `examples/01_bo_intro.ipynb`: Phase 1 one-dimensional BO loop.
- `examples/02_lengthscale_sensitivity.py`: Phase 1 GP misspecification example.
- `src/benchmarks.py`: Phase 2 target functions, all maximization on normalized inputs.
- `experiments/characterization_matrix.py`: reproducible NumPy exact-GP screening experiment over 9 systems and 12 configurations.
- `tests/`: correctness checks for the BO example and benchmark bank.
- `results/phase1/`: saved Phase 1 trace and numerical summary.
- `results/phase2/gallery/`: generated benchmark plots (recreate using the Phase 2 README).
- `results/phase3/characterization_matrix/`: per-step traces, per-system summaries, paired deltas, stop-rule outcomes, and comparison plot; see the report for assumptions and limitations.

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase1.txt
python -m pytest -q
python -m examples.phase2_gallery
~~~

The gallery visualizes test-function landscapes; it is not an optimization result. The Phase 3 matrix is a small fixed-hyperparameter NumPy screen, not the final BoTorch study. It maps how optimizer behavior changes across conditions; it does not identify a universally best method.
