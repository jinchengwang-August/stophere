# Bayesian Optimization Stopping Study

This project studies when Bayesian optimization can stop while still returning an acceptably good solution. The repository is organized around the project phases.

## Start here

- [Workplan](workplan.md) — project question, scope, milestones, and planned experiments.
- [Phase 1 overview](PHASE1_README.md) — core concepts, reading list, worked BO example, results, and presentation.
- [Phase 2 overview](PHASE2_README.md) — benchmark bank, checks, and landscape gallery.
- [Documentation index](docs/README.md) — all written materials grouped by phase.

## Code map

- `examples/bo_intro.py` and `examples/01_bo_intro.ipynb`: Phase 1 one-dimensional BO loop.
- `examples/02_lengthscale_sensitivity.py`: Phase 1 GP misspecification example.
- `src/benchmarks.py`: Phase 2 continuous target functions plus the separately reported Ising MRF.
- `tests/`: correctness checks for the BO example and benchmark bank.
- `results/phase1/`: saved Phase 1 trace and numerical summary.
- `results/phase2/gallery/`: generated benchmark plots (recreate using the Phase 2 README).

## Quick start

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-phase1.txt
python -m pytest -q
python -m examples.phase2_gallery
~~~

The gallery visualizes test-function landscapes; it is not an optimization result. This branch ends at Phase 2. Acquisition-policy and stopping-rule experiments belong on later phase branches.
