# Phase 2: benchmark and validation suite

This branch adds a reproducible set of synthetic objective functions for studying Bayesian optimization (BO) behavior in one, two, and three dimensions. All functions use a common maximization convention on normalized inputs in [0, 1]^d.

## What is included

- `src/benchmarks.py`: benchmark definitions, bounds, known optima, and reference values.
- `tests/test_benchmarks.py`: checks for dimensions, bounds, deterministic behavior, finite outputs, and known optima.
- `docs/benchmarks.md`: benchmark catalog and usage notes.
- `examples/phase2_gallery.py`: reproducibly generates 2D contour and 3D slice visualizations.
- `src/benchmarks.py` also exposes an auxiliary 3x3 binary Ising MRF (`mrf_ising_3x3`), scored separately from the core 24 continuous cases; see `docs/benchmarks.md`.
- `tests/test_mrf_benchmark.py`: validates the discrete state space and exact MAP reference.
- `experiments/characterization_matrix.py`: the first Phase 3 NumPy screen, comparing kernels/acquisitions/hyperparameters and stopping rules across nine systems; this is not yet the planned BoTorch study.

## Run it

Install the project dependencies from `requirements-phase1.txt`, then from the repository root run:

```bash
python -m pytest tests/test_benchmarks.py
python -m examples.phase2_gallery
```

The gallery script writes images under `results/phase2/gallery/`.

## Scope and next step

Phase 2 establishes test landscapes; it does not claim that one BO setup or stopping rule is reliable. The initial Phase 3 screen and its limits are documented in `docs/phase3_characterization.md`. Project success means characterizing where methods work and fail—not inventing a new acquisition function. The screen is descriptive and should be followed by a BoTorch reproduction, sensitivity sweeps for stopping thresholds, and more seeds on the failure cases.

Everything on this branch is additive; `main` is unchanged by the Phase 2 work.
