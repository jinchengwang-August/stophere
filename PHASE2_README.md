# Phase 2: benchmark and validation suite

This branch adds a reproducible set of synthetic objective functions for studying Bayesian optimization (BO) behavior in one, two, and three dimensions. All functions use a common maximization convention on normalized inputs in [0, 1]^d.

## What is included

- `src/benchmarks.py`: benchmark definitions, bounds, known optima, and reference values.
- `tests/test_benchmarks.py`: checks for dimensions, bounds, deterministic behavior, finite outputs, and known optima.
- `docs/benchmarks.md`: benchmark catalog and usage notes.
- `examples/phase2_gallery.py`: reproducibly generates 2D contour and 3D slice visualizations.

## Run it

Install the project dependencies from `requirements-phase1.txt`, then from the repository root run:

```bash
python -m pytest tests/test_benchmarks.py
python -m examples.phase2_gallery
```

The gallery script writes images under `results/phase2/gallery/`.

## Scope and next step

Phase 2 establishes test landscapes; it does not yet claim that one BO setup or stopping rule is reliable. The next experimental phase should compare acquisition strategies and stopping rules across these landscapes, dimensions, noise levels, and hyperparameter settings, using repeated seeds and summary metrics such as simple regret, evaluation count, and false-stop rate.

Everything on this branch is additive; `main` is unchanged by the Phase 2 work.
