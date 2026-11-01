# Phase 2: benchmark and validation suite

This branch adds a reproducible set of synthetic objective functions for studying Bayesian optimization (BO) behavior in one, two, and three dimensions. All functions use a common maximization convention on normalized inputs in [0, 1]^d.

## What is included

- `src/benchmarks.py`: benchmark definitions, bounds, known optima, and reference values.
- `tests/test_benchmarks.py`: checks for dimensions, bounds, deterministic behavior, finite outputs, and known optima.
- `docs/benchmarks.md`: benchmark catalog and usage notes.
- `examples/phase2_gallery.py`: reproducibly generates 2D contour and 3D slice visualizations.
- `src/benchmarks.py`: also exposes `mrf_ising_3x3`, a separately reported 3x3 binary Ising MRF.
- `tests/test_mrf_benchmark.py`: validates the binary state space and exact MAP reference.

## Run it

Install the project dependencies from `requirements-phase1.txt`, then from the repository root run:

```bash
python -m pytest -q tests/test_benchmarks.py tests/test_mrf_benchmark.py
python -m examples.phase2_gallery
```

The gallery script writes images under `results/phase2/gallery/`.

Generated figures on this branch:

- `results/phase2/gallery/benchmark_gallery_2d.png`
- `results/phase2/gallery/benchmark_gallery_3d_slices_x3_025.png`
- `results/phase2/gallery/benchmark_gallery_3d_slices.png`
- `results/phase2/gallery/benchmark_gallery_3d_slices_x3_075.png`

Phase 2 verification record: `python -m pytest -q tests/test_benchmarks.py tests/test_mrf_benchmark.py` — **33 passed**.

## Scope and next step

Phase 2 establishes test landscapes; it does not claim that one BO setup or stopping rule is reliable. The Ising MRF is an auxiliary discrete case, not one of the 24 continuous cases. Phase 3 should compare acquisition strategies across these validated systems under a common protocol.

Everything on this branch is additive; `main` is unchanged by the Phase 2 work.
