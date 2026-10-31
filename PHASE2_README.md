# Phase 2: benchmark and validation suite

This branch adds a reproducible set of synthetic objective functions for studying Bayesian optimization (BO) behavior in one, two, and three dimensions. All functions use a common maximization convention on normalized inputs in [0, 1]^d.

## What is included

- `src/benchmarks.py`: benchmark definitions, bounds, known optima, and reference values.
- `tests/test_benchmarks.py`: checks for dimensions, bounds, deterministic behavior, finite outputs, and known optima.
- `docs/benchmarks.md`: benchmark catalog and usage notes.
- `examples/phase2_gallery.py`: reproducibly generates 2D contour and 3D slice visualizations.
- `src/benchmarks.py` also exposes an auxiliary 3x3 binary Ising MRF (`mrf_ising_3x3`), added as a separate structured-discrete extension and scored separately from the core 24 continuous cases; see `docs/benchmarks.md`.
- `tests/test_mrf_benchmark.py`: validates the discrete state space and exact MAP reference.
- `experiments/characterization_matrix.py`: an exploratory prototype spanning parts of planned Phases 3 and 4. It compares kernels/acquisitions/hyperparameters and stopping proxies across nine systems; it is not the planned BoTorch study and does not complete Phase 3. See `docs/roadmap_and_status.md`.

## Run it

Install the project dependencies from `requirements-phase1.txt`, then from the repository root run:

```bash
python -m pytest tests/test_benchmarks.py
python -m examples.phase2_gallery
```

The gallery script writes images under `results/phase2/gallery/`.

## Scope and next step

Phase 2 establishes test landscapes; the Ising MRF is explicitly an auxiliary discrete case, not one of the 24 continuous cases. Phase 2 does not claim that one BO setup or stopping rule is reliable. The NumPy screen and its limits are documented in `docs/phase3_characterization.md`; it is a useful prototype but does not count as completing Peter's planned Phase 3. The proper next step is a small BoTorch acquisition-policy implementation, followed by the separately specified Phase 4 stopping rules.

Everything on this branch is additive; `main` is unchanged by the Phase 2 work.
