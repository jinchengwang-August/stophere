# Phase 3: acquisition and model characterization

Phase 3 studies how acquisition choices and Gaussian-process assumptions affect Bayesian-optimization behavior. This branch builds on the validated Phase 2 benchmark bank; Phase 2 files remain available as dependencies, while Phase 3-specific code, tests, documentation, and results use explicit `phase3` names and paths.

## Phase 3 files

- `experiments/phase3_characterization.py`: reproducible NumPy exact-GP characterization runner.
- `tests/test_phase3_characterization.py`: scientific and software checks for the Phase 3 runner.
- `docs/phase3_characterization.md`: methods, results, limitations, and next steps.
- `results/phase3/characterization/`: raw trajectories, summaries, paired deltas, exploratory stopping diagnostics, and summary plot.
- `results/phase3/mrf_bo_pilot/`: earlier MRF pilot retained as a separate historical protocol.

## Run the prototype

```bash
MPLBACKEND=Agg python -m experiments.phase3_characterization
python -m pytest -q tests/test_phase3_characterization.py
```

## Scope boundary

The current runner compares UCB, EI, PI, kernel families, length scales, exploration weights, matched observation noise, and random search across nine systems. It uses a transparent fixed-hyperparameter NumPy GP.

This prototype does not complete Peter's planned Phase 3. The formal Phase 3 deliverable still requires a shared BoTorch protocol for LogEI, LogPI, UCB, Thompson sampling, and max-value entropy search. The included stopping diagnostics are exploratory measurements, not formal Phase 4 stopping rules.
