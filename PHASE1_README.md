# Phase 1: Bayesian Optimization Primer

This branch completes the first learning milestone in `workplan.md`:

- `docs/reading_list.md` — annotated ten-paper reading list.
- `docs/bo_primer.md` — compact guide to BO, Gaussian processes, acquisition,
  recommendation, and stopping.
- `docs/bo_concept_map.md` — a visual map and a worked numerical example.
- `docs/handwritten_note_english.md` — cleaned English notes based on the
  original handwritten study outline.
- `examples/01_bo_intro.ipynb` — visual BoTorch closed-loop example.
- `examples/bo_intro.py` — script version of the same experiment.
- `examples/02_lengthscale_sensitivity.py` — model-misspecification example.
- `tests/test_bo_intro.py` — inexpensive correctness checks.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-phase1.txt
python examples/bo_intro.py
python examples/02_lengthscale_sensitivity.py
pytest -q
```

The toy objective is only evaluated at points selected by the optimizer. Its
dense curve is used for visualization and retrospective scoring, not by BO.

