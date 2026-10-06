# Phase 1 examples

`01_bo_intro.ipynb` is the main Task 1.3 deliverable. It shows the true toy
function, observed data, GP posterior, LogEI acquisition, selected point, and
best value over iterations.

The two scripts make the notebook logic easy to test and rerun:

```bash
python examples/bo_intro.py
python examples/02_lengthscale_sensitivity.py
pytest -q
```

The second example is intentionally diagnostic: it demonstrates that posterior
uncertainty depends on the assumed kernel and length scale. A smooth-looking,
low-uncertainty posterior is not automatically correct.

