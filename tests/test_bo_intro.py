import importlib.util
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).parents[1] / "examples" / "bo_intro.py"
SPEC = importlib.util.spec_from_file_location("bo_intro", MODULE_PATH)
bo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bo)


def test_objective_shape_and_repeatability():
    x = torch.tensor([[0.1], [0.5], [0.9]], dtype=torch.double)
    y1, y2 = bo.objective(x), bo.objective(x)
    assert y1.shape == (3, 1)
    assert torch.equal(y1, y2)


def test_bo_adds_exactly_one_observation_per_step():
    x, y, _, trace = bo.run_bo(seed=17, n_steps=2)
    assert x.shape == y.shape == (5, 1)
    assert len(trace) == 2
    assert torch.all((x >= 0) & (x <= 1))

