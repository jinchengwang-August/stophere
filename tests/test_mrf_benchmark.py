import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from src.benchmarks import get_benchmark


def test_mrf_state_space_and_exact_map_reference():
    mrf = get_benchmark("mrf_ising_3x3", 9)
    states = np.asarray(list(product((0.0, 1.0), repeat=9)))
    values = mrf(states)
    assert mrf.discrete
    assert len(states) == 512
    assert np.max(values) == pytest.approx(mrf.reference_value)
    assert mrf(mrf.reference_x) == pytest.approx(mrf.reference_value)


def test_mrf_is_deterministic_and_supports_batches():
    mrf = get_benchmark("mrf_ising_3x3", 9)
    points = np.asarray([[0.0] * 9, [1.0] * 9, mrf.reference_x])
    np.testing.assert_allclose(mrf(points), mrf(points))
    assert mrf(points).shape == (3,)


def test_mrf_rejects_fractional_and_wrong_dimension_states():
    mrf = get_benchmark("mrf_ising_3x3", 9)
    with pytest.raises(ValueError, match="only 0 or 1"):
        mrf(np.full(9, 0.5))
    with pytest.raises(ValueError, match="nine binary variables"):
        get_benchmark("mrf_ising_3x3", 3)

