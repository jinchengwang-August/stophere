import numpy as np
import pytest

from src.benchmarks import FAMILIES, benchmark_suite, get_benchmark

@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("dimension", [2, 3])
def test_batch_shape_and_finite_values(name, dimension):
    objective = get_benchmark(name, dimension, field_seed=13)
    points = np.full((4, dimension), 0.43)
    values = objective(points)
    assert values.shape == (4,)
    assert np.isfinite(values).all()
    assert np.isfinite(objective(points[0]))

def test_suite_has_24_unique_cases():
    suite = benchmark_suite()
    assert len(suite) == 24
    assert len({(item.name, item.dimension) for item in suite}) == 24

@pytest.mark.parametrize("point", [np.array([0.5]), np.array([1.1, 0.2]), np.array([np.nan, 0.2])])
def test_invalid_inputs_are_rejected(point):
    with pytest.raises(ValueError):
        get_benchmark("quadratic_peak", 2)(point)

def test_documented_analytic_optima():
    for name in ("quadratic_peak", "rotated_ellipsoid", "hidden_narrow_peak"):
        objective = get_benchmark(name, 3)
        assert objective(objective.reference_x) == pytest.approx(objective.reference_value)

def test_random_field_is_fixed_and_reproducible():
    x = np.array([[0.11, 0.29], [0.61, 0.87]])
    a = get_benchmark("grf_short", 2, field_seed=8)
    b = get_benchmark("grf_short", 2, field_seed=8)
    c = get_benchmark("grf_short", 2, field_seed=9)
    np.testing.assert_allclose(a(x), b(x))
    assert not np.allclose(a(x), c(x))
    first = a(x[0])
    a(np.array([0.1, 0.3]))
    assert a(x[0]) == first

@pytest.mark.parametrize("name,dimension", [("branin", 2), ("hartmann3", 3)])
def test_standard_anchors_accept_expected_dimension(name, dimension):
    assert np.isfinite(get_benchmark(name, dimension)(np.full(dimension, 0.5)))
