import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from experiments.characterization_matrix import (
    BUDGET,
    SYSTEMS,
    catalogue,
    configuration_rows,
    kernel_matrix,
    run_trajectory,
    stopping_summaries,
)


def test_characterization_system_bank_covers_continuous_and_discrete_cases():
    names = {system.name for system in SYSTEMS}
    assert len(names) == len(SYSTEMS)
    assert "ising_mrf_3x3" in names
    assert any(system.dimension == 3 for system in SYSTEMS)
    assert all(catalogue(system).shape[1] == system.dimension for system in SYSTEMS)


def test_one_factor_configuration_list_has_no_duplicates():
    configs = configuration_rows(SYSTEMS[0])
    assert len({config[0] for config in configs}) == len(configs)
    # The baseline is represented once instead of repeated for every factor.
    assert sum(config[1:] == ("rbf", 1.0, "ucb", 2.0, 0.0) for config in configs) == 1


def test_kernels_are_symmetric_positive_semidefinite_on_simple_inputs():
    x = np.array([[0.0], [0.3], [1.0]])
    for kind in ("rbf", "matern32", "matern52"):
        matrix = kernel_matrix(x, x, kind, 0.25)
        assert np.allclose(matrix, matrix.T)
        assert np.linalg.eigvalsh(matrix).min() > -1e-8


def test_small_trajectory_and_stopping_rules_are_bounded():
    system = next(s for s in SYSTEMS if s.name == "smooth_2d")
    config = configuration_rows(system)[0]
    trace = run_trajectory(system, config, seed=0)
    assert len(trace) == BUDGET - 4  # 5 initial points are recorded at n=5..24
    result = stopping_summaries(trace)
    assert len(result) == 4
    assert all(5 <= row["stop_at_evals"] <= BUDGET for row in result)
