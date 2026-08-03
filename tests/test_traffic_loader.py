from __future__ import annotations

import numpy as np

from llm_topology.traffic.matrix_loader import load_traffic_matrix, save_traffic_matrix


def _matrix_with_nonzero_diagonal(n: int = 4) -> np.ndarray:
    matrix = np.arange(n * n, dtype=float).reshape(n, n)
    np.fill_diagonal(matrix, 99.0)
    return matrix


def test_npy_round_trip_zeroes_diagonal(tmp_path):
    matrix = _matrix_with_nonzero_diagonal()
    path = tmp_path / "traffic.npy"

    save_traffic_matrix(matrix, path)
    loaded = load_traffic_matrix(path)

    assert loaded.shape == matrix.shape
    assert np.all(np.diagonal(loaded) == 0)


def test_csv_round_trip_zeroes_diagonal(tmp_path):
    matrix = _matrix_with_nonzero_diagonal()
    path = tmp_path / "traffic.csv"

    save_traffic_matrix(matrix, path)
    loaded = load_traffic_matrix(path)

    assert loaded.shape == matrix.shape
    assert np.all(np.diagonal(loaded) == 0)
