from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_traffic_matrix(path: str | Path) -> np.ndarray:
    """
    Load a GPU x GPU traffic matrix from .npy, .csv, or plain-numeric .txt.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix == ".csv":
        matrix = pd.read_csv(path, header=None).to_numpy()
    elif suffix == ".txt":
        matrix = np.loadtxt(path)
    else:
        raise ValueError(f"Unsupported traffic matrix file type {suffix!r} for {path}")

    matrix = np.asarray(matrix, dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Traffic matrix must be square, got shape {matrix.shape}")

    np.fill_diagonal(matrix, 0.0)
    return matrix


def save_traffic_matrix(matrix: np.ndarray, path: str | Path) -> Path:
    """
    Save a traffic matrix to .npy, .csv, or .txt based on the path suffix.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        np.save(path, matrix)
    elif suffix == ".csv":
        np.savetxt(path, matrix, delimiter=",")
    elif suffix == ".txt":
        np.savetxt(path, matrix)
    else:
        raise ValueError(f"Unsupported traffic matrix file type {suffix!r} for {path}")

    return path
