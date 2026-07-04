from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .link_load import path_edges


def compute_edge_utilization(
    loads: dict[tuple[str, str], float],
    capacity: float,
) -> dict[tuple[str, str], float]:
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity!r}")

    utilization: dict[tuple[str, str], float] = {}
    for edge, load in loads.items():
        rho = load / capacity
        if rho >= 1:
            raise ValueError(
                f"Edge {edge} utilization {rho:.4f} >= 1; increase capacity"
            )
        utilization[edge] = rho

    return utilization


def path_congestion_factor(
    path: list[str],
    utilization: dict[tuple[str, str], float],
) -> float:
    """
    M/M/1-style congestion factor: sum of 1 / (1 - rho_e) over path edges.
    """
    factor = 0.0
    for edge in path_edges(path):
        factor += 1.0 / (1.0 - utilization[edge])
    return factor


def compute_pair_latencies(
    traffic_matrix: np.ndarray,
    path_provider: Callable[[int, int], list[list[str]]],
    loads: dict[tuple[str, str], float],
    *,
    capacity: float,
    bandwidth_bytes_per_sec: float = 50e9,
    traffic_unit_bytes: float = 1_000_000,
) -> pd.DataFrame:
    utilization = compute_edge_utilization(loads, capacity)
    total_gpus = traffic_matrix.shape[0]

    rows = []

    for src_gpu in range(total_gpus):
        for dst_gpu in range(total_gpus):
            traffic = traffic_matrix[src_gpu, dst_gpu]
            if traffic <= 0:
                continue

            paths = path_provider(src_gpu, dst_gpu)
            factors = [path_congestion_factor(path, utilization) for path in paths]
            avg_congestion_factor = sum(factors) / len(factors)

            data_bytes = traffic * traffic_unit_bytes
            latency_sec = (data_bytes / bandwidth_bytes_per_sec) * avg_congestion_factor
            latency_ms = latency_sec * 1000

            rows.append(
                {
                    "src_gpu": src_gpu,
                    "dst_gpu": dst_gpu,
                    "traffic": float(traffic),
                    "latency_ms": latency_ms,
                    "path_count": len(paths),
                }
            )

    return pd.DataFrame(
        rows,
        columns=["src_gpu", "dst_gpu", "traffic", "latency_ms", "path_count"],
    )


def summarize_latency(
    topology: str,
    latency_df: pd.DataFrame,
) -> dict[str, float | str]:
    values = latency_df["latency_ms"].to_numpy(dtype=float)

    return {
        "topology": topology,
        "active_pairs": int(values.size),
        "mean_latency_ms": float(values.mean()),
        "p50_latency_ms": float(np.percentile(values, 50)),
        "p90_latency_ms": float(np.percentile(values, 90)),
        "p95_latency_ms": float(np.percentile(values, 95)),
        "p97_latency_ms": float(np.percentile(values, 97)),
        "p99_latency_ms": float(np.percentile(values, 99)),
        "p100_latency_ms": float(np.percentile(values, 100)),
    }
