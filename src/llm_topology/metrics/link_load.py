from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def canonical_edge(u: str, v: str) -> tuple[str, str]:
    """
    Deterministic undirected edge key, so a->b and b->a accumulate load
    on the same entry.
    """
    return tuple(sorted((u, v)))


def path_edges(path: list[str]) -> list[tuple[str, str]]:
    return [canonical_edge(path[i], path[i + 1]) for i in range(len(path) - 1)]


def add_path_load(
    loads: dict[tuple[str, str], float],
    path: list[str],
    amount: float,
) -> None:
    for edge in path_edges(path):
        loads[edge] = loads.get(edge, 0.0) + amount


def compute_ecmp_link_loads(
    traffic_matrix: np.ndarray,
    path_provider: Callable[[int, int], list[list[str]]],
) -> dict[tuple[str, str], float]:
    """
    Route every active traffic pair over its equal-cost paths, splitting
    the traffic volume evenly across paths, and accumulate load per edge.
    """
    loads: dict[tuple[str, str], float] = {}
    total_gpus = traffic_matrix.shape[0]

    for src_gpu in range(total_gpus):
        for dst_gpu in range(total_gpus):
            traffic = traffic_matrix[src_gpu, dst_gpu]
            if traffic <= 0:
                continue

            paths = path_provider(src_gpu, dst_gpu)
            if not paths:
                raise ValueError(f"No path found for active pair ({src_gpu}, {dst_gpu})")

            split = traffic / len(paths)
            for path in paths:
                add_path_load(loads, path, split)

    return loads


def link_loads_to_dataframe(
    topology: str,
    loads: dict[tuple[str, str], float],
) -> pd.DataFrame:
    rows = [
        {"topology": topology, "edge_u": u, "edge_v": v, "load": load}
        for (u, v), load in loads.items()
    ]
    return pd.DataFrame(rows, columns=["topology", "edge_u", "edge_v", "load"])


def summarize_link_loads(
    topology: str,
    loads: dict[tuple[str, str], float],
) -> dict[str, float | str]:
    values = np.array(list(loads.values()), dtype=float)

    return {
        "topology": topology,
        "link_count": int(values.size),
        "min_load": float(values.min()),
        "mean_load": float(values.mean()),
        "p50_load": float(np.percentile(values, 50)),
        "p90_load": float(np.percentile(values, 90)),
        "p95_load": float(np.percentile(values, 95)),
        "p99_load": float(np.percentile(values, 99)),
        "max_load": float(values.max()),
    }


def add_utilization(df: pd.DataFrame, capacity: float) -> pd.DataFrame:
    if capacity <= 0:
        raise ValueError(f"capacity must be positive, got {capacity!r}")

    df = df.copy()
    df["capacity"] = capacity
    df["utilization"] = df["load"] / capacity
    return df


def summarize_utilization(
    topology: str,
    utilization_values: np.ndarray,
) -> dict[str, float | str]:
    values = np.asarray(utilization_values, dtype=float)

    return {
        "topology": topology,
        "min_utilization": float(values.min()),
        "mean_utilization": float(values.mean()),
        "p50_utilization": float(np.percentile(values, 50)),
        "p90_utilization": float(np.percentile(values, 90)),
        "p95_utilization": float(np.percentile(values, 95)),
        "p99_utilization": float(np.percentile(values, 99)),
        "max_utilization": float(values.max()),
    }
