from __future__ import annotations

import networkx as nx
import numpy as np

from llm_topology.topologies.common import HOP_WEIGHT, gpu_node


def shortest_path_hop_matrix(graph: nx.Graph, total_gpus: int) -> np.ndarray:
    """
    Return NxN shortest-hop matrix between GPU nodes.
    """
    if total_gpus <= 0:
        raise ValueError("total_gpus must be positive")

    result = np.full((total_gpus, total_gpus), np.inf, dtype=float)

    for src in range(total_gpus):
        lengths = nx.single_source_dijkstra_path_length(
            graph,
            gpu_node(src),
            weight=HOP_WEIGHT,
        )

        for dst in range(total_gpus):
            result[src, dst] = lengths.get(gpu_node(dst), np.inf)

    return result


def active_hop_values(hop_matrix: np.ndarray, traffic_matrix: np.ndarray) -> np.ndarray:
    """
    Return hop counts only for pairs where traffic_matrix[i, j] > 0.
    """
    if hop_matrix.shape != traffic_matrix.shape:
        raise ValueError(
            f"Shape mismatch: hop_matrix={hop_matrix.shape}, "
            f"traffic_matrix={traffic_matrix.shape}"
        )

    mask = traffic_matrix > 0
    values = hop_matrix[mask]

    if np.any(np.isinf(values)):
        raise ValueError("Some active traffic pairs are unreachable")

    return values


def hop_distribution(values: np.ndarray) -> dict[int, float]:
    """
    Return normalized hop-count distribution.

    Example:
        {1: 0.7, 3: 0.3}
    """
    if values.size == 0:
        return {}

    unique, counts = np.unique(values.astype(int), return_counts=True)
    total = counts.sum()

    return {int(hop): float(count / total) for hop, count in zip(unique, counts)}


def weighted_average_hops(hop_matrix: np.ndarray, traffic_matrix: np.ndarray) -> float:
    """
    Traffic-weighted average hop count.
    """
    if hop_matrix.shape != traffic_matrix.shape:
        raise ValueError(
            f"Shape mismatch: hop_matrix={hop_matrix.shape}, "
            f"traffic_matrix={traffic_matrix.shape}"
        )

    total_traffic = float(traffic_matrix.sum())
    if total_traffic == 0:
        return 0.0

    return float((hop_matrix * traffic_matrix).sum() / total_traffic)


def summarize_active_hops(
    topology: str,
    hop_matrix: np.ndarray,
    traffic_matrix: np.ndarray,
) -> dict[str, float | str]:
    """
    Paper Section 5.2: effective hop-count analysis over active pairs only.
    """
    values = active_hop_values(hop_matrix, traffic_matrix)

    return {
        "topology": topology,
        "active_pairs": int(values.size),
        "weighted_average_hops": weighted_average_hops(hop_matrix, traffic_matrix),
        "min_active_hop": float(values.min()),
        "max_active_hop": float(values.max()),
        "mean_active_hop_unweighted": float(values.mean()),
        "p50_active_hop": float(np.percentile(values, 50)),
        "p90_active_hop": float(np.percentile(values, 90)),
        "p95_active_hop": float(np.percentile(values, 95)),
        "p99_active_hop": float(np.percentile(values, 99)),
    }