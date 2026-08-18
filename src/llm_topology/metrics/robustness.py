from __future__ import annotations

import numpy as np


def _sorted_loads_desc(loads: dict[tuple[str, str], float]) -> np.ndarray:
    return np.sort(np.array(list(loads.values()), dtype=float))[::-1]


def summarize_critical_link_dependency(
    topology: str,
    loads: dict[tuple[str, str], float],
) -> dict[str, float | str]:
    """
    Section 5.5.3: how much of the routed traffic depends on a small number
    of the busiest links. Lower is better (traffic is less concentrated).
    """
    values = _sorted_loads_desc(loads)
    total_routed_load = float(values.sum())

    if total_routed_load <= 0:
        raise ValueError(
            f"total_routed_load is zero for topology {topology!r}; "
            "cannot compute critical-link dependency"
        )

    link_count = values.size
    # Always include at least one link, even when 5%/10% of the link count
    # rounds down to zero on a small topology.
    top_5_count = max(1, int(np.ceil(link_count * 0.05)))
    top_10_count = max(1, int(np.ceil(link_count * 0.10)))

    return {
        "topology": topology,
        "total_routed_load": total_routed_load,
        "top_1_link_dependency": float(values[0] / total_routed_load),
        "top_5_percent_link_dependency": float(values[:top_5_count].sum() / total_routed_load),
        "top_10_percent_link_dependency": float(values[:top_10_count].sum() / total_routed_load),
    }


def summarize_load_imbalance(
    topology: str,
    loads: dict[tuple[str, str], float],
) -> dict[str, float | str]:
    """
    Section 5.5.4: how evenly load is spread across links. Lower is better.
    """
    values = np.array(list(loads.values()), dtype=float)
    mean_load = float(values.mean())

    if mean_load <= 0:
        raise ValueError(
            f"mean link load is zero for topology {topology!r}; "
            "cannot compute load imbalance"
        )

    return {
        "topology": topology,
        "load_imbalance_coefficient": float(values.std() / mean_load),
        "max_to_mean_load_ratio": float(values.max() / mean_load),
        "p95_to_mean_load_ratio": float(np.percentile(values, 95) / mean_load),
        "p99_to_mean_load_ratio": float(np.percentile(values, 99) / mean_load),
    }
