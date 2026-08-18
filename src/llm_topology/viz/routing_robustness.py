from __future__ import annotations

from pathlib import Path

from .cdf import (
    plot_grouped_metric_bar,
    plot_path_count_cdf,
    plot_path_diversity_bar,
    plot_single_bar,
    plot_single_path_exposure_bar,
)

# Re-exported so all Section 5.5 plotting helpers can be imported from one
# module. The implementations stay in cdf.py (used since the path-diversity
# metrics were added) to avoid duplicating the matplotlib code.
__all__ = [
    "plot_path_diversity_bar",
    "plot_single_path_exposure_bar",
    "plot_path_count_cdf",
    "plot_critical_link_dependency_bar",
    "plot_load_imbalance_bar",
]


def plot_critical_link_dependency_bar(
    values: dict[str, dict[str, float]],
    output_path: str | Path,
    *,
    title: str = "Critical-Link Dependency",
    xlabel: str = "Dependency metric",
    ylabel: str = "Fraction of total routed load",
) -> Path:
    """
    `values` maps topology -> {"top_1": ..., "top_5_percent": ..., "top_10_percent": ...}.
    Lower is better: traffic is less concentrated on a handful of links.
    """
    return plot_grouped_metric_bar(values, output_path, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_load_imbalance_bar(
    values: dict[str, float],
    output_path: str | Path,
    *,
    title: str = "Load Imbalance Coefficient",
    ylabel: str = "Std / mean link load",
) -> Path:
    """
    One bar per topology. Lower is better: load is spread more evenly.
    """
    return plot_single_bar(values, output_path, title=title, ylabel=ylabel)
