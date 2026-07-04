from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_grouped_bar(
    data: dict[str, dict[int, float]],
    output_path: str | Path,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    """
    Plot a grouped bar chart comparing per-hop-count distributions across
    multiple topologies.

    `data` maps topology name -> {hop_count: percentage}. Missing hop counts
    for a topology are treated as 0.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topologies = list(data.keys())
    all_hops = sorted({hop for distribution in data.values() for hop in distribution})

    fig, ax = plt.subplots(figsize=(8, 5))

    num_topologies = len(topologies)
    bar_width = 0.8 / max(1, num_topologies)
    x = np.arange(len(all_hops))

    for i, topology in enumerate(topologies):
        distribution = data[topology]
        values = [distribution.get(hop, 0.0) for hop in all_hops]
        offset = (i - (num_topologies - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width, label=topology)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(hop) for hop in all_hops])
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_single_bar(
    values: dict[str, float],
    output_path: str | Path,
    *,
    title: str,
    ylabel: str,
) -> Path:
    """
    Plot a single bar chart comparing one scalar value per topology.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    topologies = list(values.keys())
    scores = [values[name] for name in topologies]

    fig, ax = plt.subplots(figsize=(6, 5))

    x = np.arange(len(topologies))
    ax.bar(x, scores, width=0.5)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(topologies)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path
