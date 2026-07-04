from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(
    matrix: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "Heatmap",
    xlabel: str = "Destination GPU",
    ylabel: str = "Source GPU",
    colorbar_label: str = "Value",
    show_values: bool = False,
) -> Path:
    """
    Plot and save a matrix heatmap.

    Parameters
    ----------
    matrix:
        2D numpy array to visualize.
    output_path:
        Path where the image will be saved.
    title:
        Figure title.
    xlabel:
        Label for x-axis.
    ylabel:
        Label for y-axis.
    colorbar_label:
        Label for the colorbar.
    show_values:
        If True, writes cell values inside the heatmap.
        Recommended only for small matrices like 32x32.

    Returns
    -------
    Path
        The saved image path.
    """
    data = np.asarray(matrix)

    if data.ndim != 2:
        raise ValueError(f"matrix must be 2D, got shape={data.shape}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    image = ax.imshow(data, interpolation="nearest", aspect="equal")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)

    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))

    if data.shape[0] > 32 or data.shape[1] > 32:
        ax.set_xticks(np.arange(0, data.shape[1], max(1, data.shape[1] // 16)))
        ax.set_yticks(np.arange(0, data.shape[0], max(1, data.shape[0] // 16)))

    if show_values:
        if data.shape[0] > 32 or data.shape[1] > 32:
            raise ValueError("show_values=True is only allowed for matrices up to 32x32")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{data[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return output_path


def plot_hop_heatmap(
    hop_matrix: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "GPU-to-GPU Hop Distance Heatmap",
    show_values: bool = False,
) -> Path:
    """
    Convenience wrapper for hop-distance matrices.
    """
    return plot_heatmap(
        hop_matrix,
        output_path,
        title=title,
        xlabel="Destination GPU",
        ylabel="Source GPU",
        colorbar_label="Hop count",
        show_values=show_values,
    )


def plot_traffic_heatmap(
    traffic_matrix: np.ndarray,
    output_path: str | Path,
    *,
    title: str = "GPU-to-GPU Traffic Matrix",
    show_values: bool = False,
) -> Path:
    """
    Convenience wrapper for traffic matrices.
    """
    return plot_heatmap(
        traffic_matrix,
        output_path,
        title=title,
        xlabel="Destination GPU",
        ylabel="Source GPU",
        colorbar_label="Traffic volume",
        show_values=show_values,
    )