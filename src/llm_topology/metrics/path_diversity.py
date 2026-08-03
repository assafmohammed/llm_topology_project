from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PAIR_COLUMNS = ["src_gpu", "dst_gpu", "traffic", "path_count", "is_single_path"]


@dataclass
class PathDiversityPair:
    src_gpu: int
    dst_gpu: int
    traffic: float
    path_count: int
    is_single_path: bool


def compute_path_diversity_pairs(
    traffic_matrix: np.ndarray,
    path_provider: Callable[[int, int], list[list[str]]],
) -> pd.DataFrame:
    """
    For every active traffic pair (traffic_matrix[i, j] > 0), record how many
    equal-cost paths the topology's path provider offers between i and j.
    """
    total_gpus = traffic_matrix.shape[0]
    pairs: list[PathDiversityPair] = []

    for src_gpu in range(total_gpus):
        for dst_gpu in range(total_gpus):
            traffic = traffic_matrix[src_gpu, dst_gpu]
            if traffic <= 0:
                continue

            paths = path_provider(src_gpu, dst_gpu)
            path_count = len(paths)

            if path_count <= 0:
                raise ValueError(
                    f"path_provider returned no paths for active pair "
                    f"({src_gpu}, {dst_gpu})"
                )

            pairs.append(
                PathDiversityPair(
                    src_gpu=src_gpu,
                    dst_gpu=dst_gpu,
                    traffic=float(traffic),
                    path_count=path_count,
                    is_single_path=(path_count == 1),
                )
            )

    return pd.DataFrame([asdict(pair) for pair in pairs], columns=PAIR_COLUMNS)


def summarize_path_diversity(topology: str, pair_df: pd.DataFrame) -> dict[str, float | str]:
    if pair_df.empty:
        raise ValueError("pair_df is empty; cannot summarize path diversity")

    traffic = pair_df["traffic"].to_numpy(dtype=float)
    path_count = pair_df["path_count"].to_numpy(dtype=float)
    single_path_mask = pair_df["path_count"].to_numpy() == 1

    total_active_traffic = float(traffic.sum())
    active_pairs = int(len(pair_df))

    return {
        "topology": topology,
        "active_pairs": active_pairs,
        "total_active_traffic": total_active_traffic,
        "min_path_count": float(path_count.min()),
        "mean_path_count": float(path_count.mean()),
        "traffic_weighted_path_diversity": float(
            (traffic * path_count).sum() / total_active_traffic
        ),
        "p50_path_count": float(np.percentile(path_count, 50)),
        "p90_path_count": float(np.percentile(path_count, 90)),
        "p95_path_count": float(np.percentile(path_count, 95)),
        "p99_path_count": float(np.percentile(path_count, 99)),
        "max_path_count": float(path_count.max()),
        "single_path_pair_fraction": float(single_path_mask.sum() / active_pairs),
        "single_path_traffic_exposure": float(
            traffic[single_path_mask].sum() / total_active_traffic
        ),
    }


def save_path_diversity_pairs(pair_df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(output_path, index=False)
    return output_path
