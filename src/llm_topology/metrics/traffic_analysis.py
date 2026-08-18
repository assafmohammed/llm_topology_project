from __future__ import annotations

import numpy as np
import pandas as pd

from ..topologies.common import ParallelismConfig
from ..traffic.synthetic import decompose_gpu_index


def analyze_traffic_matrix(traffic_matrix: np.ndarray, cfg: ParallelismConfig) -> dict:
    """
    Paper Section 5.1: transport matrix / traffic analysis.

    Classifies every active pair (S[i,j] > 0) as TP/DP/PP/other traffic based
    on which single index (TP, DP, or PP) differs between src and dst, using
    the same GPU-index decomposition the synthetic traffic generator uses.
    """
    cfg.validate()

    n = traffic_matrix.shape[0]
    if traffic_matrix.shape != (n, n):
        raise ValueError(f"traffic_matrix must be square, got shape {traffic_matrix.shape}")

    src_idx, dst_idx = np.nonzero(traffic_matrix > 0)
    active_pairs = int(src_idx.size)
    total_traffic_volume = float(traffic_matrix.sum())
    traffic_density = active_pairs / (n * (n - 1)) if n > 1 else 0.0

    tp_volume = 0.0
    dp_volume = 0.0
    pp_volume = 0.0
    other_volume = 0.0
    active_values = np.empty(active_pairs, dtype=float)

    for k, (src, dst) in enumerate(zip(src_idx, dst_idx)):
        traffic = float(traffic_matrix[src, dst])
        active_values[k] = traffic

        src_tp, src_dp, src_pp = decompose_gpu_index(int(src), cfg)
        dst_tp, dst_dp, dst_pp = decompose_gpu_index(int(dst), cfg)

        same_tp = src_tp == dst_tp
        same_dp = src_dp == dst_dp
        same_pp = src_pp == dst_pp

        if same_dp and same_pp and not same_tp:
            tp_volume += traffic
        elif same_tp and same_pp and not same_dp:
            dp_volume += traffic
        elif same_tp and same_dp and not same_pp:
            pp_volume += traffic
        else:
            other_volume += traffic

    def frac(volume: float) -> float:
        return volume / total_traffic_volume if total_traffic_volume > 0 else 0.0

    result = {
        "active_pairs": active_pairs,
        "total_traffic_volume": total_traffic_volume,
        "traffic_density": traffic_density,
        "tp_traffic_volume": tp_volume,
        "dp_traffic_volume": dp_volume,
        "pp_traffic_volume": pp_volume,
        "other_traffic_volume": other_volume,
        "tp_traffic_fraction": frac(tp_volume),
        "dp_traffic_fraction": frac(dp_volume),
        "pp_traffic_fraction": frac(pp_volume),
        "other_traffic_fraction": frac(other_volume),
    }

    if active_values.size == 0:
        result.update(
            {
                "max_pair_traffic": 0.0,
                "mean_active_pair_traffic": 0.0,
                "p50_active_pair_traffic": 0.0,
                "p90_active_pair_traffic": 0.0,
                "p95_active_pair_traffic": 0.0,
                "p99_active_pair_traffic": 0.0,
            }
        )
    else:
        result.update(
            {
                "max_pair_traffic": float(active_values.max()),
                "mean_active_pair_traffic": float(active_values.mean()),
                "p50_active_pair_traffic": float(np.percentile(active_values, 50)),
                "p90_active_pair_traffic": float(np.percentile(active_values, 90)),
                "p95_active_pair_traffic": float(np.percentile(active_values, 95)),
                "p99_active_pair_traffic": float(np.percentile(active_values, 99)),
            }
        )

    return result


def traffic_analysis_dataframe(experiment: str, analysis: dict) -> pd.DataFrame:
    return pd.DataFrame([{"experiment": experiment, **analysis}])
