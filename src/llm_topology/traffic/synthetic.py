from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llm_topology.topologies.common import ParallelismConfig


@dataclass(frozen=True)
class TrafficWeights:
    """
    Synthetic traffic volumes.

    TP is dominant.
    DP is smaller.
    PP is smallest.
    """

    tp_bytes: float = 100.0
    dp_bytes: float = 5.0
    pp_bytes: float = 1.0


def decompose_gpu_index(gpu_id: int, cfg: ParallelismConfig) -> tuple[int, int, int]:
    cfg.validate()

    if gpu_id < 0 or gpu_id >= cfg.total_gpus:
        raise ValueError(f"gpu_id must be in [0, {cfg.total_gpus}), got {gpu_id}")

    tp_index = gpu_id % cfg.tp
    dp_index = (gpu_id // cfg.tp) % cfg.dp
    pp_index = gpu_id // (cfg.tp * cfg.dp)

    return tp_index, dp_index, pp_index


def compose_gpu_index(
    tp_index: int,
    dp_index: int,
    pp_index: int,
    cfg: ParallelismConfig,
) -> int:
    cfg.validate()

    if not (0 <= tp_index < cfg.tp):
        raise ValueError("tp_index out of range")
    if not (0 <= dp_index < cfg.dp):
        raise ValueError("dp_index out of range")
    if not (0 <= pp_index < cfg.pp):
        raise ValueError("pp_index out of range")

    return tp_index + cfg.tp * dp_index + cfg.tp * cfg.dp * pp_index


def generate_llm_like_traffic(
    cfg: ParallelismConfig,
    weights: TrafficWeights | None = None,
    *,
    bidirectional: bool = True,
) -> np.ndarray:
    """
    Generate a simple LLM-like transport matrix.

    S[i, j] is the traffic volume from GPU i to GPU j.

    Patterns:
    1. Within TP group: ring traffic between adjacent TP ranks.
    2. Within stage / DP dimension: traffic to same TP rank in adjacent DP group.
    3. Cross stage / PP dimension: traffic to same TP and DP rank in adjacent PP stage.
    """
    cfg.validate()

    if weights is None:
        weights = TrafficWeights()

    matrix = np.zeros((cfg.total_gpus, cfg.total_gpus), dtype=float)

    for gpu_id in range(cfg.total_gpus):
        tp_index, dp_index, pp_index = decompose_gpu_index(gpu_id, cfg)

        # 1) Dominant TP ring traffic.
        tp_next = (tp_index + 1) % cfg.tp
        dst = compose_gpu_index(tp_next, dp_index, pp_index, cfg)
        matrix[gpu_id, dst] += weights.tp_bytes

        if bidirectional:
            tp_prev = (tp_index - 1) % cfg.tp
            dst = compose_gpu_index(tp_prev, dp_index, pp_index, cfg)
            matrix[gpu_id, dst] += weights.tp_bytes

        # 2) Smaller within-stage / DP traffic.
        if cfg.dp > 1:
            dp_next = (dp_index + 1) % cfg.dp
            dst = compose_gpu_index(tp_index, dp_next, pp_index, cfg)
            matrix[gpu_id, dst] += weights.dp_bytes

            if bidirectional and cfg.dp > 2:
                dp_prev = (dp_index - 1) % cfg.dp
                dst = compose_gpu_index(tp_index, dp_prev, pp_index, cfg)
                matrix[gpu_id, dst] += weights.dp_bytes

        # 3) Smallest cross-stage / PP traffic.
        if cfg.pp > 1:
            pp_next = (pp_index + 1) % cfg.pp
            dst = compose_gpu_index(tp_index, dp_index, pp_next, cfg)
            matrix[gpu_id, dst] += weights.pp_bytes

            if bidirectional and cfg.pp > 2:
                pp_prev = (pp_index - 1) % cfg.pp
                dst = compose_gpu_index(tp_index, dp_index, pp_prev, cfg)
                matrix[gpu_id, dst] += weights.pp_bytes

    np.fill_diagonal(matrix, 0.0)
    return matrix