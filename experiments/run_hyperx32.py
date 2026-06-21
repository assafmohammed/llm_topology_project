from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_topology.metrics.hops import (
    active_hop_values,
    hop_distribution,
    shortest_path_hop_matrix,
    weighted_average_hops,
)
from llm_topology.topologies.common import HOP_WEIGHT, ParallelismConfig, gpu_node
from llm_topology.topologies.hyperx import build_hyperx, hyperx_summary
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic
from llm_topology.viz.heatmap import plot_hop_heatmap, plot_traffic_heatmap


def nx_distance(graph, src_gpu: int, dst_gpu: int) -> int:
    return int(
        nx.shortest_path_length(
            graph,
            gpu_node(src_gpu),
            gpu_node(dst_gpu),
            weight=HOP_WEIGHT,
        )
    )


def main() -> None:
    cfg = ParallelismConfig(
        total_gpus=32,
        tp=8,
        dp=2,
        pp=2,
        hbi_size=8,
    )

    graph = build_hyperx(cfg)

    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(
            tp_bytes=100.0,
            dp_bytes=5.0,
            pp_bytes=1.0,
        ),
    )

    hop_matrix = shortest_path_hop_matrix(graph, cfg.total_gpus)
    active_values = active_hop_values(hop_matrix, traffic)
    distribution = hop_distribution(active_values)
    weighted_avg = weighted_average_hops(hop_matrix, traffic)

    results_dir = ROOT / "results"
    traffic_dir = ROOT / "traffic"

    results_dir.mkdir(exist_ok=True)
    traffic_dir.mkdir(exist_ok=True)

    hop_matrix_path = results_dir / "hyperx32_hop_matrix.csv"
    traffic_matrix_path = results_dir / "hyperx32_traffic_matrix.csv"
    distribution_path = results_dir / "hyperx32_active_hop_distribution.csv"
    traffic_npy_path = traffic_dir / "synthetic_hyperx32_tp8_dp2_pp2.npy"

    hop_heatmap_path = results_dir / "hyperx32_hop_heatmap.png"
    traffic_heatmap_path = results_dir / "hyperx32_traffic_heatmap.png"

    np.savetxt(hop_matrix_path, hop_matrix, delimiter=",", fmt="%.0f")
    np.savetxt(traffic_matrix_path, traffic, delimiter=",", fmt="%.1f")
    np.save(traffic_npy_path, traffic)

    with open(distribution_path, "w", encoding="utf-8") as f:
        f.write("hop,percentage\n")
        for hop, percentage in sorted(distribution.items()):
            f.write(f"{hop},{percentage}\n")

    plot_hop_heatmap(
        hop_matrix,
        hop_heatmap_path,
        title="HyperX-32 GPU-to-GPU Hop Distance Heatmap",
        show_values=False,
    )

    plot_traffic_heatmap(
        traffic,
        traffic_heatmap_path,
        title="HyperX-32 Synthetic LLM-like Traffic Matrix",
        show_values=False,
    )

    print(hyperx_summary(graph))
    print(f"Total traffic volume: {traffic.sum():.1f}")
    print(f"Active communication pairs: {active_values.size}")
    print(f"Active hop distribution: {distribution}")
    print(f"Traffic-weighted average hops: {weighted_avg:.4f}")

    print()
    print("Sanity checks:")
    print("GPU 0 -> GPU 7 same HBI:", nx_distance(graph, 0, 7), "hop")
    print("GPU 0 -> GPU 8 DP dimension:", nx_distance(graph, 0, 8), "hops")
    print("GPU 0 -> GPU 16 PP dimension:", nx_distance(graph, 0, 16), "hops")

    print()
    print("Saved files:")
    print(hop_matrix_path)
    print(traffic_matrix_path)
    print(distribution_path)
    print(traffic_npy_path)
    print(hop_heatmap_path)
    print(traffic_heatmap_path)


if __name__ == "__main__":
    main()