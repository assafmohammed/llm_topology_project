from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_topology.metrics.hops import (
    active_hop_values,
    hop_distribution,
    shortest_path_hop_matrix,
    weighted_average_hops,
)
from llm_topology.topologies.common import ParallelismConfig
from llm_topology.topologies.dragonfly_plus import build_dragonfly_plus, dragonfly_hop_matrix
from llm_topology.topologies.fat_tree import build_fat_tree, fat_tree_hop_matrix
from llm_topology.topologies.hyperx import build_hyperx
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic
from llm_topology.viz.cdf import plot_grouped_bar, plot_single_bar


def compare_topologies(cfg: ParallelismConfig, traffic) -> dict[str, dict]:
    """
    Build all three 32-GPU topologies and compute active-hop metrics for
    the same synthetic traffic matrix.
    """
    topology_hop_matrices = {
        "HyperX": shortest_path_hop_matrix(build_hyperx(cfg), cfg.total_gpus),
        "Fat-tree": fat_tree_hop_matrix(cfg),
        "Dragonfly+": dragonfly_hop_matrix(cfg),
    }

    results: dict[str, dict] = {}

    for name, hop_matrix in topology_hop_matrices.items():
        active_values = active_hop_values(hop_matrix, traffic)
        distribution = hop_distribution(active_values)
        weighted_avg = weighted_average_hops(hop_matrix, traffic)

        results[name] = {
            "hop_matrix": hop_matrix,
            "active_values": active_values,
            "distribution": distribution,
            "weighted_avg": weighted_avg,
            "min_active_hop": int(active_values.min()),
            "max_active_hop": int(active_values.max()),
            "active_pair_count": int(active_values.size),
        }

    return results


def main() -> None:
    cfg = ParallelismConfig(
        total_gpus=32,
        tp=8,
        dp=2,
        pp=2,
        hbi_size=8,
    )

    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(
            tp_bytes=100.0,
            dp_bytes=5.0,
            pp_bytes=1.0,
        ),
    )

    results = compare_topologies(cfg, traffic)

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    summary_path = results_dir / "compare32_summary.csv"
    distribution_path = results_dir / "compare32_active_hop_distribution.csv"
    distribution_plot_path = results_dir / "compare32_active_hop_distribution.png"
    weighted_avg_plot_path = results_dir / "compare32_weighted_average_hops.png"

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            "topology,total_gpus,tp,dp,pp,hbi_size,active_pairs,"
            "min_active_hop,max_active_hop,weighted_average_hops\n"
        )
        for name, result in results.items():
            f.write(
                f"{name},{cfg.total_gpus},{cfg.tp},{cfg.dp},{cfg.pp},{cfg.hbi_size},"
                f"{result['active_pair_count']},{result['min_active_hop']},"
                f"{result['max_active_hop']},{result['weighted_avg']:.4f}\n"
            )

    with open(distribution_path, "w", encoding="utf-8") as f:
        f.write("topology,hop,percentage\n")
        for name, result in results.items():
            for hop, percentage in sorted(result["distribution"].items()):
                f.write(f"{name},{hop},{percentage}\n")

    plot_grouped_bar(
        {name: result["distribution"] for name, result in results.items()},
        distribution_plot_path,
        title="32-GPU Active Hop Distribution by Topology",
        xlabel="Hop count",
        ylabel="Fraction of active pairs",
    )

    ranked = sorted(results.items(), key=lambda item: item[1]["weighted_avg"])

    plot_single_bar(
        {name: result["weighted_avg"] for name, result in ranked},
        weighted_avg_plot_path,
        title="32-GPU Traffic-Weighted Average Hops by Topology",
        ylabel="Weighted average hops",
    )

    print("32-GPU topology comparison")
    print(f"Configuration: TP={cfg.tp}, DP={cfg.dp}, PP={cfg.pp}, HBI={cfg.hbi_size}")
    print()
    print(f"{'Topology':<14}{'Active distribution':<28}{'Weighted avg hops'}")
    for name, result in ranked:
        print(f"{name:<14}{str(result['distribution']):<28}{result['weighted_avg']:.4f}")

    print()
    print("Interpretation:")
    print("- In this small 32-GPU case, Fat-tree has the lowest active weighted hop count.")
    print("- HyperX keeps active communication bounded by 3 hops.")
    print(
        "- Dragonfly+ has longer active paths because same-group traffic is 4 hops "
        "and cross-group traffic is 5 hops."
    )
    print(
        "- This does not prove Fat-tree is better overall; larger scale, TP > 8, "
        "ECMP load, and latency are still needed."
    )

    print()
    print("Saved files:")
    print(summary_path)
    print(distribution_path)
    print(distribution_plot_path)
    print(weighted_avg_plot_path)


if __name__ == "__main__":
    main()
