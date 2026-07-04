from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_topology.metrics.hops import (
    active_hop_values,
    hop_distribution,
    shortest_path_hop_matrix,
    weighted_average_hops,
)
from llm_topology.metrics.latency import compute_pair_latencies, summarize_latency
from llm_topology.metrics.link_load import (
    add_utilization,
    compute_ecmp_link_loads,
    link_loads_to_dataframe,
    summarize_link_loads,
    summarize_utilization,
)
from llm_topology.metrics.routing import dragonfly_paths, fat_tree_paths, hyperx_paths
from llm_topology.topologies.common import ParallelismConfig
from llm_topology.topologies.dragonfly_plus import build_dragonfly_plus, dragonfly_hop_matrix
from llm_topology.topologies.fat_tree import build_fat_tree, fat_tree_hop_matrix
from llm_topology.topologies.hyperx import build_hyperx
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic
from llm_topology.viz.cdf import (
    plot_cdf,
    plot_grouped_bar,
    plot_grouped_metric_bar,
    plot_single_bar,
    plot_tail_bar,
)

BANDWIDTH_BYTES_PER_SEC = 50e9
TRAFFIC_UNIT_BYTES = 1_000_000

FILE_PREFIX = {
    "HyperX": "hyperx32",
    "Fat-tree": "fat_tree32",
    "Dragonfly+": "dragonfly32",
}


def build_topologies(cfg: ParallelismConfig) -> dict[str, dict]:
    hyperx_graph = build_hyperx(cfg)
    fat_tree_graph = build_fat_tree(cfg)
    dragonfly_graph = build_dragonfly_plus(cfg)

    return {
        "HyperX": {
            "graph": hyperx_graph,
            "hop_matrix": shortest_path_hop_matrix(hyperx_graph, cfg.total_gpus),
            "path_provider": lambda i, j: hyperx_paths(i, j, cfg),
        },
        "Fat-tree": {
            "graph": fat_tree_graph,
            "hop_matrix": fat_tree_hop_matrix(cfg),
            "path_provider": lambda i, j: fat_tree_paths(i, j, cfg, num_spines=2),
        },
        "Dragonfly+": {
            "graph": dragonfly_graph,
            "hop_matrix": dragonfly_hop_matrix(cfg),
            "path_provider": lambda i, j: dragonfly_paths(i, j, cfg, spines_per_group=2),
        },
    }


def compare_topologies(cfg: ParallelismConfig, traffic: np.ndarray) -> dict[str, dict]:
    """
    Build all three 32-GPU topologies and compute hop, link-load, and
    latency metrics for the same synthetic traffic matrix.
    """
    topologies = build_topologies(cfg)

    results: dict[str, dict] = {}

    for name, info in topologies.items():
        hop_matrix = info["hop_matrix"]
        path_provider = info["path_provider"]

        active_values = active_hop_values(hop_matrix, traffic)
        distribution = hop_distribution(active_values)
        weighted_avg = weighted_average_hops(hop_matrix, traffic)

        loads = compute_ecmp_link_loads(traffic, path_provider)
        link_load_df = link_loads_to_dataframe(name, loads)
        link_load_summary = summarize_link_loads(name, loads)

        results[name] = {
            "graph": info["graph"],
            "hop_matrix": hop_matrix,
            "path_provider": path_provider,
            "active_values": active_values,
            "distribution": distribution,
            "weighted_avg": weighted_avg,
            "min_active_hop": int(active_values.min()),
            "max_active_hop": int(active_values.max()),
            "active_pair_count": int(active_values.size),
            "loads": loads,
            "link_load_df": link_load_df,
            "link_load_summary": link_load_summary,
        }

    # Shared capacity across topologies, sized off the busiest link overall,
    # so utilization/latency numbers are comparable apples-to-apples.
    global_max_load = max(r["link_load_summary"]["max_load"] for r in results.values())
    capacity = global_max_load * 1.2

    for name, result in results.items():
        utilization_df = add_utilization(result["link_load_df"], capacity)
        utilization_summary = summarize_utilization(
            name, utilization_df["utilization"].to_numpy()
        )

        latency_df = compute_pair_latencies(
            traffic,
            result["path_provider"],
            result["loads"],
            capacity=capacity,
            bandwidth_bytes_per_sec=BANDWIDTH_BYTES_PER_SEC,
            traffic_unit_bytes=TRAFFIC_UNIT_BYTES,
        )
        latency_summary = summarize_latency(name, latency_df)

        result["capacity"] = capacity
        result["utilization_df"] = utilization_df
        result["utilization_summary"] = utilization_summary
        result["latency_df"] = latency_df
        result["latency_summary"] = latency_summary

    return results


def save_per_topology_files(results_dir: Path, results: dict[str, dict]) -> None:
    for name, result in results.items():
        prefix = FILE_PREFIX[name]
        result["link_load_df"].to_csv(results_dir / f"{prefix}_link_load.csv", index=False)
        result["utilization_df"].to_csv(
            results_dir / f"{prefix}_link_utilization.csv", index=False
        )
        result["latency_df"].to_csv(results_dir / f"{prefix}_latency_pairs.csv", index=False)


def save_combined_summary(results_dir: Path, cfg: ParallelismConfig, results: dict[str, dict]) -> None:
    rows = []
    for name, result in results.items():
        link_load_summary = result["link_load_summary"]
        utilization_summary = result["utilization_summary"]
        latency_summary = result["latency_summary"]

        rows.append(
            {
                "topology": name,
                "total_gpus": cfg.total_gpus,
                "tp": cfg.tp,
                "dp": cfg.dp,
                "pp": cfg.pp,
                "hbi_size": cfg.hbi_size,
                "active_pairs": result["active_pair_count"],
                "min_active_hop": result["min_active_hop"],
                "max_active_hop": result["max_active_hop"],
                "weighted_average_hops": round(result["weighted_avg"], 4),
                "link_count": link_load_summary["link_count"],
                "max_link_load": link_load_summary["max_load"],
                "p95_link_load": link_load_summary["p95_load"],
                "p99_link_load": link_load_summary["p99_load"],
                "max_utilization": utilization_summary["max_utilization"],
                "p95_latency_ms": latency_summary["p95_latency_ms"],
                "p97_latency_ms": latency_summary["p97_latency_ms"],
                "p99_latency_ms": latency_summary["p99_latency_ms"],
                "p100_latency_ms": latency_summary["p100_latency_ms"],
            }
        )

    pd.DataFrame(rows).to_csv(results_dir / "compare32_summary.csv", index=False)


def save_combined_hop_distribution(results_dir: Path, results: dict[str, dict]) -> None:
    with open(results_dir / "compare32_active_hop_distribution.csv", "w", encoding="utf-8") as f:
        f.write("topology,hop,percentage\n")
        for name, result in results.items():
            for hop, percentage in sorted(result["distribution"].items()):
                f.write(f"{name},{hop},{percentage}\n")


def save_combined_link_load_summary(results_dir: Path, results: dict[str, dict]) -> None:
    rows = [result["link_load_summary"] for result in results.values()]
    columns = [
        "topology",
        "link_count",
        "min_load",
        "mean_load",
        "p50_load",
        "p90_load",
        "p95_load",
        "p99_load",
        "max_load",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        results_dir / "compare32_link_load_summary.csv", index=False
    )


def save_combined_utilization_summary(results_dir: Path, results: dict[str, dict]) -> None:
    rows = [result["utilization_summary"] for result in results.values()]
    columns = [
        "topology",
        "min_utilization",
        "mean_utilization",
        "p50_utilization",
        "p90_utilization",
        "p95_utilization",
        "p99_utilization",
        "max_utilization",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        results_dir / "compare32_link_utilization_summary.csv", index=False
    )


def save_combined_latency_percentiles(results_dir: Path, results: dict[str, dict]) -> None:
    rows = [result["latency_summary"] for result in results.values()]
    columns = [
        "topology",
        "active_pairs",
        "mean_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "p95_latency_ms",
        "p97_latency_ms",
        "p99_latency_ms",
        "p100_latency_ms",
    ]
    pd.DataFrame(rows, columns=columns).to_csv(
        results_dir / "compare32_latency_percentiles.csv", index=False
    )


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

    save_per_topology_files(results_dir, results)
    save_combined_summary(results_dir, cfg, results)
    save_combined_hop_distribution(results_dir, results)
    save_combined_link_load_summary(results_dir, results)
    save_combined_utilization_summary(results_dir, results)
    save_combined_latency_percentiles(results_dir, results)

    plot_grouped_bar(
        {name: result["distribution"] for name, result in results.items()},
        results_dir / "compare32_active_hop_distribution.png",
        title="32-GPU Active Hop Distribution by Topology",
        xlabel="Hop count",
        ylabel="Fraction of active pairs",
    )

    ranked = sorted(results.items(), key=lambda item: item[1]["weighted_avg"])

    plot_single_bar(
        {name: result["weighted_avg"] for name, result in ranked},
        results_dir / "compare32_weighted_average_hops.png",
        title="32-GPU Traffic-Weighted Average Hops by Topology",
        ylabel="Weighted average hops",
    )

    plot_cdf(
        {name: result["link_load_df"]["load"].to_numpy() for name, result in results.items()},
        results_dir / "compare32_link_load_cdf.png",
        title="32-GPU Link-Load CDF by Topology",
        xlabel="Link load",
    )

    plot_cdf(
        {name: result["latency_df"]["latency_ms"].to_numpy() for name, result in results.items()},
        results_dir / "compare32_latency_cdf.png",
        title="32-GPU Congestion-Aware Latency CDF by Topology",
        xlabel="Latency (ms)",
    )

    plot_tail_bar(
        {name: result["latency_summary"]["p99_latency_ms"] for name, result in results.items()},
        results_dir / "compare32_latency_p99_bar.png",
        title="32-GPU p99 Latency by Topology",
        ylabel="p99 latency (ms)",
    )

    plot_tail_bar(
        {name: result["latency_summary"]["p100_latency_ms"] for name, result in results.items()},
        results_dir / "compare32_latency_p100_bar.png",
        title="32-GPU p100 (max) Latency by Topology",
        ylabel="p100 latency (ms)",
    )

    # p99/p100 tend to collapse to the same value in this small 32-GPU case
    # (tail latency is dominated by local TP traffic inside HBI, which is
    # identical across topologies), so mean/p50/p90/p95 are the plots that
    # actually show topology differences here.
    mean_latency_values = {
        name: result["latency_summary"]["mean_latency_ms"] for name, result in results.items()
    }
    p50_latency_values = {
        name: result["latency_summary"]["p50_latency_ms"] for name, result in results.items()
    }
    p90_latency_values = {
        name: result["latency_summary"]["p90_latency_ms"] for name, result in results.items()
    }
    p95_latency_values = {
        name: result["latency_summary"]["p95_latency_ms"] for name, result in results.items()
    }

    plot_single_bar(
        mean_latency_values,
        results_dir / "compare32_latency_mean_bar.png",
        title="32-GPU Mean Congestion-Aware Latency by Topology",
        ylabel="Mean latency (ms)",
    )

    plot_single_bar(
        p50_latency_values,
        results_dir / "compare32_latency_p50_bar.png",
        title="32-GPU p50 Congestion-Aware Latency by Topology",
        ylabel="p50 latency (ms)",
    )

    plot_single_bar(
        p90_latency_values,
        results_dir / "compare32_latency_p90_bar.png",
        title="32-GPU p90 Congestion-Aware Latency by Topology",
        ylabel="p90 latency (ms)",
    )

    plot_single_bar(
        p95_latency_values,
        results_dir / "compare32_latency_p95_bar.png",
        title="32-GPU p95 Congestion-Aware Latency by Topology",
        ylabel="p95 latency (ms)",
    )

    latency_percentile_data = {
        name: {
            "p50": result["latency_summary"]["p50_latency_ms"],
            "p90": result["latency_summary"]["p90_latency_ms"],
            "p95": result["latency_summary"]["p95_latency_ms"],
            "p99": result["latency_summary"]["p99_latency_ms"],
            "p100": result["latency_summary"]["p100_latency_ms"],
        }
        for name, result in results.items()
    }

    plot_grouped_metric_bar(
        latency_percentile_data,
        results_dir / "compare32_latency_percentile_bars.png",
        title="32-GPU Latency Percentiles by Topology",
        xlabel="Latency percentile",
        ylabel="Latency (ms)",
    )

    print("32-GPU full metrics comparison")
    print(f"Configuration: TP={cfg.tp}, DP={cfg.dp}, PP={cfg.pp}, HBI={cfg.hbi_size}")
    print()

    for name, result in ranked:
        link_load_summary = result["link_load_summary"]
        utilization_summary = result["utilization_summary"]
        latency_summary = result["latency_summary"]

        print(f"{name}:")
        print(f"  Active hop distribution: {result['distribution']}")
        print(f"  Weighted avg hops: {result['weighted_avg']:.4f}")
        print(f"  Max link load: {link_load_summary['max_load']:.4f}")
        print(f"  p95 link load: {link_load_summary['p95_load']:.4f}")
        print(f"  Max utilization: {utilization_summary['max_utilization']:.4f}")
        print(f"  p95 latency: {latency_summary['p95_latency_ms']:.4f} ms")
        print(f"  p99 latency: {latency_summary['p99_latency_ms']:.4f} ms")
        print(f"  p100 latency: {latency_summary['p100_latency_ms']:.4f} ms")
        print()

    print("Interpretation:")
    print("- Hop count alone is not enough.")
    print("- Link-load CDF shows how evenly traffic is distributed.")
    print("- Latency combines message size, path length, and congestion factor.")
    print("- p99/p100 are identical here because tail latency is dominated by local HBI TP traffic.")
    print("- Mean/p50/p90 latency are more useful for showing topology differences at this scale.")
    print("- Tail latency should become more informative after scaling and testing TP > 8.")
    print("- This is still a 32-GPU debug case; scaling is Phase 2.")

    print()
    print("Saved files:")
    for path in sorted(results_dir.glob("compare32_*")):
        print(path)
    for name in results:
        prefix = FILE_PREFIX[name]
        print(results_dir / f"{prefix}_link_load.csv")
        print(results_dir / f"{prefix}_link_utilization.csv")
        print(results_dir / f"{prefix}_latency_pairs.csv")


if __name__ == "__main__":
    main()
