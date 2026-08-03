from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison
from llm_topology.viz.cdf import plot_grouped_metric_bar

CONFIGS = [
    {"name": "compare1024_tp8_dp16_pp8", "tp": 8, "dp": 16, "pp": 8},
    {"name": "compare1024_tp16_dp8_pp8", "tp": 16, "dp": 8, "pp": 8},
    {"name": "compare1024_tp32_dp4_pp8", "tp": 32, "dp": 4, "pp": 8},
]

SUMMARY_COLUMNS = [
    "experiment",
    "total_gpus",
    "tp",
    "dp",
    "pp",
    "topology",
    "weighted_avg_hops",
    "min_active_hop",
    "max_active_hop",
    "max_link_load",
    "p95_link_load",
    "p99_link_load",
    "max_utilization",
    "mean_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p97_latency_ms",
    "p99_latency_ms",
    "p100_latency_ms",
    "traffic_weighted_path_diversity",
    "mean_path_count",
    "p95_path_count",
    "max_path_count",
    "single_path_pair_fraction",
    "single_path_traffic_exposure",
]


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []

    for config in CONFIGS:
        spec = ExperimentSpec(
            name=config["name"],
            total_gpus=1024,
            tp=config["tp"],
            dp=config["dp"],
            pp=config["pp"],
            hbi_size=8,
            traffic_mode="synthetic",
            output_dir=results_dir / config["name"],
        )

        print(f"Running {spec.name}: TP={spec.tp}, DP={spec.dp}, PP={spec.pp} ...")
        results = run_comparison(spec)

        for topology, result in results.items():
            rows.append(
                {
                    "experiment": spec.name,
                    "total_gpus": spec.total_gpus,
                    "tp": spec.tp,
                    "dp": spec.dp,
                    "pp": spec.pp,
                    "topology": topology,
                    "weighted_avg_hops": result.weighted_avg_hops,
                    "min_active_hop": result.min_active_hop,
                    "max_active_hop": result.max_active_hop,
                    "max_link_load": result.link_load_summary["max_load"],
                    "p95_link_load": result.link_load_summary["p95_load"],
                    "p99_link_load": result.link_load_summary["p99_load"],
                    "max_utilization": result.utilization_summary["max_utilization"],
                    "mean_latency_ms": result.latency_summary["mean_latency_ms"],
                    "p50_latency_ms": result.latency_summary["p50_latency_ms"],
                    "p90_latency_ms": result.latency_summary["p90_latency_ms"],
                    "p95_latency_ms": result.latency_summary["p95_latency_ms"],
                    "p97_latency_ms": result.latency_summary["p97_latency_ms"],
                    "p99_latency_ms": result.latency_summary["p99_latency_ms"],
                    "p100_latency_ms": result.latency_summary["p100_latency_ms"],
                    "traffic_weighted_path_diversity": result.path_diversity_summary[
                        "traffic_weighted_path_diversity"
                    ],
                    "mean_path_count": result.path_diversity_summary["mean_path_count"],
                    "p95_path_count": result.path_diversity_summary["p95_path_count"],
                    "max_path_count": result.path_diversity_summary["max_path_count"],
                    "single_path_pair_fraction": result.path_diversity_summary[
                        "single_path_pair_fraction"
                    ],
                    "single_path_traffic_exposure": result.path_diversity_summary[
                        "single_path_traffic_exposure"
                    ],
                }
            )

        print(f"  Saved output tree under: {spec.output_dir}")

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = results_dir / "compare1024_all_configs_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    def by_config_by_topology(metric: str) -> dict[str, dict[str, float]]:
        data: dict[str, dict[str, float]] = {}
        for row in rows:
            data.setdefault(row["topology"], {})[row["experiment"]] = row[metric]
        return data

    plot_grouped_metric_bar(
        by_config_by_topology("weighted_avg_hops"),
        results_dir / "compare1024_weighted_avg_hops_by_config.png",
        title="1024-GPU Weighted Average Hops by Config",
        xlabel="Experiment config",
        ylabel="Weighted average hops",
    )

    plot_grouped_metric_bar(
        by_config_by_topology("p99_latency_ms"),
        results_dir / "compare1024_p99_latency_by_config.png",
        title="1024-GPU p99 Latency by Config",
        xlabel="Experiment config",
        ylabel="p99 latency (ms)",
    )

    plot_grouped_metric_bar(
        by_config_by_topology("max_link_load"),
        results_dir / "compare1024_max_link_load_by_config.png",
        title="1024-GPU Max Link Load by Config",
        xlabel="Experiment config",
        ylabel="Max link load",
    )

    plot_grouped_metric_bar(
        by_config_by_topology("p95_link_load"),
        results_dir / "compare1024_p95_link_load_by_config.png",
        title="1024-GPU p95 Link Load by Config",
        xlabel="Experiment config",
        ylabel="p95 link load",
    )

    plot_grouped_metric_bar(
        by_config_by_topology("traffic_weighted_path_diversity"),
        results_dir / "compare1024_path_diversity_by_config.png",
        title="1024-GPU Traffic-Weighted ECMP Path Diversity by Config",
        xlabel="Experiment config",
        ylabel="Weighted average number of ECMP paths",
    )

    plot_grouped_metric_bar(
        by_config_by_topology("single_path_traffic_exposure"),
        results_dir / "compare1024_single_path_exposure_by_config.png",
        title="1024-GPU Single-Path Traffic Exposure by Config",
        xlabel="Experiment config",
        ylabel="Fraction of active traffic with only one path",
    )

    plot_grouped_metric_bar(
        by_config_by_topology("max_path_count"),
        results_dir / "compare1024_max_path_count_by_config.png",
        title="1024-GPU Max ECMP Path Count by Config",
        xlabel="Experiment config",
        ylabel="Max number of ECMP paths",
    )

    print()
    print(f"Saved aggregate summary: {summary_path}")
    print("Saved aggregate plots:")
    print(results_dir / "compare1024_weighted_avg_hops_by_config.png")
    print(results_dir / "compare1024_p99_latency_by_config.png")
    print(results_dir / "compare1024_max_link_load_by_config.png")
    print(results_dir / "compare1024_p95_link_load_by_config.png")
    print(results_dir / "compare1024_path_diversity_by_config.png")
    print(results_dir / "compare1024_single_path_exposure_by_config.png")
    print(results_dir / "compare1024_max_path_count_by_config.png")


if __name__ == "__main__":
    main()
