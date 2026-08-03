from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison
from llm_topology.viz.cdf import plot_grouped_metric_bar

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


def generate_valid_specs(
    total_gpus: int,
    tp_values: list[int],
    pp_values: list[int],
    hbi_size: int = 8,
) -> list[ExperimentSpec]:
    """
    Build one ExperimentSpec per valid (TP, PP) combination, where
    DP = total_gpus // (TP * PP) divides evenly and is >= 1.
    """
    specs = []

    for tp in tp_values:
        for pp in pp_values:
            if total_gpus % (tp * pp) != 0:
                continue

            dp = total_gpus // (tp * pp)
            if dp < 1:
                continue

            name = f"sweep{total_gpus}_tp{tp}_dp{dp}_pp{pp}"
            specs.append(
                ExperimentSpec(
                    name=name,
                    total_gpus=total_gpus,
                    tp=tp,
                    dp=dp,
                    pp=pp,
                    hbi_size=hbi_size,
                    traffic_mode="synthetic",
                    output_dir=ROOT / "results" / "paper_sweep" / name,
                )
            )

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TP/DP/PP sweep across topologies.")
    parser.add_argument("--total-gpus", type=int, default=1024)
    parser.add_argument("--tp-values", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--pp-values", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N valid specs (useful for a quick smoke test).",
    )
    parser.add_argument(
        "--only-config",
        type=str,
        default=None,
        help="Run only the spec whose name matches exactly, e.g. sweep1024_tp8_dp16_pp8.",
    )
    return parser.parse_args()


def run_sweep(
    total_gpus: int,
    tp_values: list[int],
    pp_values: list[int],
    *,
    limit: int | None = None,
    only_config: str | None = None,
) -> None:
    """
    Core sweep logic, callable directly (e.g. from run_all.py) without
    going through argparse/sys.argv.
    """
    specs = generate_valid_specs(total_gpus, tp_values, pp_values)

    if only_config:
        specs = [spec for spec in specs if spec.name == only_config]
        if not specs:
            raise ValueError(f"No sweep spec named {only_config!r} was generated")

    if limit is not None:
        specs = specs[:limit]

    print(f"Running {len(specs)} sweep configs for total_gpus={total_gpus}")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    rows = []

    for spec in specs:
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

    if not rows:
        print("No specs produced results; nothing to summarize.")
        return

    summary_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    summary_path = results_dir / "paper_sweep_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    def by_experiment_by_topology(metric: str) -> dict[str, dict[str, float]]:
        data: dict[str, dict[str, float]] = {}
        for row in rows:
            data.setdefault(row["topology"], {})[row["experiment"]] = row[metric]
        return data

    plot_grouped_metric_bar(
        by_experiment_by_topology("weighted_avg_hops"),
        results_dir / "paper_sweep_weighted_avg_hops.png",
        title="Paper Sweep: Weighted Average Hops by Config",
        xlabel="Experiment config",
        ylabel="Weighted average hops",
    )

    plot_grouped_metric_bar(
        by_experiment_by_topology("p99_latency_ms"),
        results_dir / "paper_sweep_p99_latency.png",
        title="Paper Sweep: p99 Latency by Config",
        xlabel="Experiment config",
        ylabel="p99 latency (ms)",
    )

    plot_grouped_metric_bar(
        by_experiment_by_topology("max_link_load"),
        results_dir / "paper_sweep_max_link_load.png",
        title="Paper Sweep: Max Link Load by Config",
        xlabel="Experiment config",
        ylabel="Max link load",
    )

    plot_grouped_metric_bar(
        by_experiment_by_topology("p95_link_load"),
        results_dir / "paper_sweep_p95_link_load.png",
        title="Paper Sweep: p95 Link Load by Config",
        xlabel="Experiment config",
        ylabel="p95 link load",
    )

    plot_grouped_metric_bar(
        by_experiment_by_topology("traffic_weighted_path_diversity"),
        results_dir / "paper_sweep_path_diversity.png",
        title="Paper Sweep: Traffic-Weighted ECMP Path Diversity by Config",
        xlabel="Experiment config",
        ylabel="Weighted average number of ECMP paths",
    )

    plot_grouped_metric_bar(
        by_experiment_by_topology("single_path_traffic_exposure"),
        results_dir / "paper_sweep_single_path_exposure.png",
        title="Paper Sweep: Single-Path Traffic Exposure by Config",
        xlabel="Experiment config",
        ylabel="Fraction of active traffic with only one path",
    )

    print()
    print(f"Saved aggregate summary: {summary_path}")
    print("Saved aggregate plots:")
    print(results_dir / "paper_sweep_weighted_avg_hops.png")
    print(results_dir / "paper_sweep_p99_latency.png")
    print(results_dir / "paper_sweep_max_link_load.png")
    print(results_dir / "paper_sweep_p95_link_load.png")
    print(results_dir / "paper_sweep_path_diversity.png")
    print(results_dir / "paper_sweep_single_path_exposure.png")


def main() -> None:
    args = parse_args()
    run_sweep(
        args.total_gpus,
        args.tp_values,
        args.pp_values,
        limit=args.limit,
        only_config=args.only_config,
    )


if __name__ == "__main__":
    main()
