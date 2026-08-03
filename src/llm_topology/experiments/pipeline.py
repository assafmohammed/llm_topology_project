from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.hops import active_hop_values, hop_distribution, weighted_average_hops
from ..metrics.latency import compute_pair_latencies, summarize_latency
from ..metrics.link_load import (
    add_utilization,
    compute_ecmp_link_loads,
    link_loads_to_dataframe,
    summarize_link_loads,
    summarize_utilization,
)
from ..metrics.path_diversity import (
    compute_path_diversity_pairs,
    save_path_diversity_pairs,
    summarize_path_diversity,
)
from ..metrics.routing import dragonfly_paths, fat_tree_paths, hyperx_paths
from ..topologies.common import ParallelismConfig
from ..topologies.dragonfly_plus import dragonfly_hop_matrix
from ..topologies.fat_tree import fat_tree_hop_matrix
from ..topologies.hyperx import hyperx_hop_matrix
from ..traffic.aicb import aicb_to_matrix
from ..traffic.matrix_loader import load_traffic_matrix
from ..traffic.synthetic import TrafficWeights, generate_llm_like_traffic
from ..viz.cdf import (
    plot_cdf,
    plot_grouped_bar,
    plot_grouped_metric_bar,
    plot_path_count_cdf,
    plot_path_diversity_bar,
    plot_single_bar,
    plot_single_path_exposure_bar,
    plot_tail_bar,
)

BANDWIDTH_BYTES_PER_SEC = 50e9
TRAFFIC_UNIT_BYTES = 1_000_000
CAPACITY_HEADROOM = 1.2

TOPOLOGIES = ("Fat-tree", "HyperX", "Dragonfly+")

SAFE_NAME = {
    "Fat-tree": "fat_tree",
    "HyperX": "hyperx",
    "Dragonfly+": "dragonfly",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    total_gpus: int
    tp: int
    dp: int
    pp: int
    hbi_size: int = 8
    traffic_mode: str = "synthetic"
    traffic_path: str | None = None
    output_dir: str | Path = "results"

    def parallelism_config(self) -> ParallelismConfig:
        return ParallelismConfig(
            total_gpus=self.total_gpus,
            tp=self.tp,
            dp=self.dp,
            pp=self.pp,
            hbi_size=self.hbi_size,
        )


@dataclass
class TopologyResult:
    topology: str
    hop_distribution: dict[int, float]
    weighted_avg_hops: float
    min_active_hop: float
    max_active_hop: float
    link_load_summary: dict
    utilization_summary: dict
    latency_summary: dict
    path_diversity_summary: dict
    paths: dict[str, Path] = field(default_factory=dict)


def build_traffic(spec: ExperimentSpec) -> np.ndarray:
    """
    Build the GPU x GPU traffic matrix for an experiment, per spec.traffic_mode.
    """
    if spec.traffic_mode == "synthetic":
        traffic = generate_llm_like_traffic(
            spec.parallelism_config(),
            TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
        )
    elif spec.traffic_mode in ("npy", "csv"):
        if not spec.traffic_path:
            raise ValueError(f"traffic_path is required for traffic_mode={spec.traffic_mode!r}")
        traffic = load_traffic_matrix(spec.traffic_path)
    elif spec.traffic_mode == "aicb":
        if not spec.traffic_path:
            raise ValueError("traffic_path is required for traffic_mode='aicb'")
        traffic = aicb_to_matrix(spec.traffic_path, spec.total_gpus)
    else:
        raise ValueError(f"Unsupported traffic_mode: {spec.traffic_mode!r}")

    traffic = np.asarray(traffic, dtype=float)
    expected_shape = (spec.total_gpus, spec.total_gpus)
    if traffic.shape != expected_shape:
        raise ValueError(
            f"Traffic matrix shape {traffic.shape} does not match expected {expected_shape}"
        )

    np.fill_diagonal(traffic, 0.0)
    return traffic


def _topology_hop_matrix_and_paths(topology_name: str, cfg: ParallelismConfig):
    if topology_name == "HyperX":
        return hyperx_hop_matrix(cfg), (lambda i, j: hyperx_paths(i, j, cfg))
    if topology_name == "Fat-tree":
        return fat_tree_hop_matrix(cfg), (lambda i, j: fat_tree_paths(i, j, cfg, num_spines=2))
    if topology_name == "Dragonfly+":
        return dragonfly_hop_matrix(cfg), (
            lambda i, j: dragonfly_paths(i, j, cfg, spines_per_group=2)
        )
    raise ValueError(f"Unknown topology: {topology_name!r}")


def run_topology_metrics(
    topology_name: str,
    cfg: ParallelismConfig,
    traffic: np.ndarray,
    output_dir: str | Path,
    *,
    capacity: float | None = None,
) -> TopologyResult:
    """
    Compute hop, link-load, utilization, and latency metrics for a single
    topology and save its per-topology files under output_dir.

    Hop matrices are formula-based for all three topologies (never generic
    NetworkX shortest paths), which is both correct (avoids HBI-transit
    shortcuts) and fast enough for 1024-GPU configs.

    If capacity is not given, it defaults to this topology's own busiest
    link * 1.2. run_comparison instead passes a capacity shared across all
    topologies so utilization/latency are apples-to-apples.
    """
    cfg.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = SAFE_NAME[topology_name]

    hop_matrix, path_provider = _topology_hop_matrix_and_paths(topology_name, cfg)

    active_values = active_hop_values(hop_matrix, traffic)
    distribution = hop_distribution(active_values)
    weighted_avg = weighted_average_hops(hop_matrix, traffic)

    loads = compute_ecmp_link_loads(traffic, path_provider)
    link_load_df = link_loads_to_dataframe(topology_name, loads)
    link_load_summary = summarize_link_loads(topology_name, loads)

    if capacity is None:
        capacity = link_load_summary["max_load"] * CAPACITY_HEADROOM

    utilization_df = add_utilization(link_load_df, capacity)
    utilization_summary = summarize_utilization(
        topology_name, utilization_df["utilization"].to_numpy()
    )

    latency_df = compute_pair_latencies(
        traffic,
        path_provider,
        loads,
        capacity=capacity,
        bandwidth_bytes_per_sec=BANDWIDTH_BYTES_PER_SEC,
        traffic_unit_bytes=TRAFFIC_UNIT_BYTES,
    )
    latency_summary = summarize_latency(topology_name, latency_df)

    path_diversity_df = compute_path_diversity_pairs(traffic, path_provider)
    path_diversity_summary = summarize_path_diversity(topology_name, path_diversity_df)

    paths: dict[str, Path] = {}

    paths["hop_matrix"] = output_dir / f"{safe_name}_hop_matrix.csv"
    np.savetxt(paths["hop_matrix"], hop_matrix, delimiter=",", fmt="%.0f")

    paths["traffic_matrix"] = output_dir / f"{safe_name}_traffic_matrix.csv"
    np.savetxt(paths["traffic_matrix"], traffic, delimiter=",", fmt="%.4f")

    paths["link_load"] = output_dir / f"{safe_name}_link_load.csv"
    link_load_df.to_csv(paths["link_load"], index=False)

    paths["link_utilization"] = output_dir / f"{safe_name}_link_utilization.csv"
    utilization_df.to_csv(paths["link_utilization"], index=False)

    paths["latency_pairs"] = output_dir / f"{safe_name}_latency_pairs.csv"
    latency_df.to_csv(paths["latency_pairs"], index=False)

    paths["path_diversity_pairs"] = save_path_diversity_pairs(
        path_diversity_df, output_dir / f"{safe_name}_path_diversity_pairs.csv"
    )

    return TopologyResult(
        topology=topology_name,
        hop_distribution=distribution,
        weighted_avg_hops=weighted_avg,
        min_active_hop=float(active_values.min()),
        max_active_hop=float(active_values.max()),
        link_load_summary=link_load_summary,
        utilization_summary=utilization_summary,
        latency_summary=latency_summary,
        path_diversity_summary=path_diversity_summary,
        paths=paths,
    )


def run_comparison(spec: ExperimentSpec) -> dict[str, TopologyResult]:
    """
    Run Fat-tree, HyperX, and Dragonfly+ for one ExperimentSpec, using a
    single shared link capacity (sized off the busiest link across all
    three), and save per-topology and combined summary/plot files under
    spec.output_dir.
    """
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = spec.parallelism_config()
    traffic = build_traffic(spec)

    provisional_max_loads = []
    for name in TOPOLOGIES:
        _, path_provider = _topology_hop_matrix_and_paths(name, cfg)
        loads = compute_ecmp_link_loads(traffic, path_provider)
        provisional_max_loads.append(summarize_link_loads(name, loads)["max_load"])

    capacity = max(provisional_max_loads) * CAPACITY_HEADROOM

    results = {
        name: run_topology_metrics(name, cfg, traffic, output_dir, capacity=capacity)
        for name in TOPOLOGIES
    }

    _save_combined_outputs(spec, cfg, output_dir, results)

    return results


def _save_combined_outputs(
    spec: ExperimentSpec,
    cfg: ParallelismConfig,
    output_dir: Path,
    results: dict[str, TopologyResult],
) -> None:
    summary_rows = []
    for name, result in results.items():
        summary_rows.append(
            {
                "topology": name,
                "total_gpus": cfg.total_gpus,
                "tp": cfg.tp,
                "dp": cfg.dp,
                "pp": cfg.pp,
                "hbi_size": cfg.hbi_size,
                "weighted_avg_hops": round(result.weighted_avg_hops, 4),
                "min_active_hop": result.min_active_hop,
                "max_active_hop": result.max_active_hop,
                "link_count": result.link_load_summary["link_count"],
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
                "single_path_traffic_exposure": result.path_diversity_summary[
                    "single_path_traffic_exposure"
                ],
                "max_path_count": result.path_diversity_summary["max_path_count"],
                "p95_path_count": result.path_diversity_summary["p95_path_count"],
                "single_path_pair_fraction": result.path_diversity_summary[
                    "single_path_pair_fraction"
                ],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary.csv", index=False)

    with open(output_dir / "active_hop_distribution.csv", "w", encoding="utf-8") as f:
        f.write("topology,hop,percentage\n")
        for name, result in results.items():
            for hop, percentage in sorted(result.hop_distribution.items()):
                f.write(f"{name},{hop},{percentage}\n")

    pd.DataFrame([r.link_load_summary for r in results.values()]).to_csv(
        output_dir / "link_load_summary.csv", index=False
    )
    pd.DataFrame([r.utilization_summary for r in results.values()]).to_csv(
        output_dir / "link_utilization_summary.csv", index=False
    )
    pd.DataFrame([r.latency_summary for r in results.values()]).to_csv(
        output_dir / "latency_percentiles.csv", index=False
    )

    routing_flexibility_columns = [
        "topology",
        "active_pairs",
        "total_active_traffic",
        "min_path_count",
        "mean_path_count",
        "traffic_weighted_path_diversity",
        "p50_path_count",
        "p90_path_count",
        "p95_path_count",
        "p99_path_count",
        "max_path_count",
        "single_path_pair_fraction",
        "single_path_traffic_exposure",
    ]
    pd.DataFrame(
        [r.path_diversity_summary for r in results.values()], columns=routing_flexibility_columns
    ).to_csv(output_dir / "routing_flexibility_summary.csv", index=False)

    plot_grouped_bar(
        {name: r.hop_distribution for name, r in results.items()},
        output_dir / "active_hop_distribution.png",
        title=f"{spec.name}: Active Hop Distribution by Topology",
        xlabel="Hop count",
        ylabel="Fraction of active pairs",
    )

    ranked = sorted(results.items(), key=lambda item: item[1].weighted_avg_hops)

    plot_single_bar(
        {name: r.weighted_avg_hops for name, r in ranked},
        output_dir / "weighted_average_hops.png",
        title=f"{spec.name}: Traffic-Weighted Average Hops by Topology",
        ylabel="Weighted average hops",
    )

    plot_cdf(
        {
            name: pd.read_csv(r.paths["link_load"])["load"].to_numpy()
            for name, r in results.items()
        },
        output_dir / "link_load_cdf.png",
        title=f"{spec.name}: Link-Load CDF by Topology",
        xlabel="Link load",
    )

    plot_cdf(
        {
            name: pd.read_csv(r.paths["latency_pairs"])["latency_ms"].to_numpy()
            for name, r in results.items()
        },
        output_dir / "latency_cdf.png",
        title=f"{spec.name}: Congestion-Aware Latency CDF by Topology",
        xlabel="Latency (ms)",
    )

    for label, key in [
        ("Mean", "mean_latency_ms"),
        ("p50", "p50_latency_ms"),
        ("p90", "p90_latency_ms"),
        ("p95", "p95_latency_ms"),
    ]:
        plot_single_bar(
            {name: r.latency_summary[key] for name, r in results.items()},
            output_dir / f"latency_{label.lower()}_bar.png",
            title=f"{spec.name}: {label} Congestion-Aware Latency by Topology",
            ylabel=f"{label} latency (ms)",
        )

    for label, key in [("p99", "p99_latency_ms"), ("p100", "p100_latency_ms")]:
        plot_tail_bar(
            {name: r.latency_summary[key] for name, r in results.items()},
            output_dir / f"latency_{label}_bar.png",
            title=f"{spec.name}: {label} Latency by Topology",
            ylabel=f"{label} latency (ms)",
        )

    plot_grouped_metric_bar(
        {
            name: {
                "p50": r.latency_summary["p50_latency_ms"],
                "p90": r.latency_summary["p90_latency_ms"],
                "p95": r.latency_summary["p95_latency_ms"],
                "p99": r.latency_summary["p99_latency_ms"],
                "p100": r.latency_summary["p100_latency_ms"],
            }
            for name, r in results.items()
        },
        output_dir / "latency_percentile_bars.png",
        title=f"{spec.name}: Latency Percentiles by Topology",
        xlabel="Latency percentile",
        ylabel="Latency (ms)",
    )

    plot_path_diversity_bar(
        {
            name: r.path_diversity_summary["traffic_weighted_path_diversity"]
            for name, r in results.items()
        },
        output_dir / "routing_flexibility_path_diversity.png",
        title=f"{spec.name}: Traffic-Weighted ECMP Path Diversity by Topology",
    )

    plot_single_path_exposure_bar(
        {
            name: r.path_diversity_summary["single_path_traffic_exposure"]
            for name, r in results.items()
        },
        output_dir / "routing_flexibility_single_path_exposure.png",
        title=f"{spec.name}: Single-Path Traffic Exposure by Topology",
    )

    plot_path_count_cdf(
        {
            name: pd.read_csv(r.paths["path_diversity_pairs"])["path_count"].to_numpy()
            for name, r in results.items()
        },
        output_dir / "routing_flexibility_path_count_cdf.png",
        title=f"{spec.name}: ECMP Path Count CDF by Topology",
    )
