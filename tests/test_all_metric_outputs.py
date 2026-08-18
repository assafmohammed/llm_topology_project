from __future__ import annotations

import numpy as np
import pandas as pd

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison

EXPERIMENT_ROOT_FILES = [
    # Section 5.1
    "traffic_analysis_summary.csv",
    "traffic_matrix.csv",
    "traffic_heatmap.png",
    # Section 5.2
    "summary.csv",
    "active_hop_distribution.csv",
    "active_hop_distribution.png",
    "weighted_average_hops.png",
    "hop_summary.csv",
    # Section 5.3
    "link_load_summary.csv",
    "link_utilization_summary.csv",
    "link_load_cdf.png",
    # Section 5.4
    "latency_percentiles.csv",
    "latency_cdf.png",
    "latency_mean_bar.png",
    "latency_p50_bar.png",
    "latency_p90_bar.png",
    "latency_p95_bar.png",
    "latency_p99_bar.png",
    "latency_p100_bar.png",
    "latency_percentile_bars.png",
    # Section 5.5
    "routing_flexibility_summary.csv",
    "routing_robustness_summary.csv",
    "routing_path_diversity_bar.png",
    "routing_single_path_exposure_bar.png",
    "routing_path_count_cdf.png",
    "critical_link_dependency_bar.png",
    "load_imbalance_bar.png",
]

PER_TOPOLOGY_SUFFIXES = [
    "hop_matrix.csv",
    "hop_heatmap.png",
    "traffic_matrix.csv",
    "link_load.csv",
    "link_utilization.csv",
    "latency_pairs.csv",
    "path_diversity_pairs.csv",
]

SAFE_NAMES = ["fat_tree", "hyperx", "dragonfly"]


def make_spec(tmp_path):
    return ExperimentSpec(
        name="test_all_metrics",
        total_gpus=32,
        tp=8,
        dp=2,
        pp=2,
        hbi_size=8,
        traffic_mode="synthetic",
        output_dir=tmp_path / "test_all_metrics",
    )


def test_hop_metrics_are_sane(tmp_path):
    results = run_comparison(make_spec(tmp_path))

    for result in results.values():
        assert result.weighted_avg_hops > 0
        assert len(result.hop_distribution) > 0


def test_link_load_metrics_are_sane(tmp_path):
    spec = make_spec(tmp_path)
    results = run_comparison(spec)

    for name, result in results.items():
        safe_name = {"Fat-tree": "fat_tree", "HyperX": "hyperx", "Dragonfly+": "dragonfly"}[name]
        link_load_df = pd.read_csv(spec.output_dir / f"{safe_name}_link_load.csv")

        assert len(link_load_df) > 0
        assert result.link_load_summary["max_load"] > 0


def test_latency_metrics_are_sane(tmp_path):
    spec = make_spec(tmp_path)
    results = run_comparison(spec)

    for name, result in results.items():
        safe_name = {"Fat-tree": "fat_tree", "HyperX": "hyperx", "Dragonfly+": "dragonfly"}[name]
        latency_df = pd.read_csv(spec.output_dir / f"{safe_name}_latency_pairs.csv")

        assert len(latency_df) == result.latency_summary["active_pairs"]
        assert (latency_df["latency_ms"] > 0).all()

        summary = result.latency_summary
        assert (
            summary["p100_latency_ms"]
            >= summary["p99_latency_ms"]
            >= summary["p95_latency_ms"]
            >= summary["p90_latency_ms"]
            >= summary["p50_latency_ms"]
        )


def test_all_required_output_files_exist(tmp_path):
    spec = make_spec(tmp_path)
    run_comparison(spec)
    output_dir = spec.output_dir

    for filename in EXPERIMENT_ROOT_FILES:
        assert (output_dir / filename).exists(), f"missing {filename}"

    for safe_name in SAFE_NAMES:
        for suffix in PER_TOPOLOGY_SUFFIXES:
            path = output_dir / f"{safe_name}_{suffix}"
            assert path.exists(), f"missing {path.name}"
