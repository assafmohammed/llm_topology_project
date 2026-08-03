from __future__ import annotations

import pandas as pd

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison


def test_pipeline_output_csv_columns(tmp_path):
    spec = ExperimentSpec(
        name="test_outputs",
        total_gpus=32,
        tp=8,
        dp=2,
        pp=2,
        hbi_size=8,
        traffic_mode="synthetic",
        output_dir=tmp_path / "test_outputs",
    )

    run_comparison(spec)
    output_dir = spec.output_dir

    summary = pd.read_csv(output_dir / "summary.csv")
    for column in [
        "topology",
        "total_gpus",
        "tp",
        "dp",
        "pp",
        "hbi_size",
        "weighted_avg_hops",
        "min_active_hop",
        "max_active_hop",
        "link_count",
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
    ]:
        assert column in summary.columns
    assert len(summary) == 3

    link_load_summary = pd.read_csv(output_dir / "link_load_summary.csv")
    for column in [
        "topology",
        "link_count",
        "min_load",
        "mean_load",
        "p50_load",
        "p90_load",
        "p95_load",
        "p99_load",
        "max_load",
    ]:
        assert column in link_load_summary.columns

    latency_percentiles = pd.read_csv(output_dir / "latency_percentiles.csv")
    for column in [
        "topology",
        "active_pairs",
        "mean_latency_ms",
        "p50_latency_ms",
        "p90_latency_ms",
        "p95_latency_ms",
        "p97_latency_ms",
        "p99_latency_ms",
        "p100_latency_ms",
    ]:
        assert column in latency_percentiles.columns
