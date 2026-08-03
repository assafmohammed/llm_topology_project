from __future__ import annotations

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison


def test_run_comparison_32gpu_produces_expected_outputs(tmp_path):
    spec = ExperimentSpec(
        name="test32",
        total_gpus=32,
        tp=8,
        dp=2,
        pp=2,
        hbi_size=8,
        traffic_mode="synthetic",
        output_dir=tmp_path / "test32",
    )

    results = run_comparison(spec)

    assert set(results.keys()) == {"Fat-tree", "HyperX", "Dragonfly+"}

    output_dir = spec.output_dir
    assert (output_dir / "summary.csv").exists()
    assert (output_dir / "active_hop_distribution.csv").exists()
    assert (output_dir / "link_load_summary.csv").exists()
    assert (output_dir / "link_utilization_summary.csv").exists()
    assert (output_dir / "latency_percentiles.csv").exists()

    for result in results.values():
        for key in ["p50_latency_ms", "p90_latency_ms", "p95_latency_ms", "p99_latency_ms", "p100_latency_ms"]:
            assert key in result.latency_summary
