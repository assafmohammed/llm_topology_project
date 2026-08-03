from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison


def main() -> None:
    spec = ExperimentSpec(
        name="compare128_tp8_dp4_pp4",
        total_gpus=128,
        tp=8,
        dp=4,
        pp=4,
        hbi_size=8,
        traffic_mode="synthetic",
        output_dir=ROOT / "results" / "compare128_tp8_dp4_pp4",
    )

    results = run_comparison(spec)

    print(f"{spec.name}: TP={spec.tp}, DP={spec.dp}, PP={spec.pp}, HBI={spec.hbi_size}")
    for name, result in sorted(results.items(), key=lambda item: item[1].weighted_avg_hops):
        print(f"{name}:")
        print(f"  Active hop distribution: {result.hop_distribution}")
        print(f"  Weighted avg hops: {result.weighted_avg_hops:.4f}")
        print(f"  Max link load: {result.link_load_summary['max_load']:.4f}")
        print(f"  Max utilization: {result.utilization_summary['max_utilization']:.4f}")
        print(f"  p99 latency: {result.latency_summary['p99_latency_ms']:.4f} ms")

    print()
    print(f"Saved output tree under: {spec.output_dir}")


if __name__ == "__main__":
    main()
