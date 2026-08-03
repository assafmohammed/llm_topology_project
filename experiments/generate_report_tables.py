from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

METRIC_COLUMNS = [
    "weighted_avg_hops",
    "max_link_load",
    "p95_link_load",
    "p99_link_load",
    "max_utilization",
    "mean_latency_ms",
    "p50_latency_ms",
    "p90_latency_ms",
    "p95_latency_ms",
    "p99_latency_ms",
    "p100_latency_ms",
    "traffic_weighted_path_diversity",
    "single_path_traffic_exposure",
]

# Lower is better for every metric here (hops, load, utilization, latency,
# single-path exposure). traffic_weighted_path_diversity is higher-is-better,
# so it is intentionally excluded from this idxmin()-based comparison.
BEST_METRICS = [
    "weighted_avg_hops",
    "max_link_load",
    "p95_link_load",
    "p99_link_load",
    "max_utilization",
    "mean_latency_ms",
    "p99_latency_ms",
    "single_path_traffic_exposure",
]


def load_input_summary(results_dir: Path) -> pd.DataFrame:
    candidates = [
        results_dir / "compare1024_all_configs_summary.csv",
        results_dir / "paper_sweep_summary.csv",
    ]
    frames = [pd.read_csv(path) for path in candidates if path.exists()]

    if not frames:
        raise FileNotFoundError(
            "No input summary found. Run experiments/run_compare1024.py or "
            "experiments/run_paper_sweep.py first, so one of "
            f"{[str(p) for p in candidates]} exists."
        )

    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["experiment", "topology"]
    )


def build_topology_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.groupby("topology")[METRIC_COLUMNS].mean().reset_index()
    summary.insert(1, "experiment_count", df.groupby("topology").size().to_numpy())
    return summary


def build_best_by_metric(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for experiment, group in df.groupby("experiment"):
        for metric in BEST_METRICS:
            best_row = group.loc[group[metric].idxmin()]
            rows.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "best_topology": best_row["topology"],
                    "value": best_row[metric],
                }
            )
    return pd.DataFrame(rows, columns=["experiment", "metric", "best_topology", "value"])


def overall_best_topology(topology_summary: pd.DataFrame, metric: str) -> tuple[str, float]:
    best_row = topology_summary.loc[topology_summary[metric].idxmin()]
    return str(best_row["topology"]), float(best_row[metric])


def build_alignment_notes(topology_summary: pd.DataFrame) -> str:
    best_hops_topology, best_hops_value = overall_best_topology(
        topology_summary, "weighted_avg_hops"
    )
    best_load_topology, best_load_value = overall_best_topology(
        topology_summary, "max_link_load"
    )
    best_latency_topology, best_latency_value = overall_best_topology(
        topology_summary, "p99_latency_ms"
    )

    return f"""# Paper Alignment Notes

## 1. What is implemented

- Fat-tree, HyperX, Dragonfly+ topologies
- TP/DP/PP synthetic traffic (dominant TP, smaller DP, smallest PP)
- Active hop distribution and traffic-weighted average hops
- ECMP-style link-load routing and link-load CDF
- Link utilization
- Congestion-aware latency (M/M/1-style approximation) and latency CDF
- Latency percentiles (p50/p90/p95/p97/p99/p100)
- Beyond-paper routing-flexibility analysis: traffic-weighted ECMP path
  diversity and single-path traffic exposure

## 2. What matches the paper

- HBI size = 8
- HyperX dimensions use TP/8, DP, PP
- Effective hop distance is measured over active communication pairs only,
  not all-pairs
- ECMP link-load idea: traffic is split evenly across equal-cost paths
- M/M/1-style latency formula: sum of 1 / (1 - utilization) over path edges
- Bandwidth = 400 Gb/s = 50 GB/s (50e9 bytes/sec)

## 3. What is simplified

- Synthetic TP/DP/PP traffic is used instead of full AICB/SimAI traces
  unless a real traffic matrix is supplied via traffic_mode="npy"/"csv"/"aicb"
- Dragonfly+ construction is simplified (fixed leaf/spine counts per group,
  single inter-group spine link) rather than a full dragonfly all-to-all
  group interconnect
- Capacity model is simplified: one shared capacity per experiment, sized
  as the busiest link across all topologies times 1.2, unless configured
  otherwise
- The 32-GPU case is only a debug baseline, not a main paper scale

## 4. Final result interpretation

Across the experiments summarized in this report:

- Lowest average hops: **{best_hops_topology}**
  (mean weighted average hops = {best_hops_value:.4f})
- Best load balance (lowest average max link load): **{best_load_topology}**
  (mean max link load = {best_load_value:.4f})
- Best tail latency (lowest average p99 latency): **{best_latency_topology}**
  (mean p99 latency = {best_latency_value:.4f} ms)

In small, low-TP configurations (e.g. the 32-GPU debug case with TP=8),
p99 and p100 latency can be identical across topologies because tail
latency is dominated by local TP traffic that stays inside the HBI domain
for every topology. This makes mean/p50/p90 latency more informative at
small scale; tail latency differences become meaningful once TP grows
past the HBI size and DP/PP traffic starts crossing the fabric more often.

See `report_topology_summary.csv` for per-topology averages across all
experiments, and `report_best_by_metric.csv` for the winning topology on
each metric, per experiment.
"""


def main() -> None:
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    df = load_input_summary(results_dir)

    topology_summary = build_topology_summary(df)
    best_by_metric = build_best_by_metric(df)
    notes = build_alignment_notes(topology_summary)

    topology_summary_path = results_dir / "report_topology_summary.csv"
    best_by_metric_path = results_dir / "report_best_by_metric.csv"
    notes_path = results_dir / "report_paper_alignment_notes.md"

    topology_summary.to_csv(topology_summary_path, index=False)
    best_by_metric.to_csv(best_by_metric_path, index=False)
    notes_path.write_text(notes, encoding="utf-8")

    print(f"Loaded {len(df)} experiment/topology rows from {results_dir}")
    print("Saved:")
    print(topology_summary_path)
    print(best_by_metric_path)
    print(notes_path)


if __name__ == "__main__":
    main()
