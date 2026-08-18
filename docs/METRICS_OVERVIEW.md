# Metrics Overview

This project reproduces the paper's topology-comparison metrics (Sections 5.1-5.4)
and adds one new section (5.5) that isn't in the paper. Every metric is computed
per topology (Fat-tree, HyperX, Dragonfly+) over the same synthetic (or loaded)
traffic matrix, so the numbers are directly comparable.

## Paper Metrics

### 5.1 Transport matrix analysis

What: classifies the traffic matrix `S[i,j]` into TP/DP/PP/other traffic (based on
which single GPU index differs between src and dst), and reports active pairs,
total volume, traffic density, and per-pair traffic percentiles.

Why useful: shows what the workload actually looks like before any topology is
involved — traffic is topology-independent, so this is computed once per
experiment. Implemented in `metrics/traffic_analysis.py`.

### 5.2 Effective hop-count analysis

What: hop distance between every active pair, the resulting hop-count
distribution, and the traffic-weighted average hop count. Lower is better.

Why useful: hop count is the simplest proxy for how "close" the traffic pattern
is to the fabric. Fat-tree and Dragonfly+ use topology-specific formulas (not
generic shortest paths) so an HBI link is never counted as a shortcut into
another GPU's traffic. HyperX also uses a formula (exact, not an approximation)
so it stays fast at 1024 GPUs. Implemented in `metrics/hops.py` and the
topology modules' `*_hop_distance`/`*_hop_matrix` functions.

### 5.3 ECMP link-load distribution

What: routes every active pair's traffic over its equal-cost paths (splitting
evenly across paths), sums load per link, and reports the load distribution
(min/mean/std/percentiles/max) plus utilization (`load / capacity`).

Why useful: hop count alone doesn't say whether traffic is spread evenly across
the fabric — this does. Implemented in `metrics/link_load.py`, using the
topology-specific path providers in `metrics/routing.py`.

### 5.4 Congestion-aware GPU-to-GPU latency

What: `D_ij = S_ij * (1/bw) * sum_over_path_edges[1 / (1 - rho_e)]`, where
`rho_e = load_e / capacity_e`. Reports mean/percentile/max latency per topology,
plus one row per active pair. Fails clearly instead of silently continuing if any
link's utilization would reach 1.0. Implemented in `metrics/latency.py`.

## New Beyond-Paper Metrics

### 5.5 Routing robustness and flexibility

Hop count, link load, and latency describe the *shortest* or *average* path
behavior. They don't say whether a topology actually has more than one way to
route each flow, or whether a few links carry a disproportionate share of the
traffic. Section 5.5 answers that, using the same path providers as 5.3/5.4.

**1. Traffic-weighted ECMP path diversity**
`sum(S[i,j] * num_paths(i,j)) / sum(S[i,j])`
Higher is better. Weights each active pair's path count by how much traffic it
actually carries, so heavy flows count more than light ones.

**2. Single-path traffic exposure**
`sum(S[i,j] where num_paths(i,j) == 1) / sum(S[i,j])`
Lower is better. The fraction of traffic that has no alternate route at all — if
that one path gets congested or fails, that traffic has nowhere else to go.

**3. Critical-link dependency**
`top_1_link_dependency = max_link_load / total_routed_load` (and the same for the
busiest 5%/10% of links). Lower is better. Measures how concentrated the routed
traffic is on a small number of links — a high value means a handful of links
matter disproportionately for both performance and fault tolerance.

**4. Load imbalance coefficient**
`std(link_loads) / mean(link_loads)`, plus max/p95/p99-to-mean ratios. Lower is
better. A pure measure of how evenly load is spread across all links, independent
of any capacity assumption.

Implemented in `metrics/path_diversity.py` (metrics 1-2) and
`metrics/robustness.py` (metrics 3-4).

## Why This Matters

Hop count, link load, and latency can look similar across topologies while
hiding very different routing structure underneath. A topology with the same
weighted-average hop count as another might route almost all of its traffic over
a single path, making it far more fragile under congestion or a link failure.
The 5.5 metrics make that difference visible.
