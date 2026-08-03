# Code Overview

## 1. Project Goal

This project compares datacenter network topologies (HyperX, Fat-tree, Dragonfly+)
for LLM training communication workloads. It builds topology graphs, generates or
loads traffic matrices, computes communication metrics (hops, link load, latency,
routing flexibility), and saves comparison results as CSVs and plots.

## 2. Main Folders

### src/llm_topology/topologies/

Topology builders and topology-specific hop/path logic.

- **common.py** — shared building blocks: `ParallelismConfig` (total_gpus/TP/DP/PP/HBI
  size), node-naming helpers (`gpu_node`, `switch_node`), and `add_link`.
- **hyperx.py** — builds the HyperX graph, maps a GPU id to its switch coordinate,
  and has a formula-based `hyperx_hop_distance`/`hyperx_hop_matrix` (fast, avoids
  running graph shortest-paths at large scale).
- **fat_tree.py** — builds the Fat-tree graph (HBI domains + rails + spines) and a
  formula-based hop distance/matrix.
- **dragonfly_plus.py** — builds the Dragonfly+ graph (HBI domains + leaves + spines,
  grouped by pipeline stage) and its formula-based hop distance/matrix.

### src/llm_topology/traffic/

Traffic matrix generation and loading.

- **synthetic.py** — generates the synthetic LLM-like traffic matrix (TP ring traffic
  dominant, DP traffic smaller, PP traffic smallest).
- **matrix_loader.py** — loads/saves a traffic matrix from `.npy`/`.csv`/`.txt`.
- **aicb.py** — a conservative parser for AICB/SimAI-style traffic traces. It only
  accepts unambiguous `src dst bytes` rows and raises `NotImplementedError` if the
  file doesn't match, instead of guessing or faking numbers.

### src/llm_topology/metrics/

Metric calculations, all operating on a hop matrix / traffic matrix / path provider.

- **hops.py** — hop matrix via NetworkX shortest paths (used for HyperX's graph),
  active hop distribution, and traffic-weighted average hops.
- **routing.py** — topology-specific path providers (`hyperx_paths`, `fat_tree_paths`,
  `dragonfly_paths`) that return the valid equal-cost paths between two GPUs.
- **link_load.py** — routes active traffic over the path providers (ECMP-style, split
  evenly across equal-cost paths) to get per-link load, plus link-load/utilization
  summaries.
- **latency.py** — congestion-aware latency using an M/M/1-style approximation
  (`1 / (1 - utilization)` per link on the path) and latency percentile summaries.
- **path_diversity.py** — the beyond-paper routing-flexibility metrics: how many
  equal-cost paths exist per active pair, traffic-weighted path diversity, and
  single-path traffic exposure.

### src/llm_topology/viz/

Plotting helpers. There is no separate `routing_flexibility.py` — those plots live
in `cdf.py`.

- **heatmap.py** — GPU x GPU heatmaps for hop matrices and traffic matrices.
- **cdf.py** — bar charts (hop distribution, single-value comparisons, grouped
  metric bars) and CDF plots (link load, latency, ECMP path count), including the
  routing-flexibility bar/CDF plots.

### src/llm_topology/experiments/

Reusable experiment pipeline code.

- **pipeline.py** — `ExperimentSpec` (config for one experiment), `TopologyResult`
  (all computed metrics for one topology), `build_traffic` (synthetic/npy/csv/aicb),
  `run_topology_metrics` (compute + save one topology's files), and `run_comparison`
  (run all three topologies with a shared link capacity and save the combined
  summaries/plots). This is what the scripts in `experiments/` call into.

### experiments/

Runnable scripts (not part of the installed package; each inserts `src/` onto
`sys.path` itself).

- **run_hyperx32.py / run_fat_tree32.py / run_dragonfly32.py** — the original
  single-topology 32-GPU scripts.
- **run_compare32.py** — the 32-GPU debug baseline. Keeps its own older
  `compare_topologies` function (for backward compatibility with earlier tests) and
  also calls the shared `run_comparison` pipeline.
- **run_compare128.py / run_compare1024.py** — scaling experiments (128 GPUs; 1024
  GPUs across three TP/DP/PP configs), using `run_comparison`.
- **run_paper_sweep.py** — sweeps many TP/PP combinations at a given GPU count.
- **run_all.py** — convenience script that runs 32/128/1024 (and optionally the
  sweep) in one command.
- **generate_report_tables.py** — reads an aggregate summary CSV and writes a
  per-topology summary, a best-topology-per-metric table, and a short markdown
  paper-alignment note.

### tests/

Unit tests and sanity checks: topology construction, hop-distance formulas, path
providers, link-load/latency/path-diversity metrics, scaling configs, traffic
loading, and pipeline output files. Two files are currently empty/unused
placeholders: `test_hyperx32.py` and `test_synthetic_matrix.py`.

### results/

Generated CSVs and plots go here (e.g. `results/compare32/`,
`results/compare128_tp8_dp4_pp4/`, `results/compare1024_*/`). Currently also
contains an older `results-32/` snapshot from before the shared pipeline existed.

### traffic/ (top-level)

Empty placeholder folder (just `.gitkeep`) for saved/generated traffic matrices,
separate from `src/llm_topology/traffic/` which holds the code.

### aicb_outputs/

Empty placeholder folder (just `.gitkeep`), reserved for AICB/SimAI workload output
files.

### docs/

Project documentation (this file).

## 3. Code Flow

1. Define experiment config: total_gpus, TP, DP, PP, HBI size (`ParallelismConfig` /
   `ExperimentSpec`).
2. Generate or load a traffic matrix: `S[i,j]` is the traffic volume from GPU i to
   GPU j (synthetic, or loaded from `.npy`/`.csv`/AICB).
3. Build each topology: HyperX, Fat-tree, Dragonfly+.
4. Compute hop metrics: hop matrix, active hop distribution, weighted average hops.
5. Compute link-load metrics: route active traffic through topology-specific paths
   and add up load per link.
6. Compute utilization: `utilization = load / capacity`.
7. Compute latency: traffic size, bandwidth, and a congestion factor from
   utilization.
8. Compute routing-flexibility metrics: path diversity and single-path traffic
   exposure.
9. Save CSVs and plots to `results/`.

## 4. Important Design Notes

- HBI domain size is 8 GPUs.
- TP/DP/PP define the communication pattern (TP traffic dominant, DP smaller, PP
  smallest).
- "Active pairs" are pairs where `traffic_matrix[i,j] > 0`.
- Fat-tree and Dragonfly+ use dedicated path providers (not generic NetworkX
  shortest paths) to avoid routing through an HBI link as an invalid shortcut
  between GPUs.
- HyperX's hop distance follows a formula based on TP/DP/PP dimensions instead of
  running shortest-paths, so it stays fast at 128/1024 GPUs.
- 32 GPUs is the debug baseline; 128/1024 are scaling experiments.

## 5. How to Run

```bash
source /home/amn/miniforge3/etc/profile.d/conda.sh
conda activate simai310
cd /home/amn/llm_topology_project

python experiments/run_compare32.py
python experiments/run_compare128.py
python experiments/run_compare1024.py
python experiments/run_all.py
```

Run tests:

```bash
pytest -q
```

## 6. Main Outputs

- `summary.csv`
- `active_hop_distribution.csv`
- `link_load_summary.csv`
- `link_utilization_summary.csv`
- `latency_percentiles.csv`
- `routing_flexibility_summary.csv`
- hop heatmaps
- link-load CDF
- latency CDF
- latency percentile plots
- routing-flexibility bar/CDF plots

## 7. What to Read First

1. `experiments/run_compare32.py`
2. `src/llm_topology/experiments/pipeline.py`
3. `src/llm_topology/topologies/common.py`
4. `src/llm_topology/topologies/hyperx.py`
5. `src/llm_topology/traffic/synthetic.py`
6. `src/llm_topology/metrics/hops.py`
7. `src/llm_topology/metrics/link_load.py`
8. `src/llm_topology/metrics/latency.py`
9. `src/llm_topology/metrics/path_diversity.py`
