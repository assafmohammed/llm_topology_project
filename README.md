# LLM Topology Project

This project compares datacenter topologies for LLM-like training traffic.

Current first milestone:
- Build HyperX for 32 GPUs.
- Preserve 8-GPU HBI domains.
- Generate synthetic TP/DP/PP traffic.
- Compute shortest-hop distances for active communication pairs.

Run:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python experiments/run_hyperx32.py
pytest -q
```

## 32-GPU baseline

Commands:

```bash
source /home/amn/miniforge3/etc/profile.d/conda.sh
conda activate simai310
cd /home/amn/llm_topology_project

python experiments/run_hyperx32.py
python experiments/run_fat_tree32.py
python experiments/run_dragonfly32.py
python experiments/run_compare32.py
```

Generated comparison outputs:
- `results/compare32_summary.csv`
- `results/compare32_active_hop_distribution.csv`
- `results/compare32_active_hop_distribution.png`
- `results/compare32_weighted_average_hops.png`

Additional latency plots:
- `compare32_latency_mean_bar.png`
- `compare32_latency_p50_bar.png`
- `compare32_latency_p90_bar.png`
- `compare32_latency_p95_bar.png`
- `compare32_latency_percentile_bars.png`

`p99` and `p100` may be identical in the 32-GPU case because dominant TP traffic is
inside HBI for all topologies. Mean/p50/p90 latency are useful debug indicators
before scaling.