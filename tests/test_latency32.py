from __future__ import annotations

import numpy as np

from llm_topology.metrics.latency import compute_pair_latencies, summarize_latency
from llm_topology.metrics.link_load import compute_ecmp_link_loads, summarize_link_loads
from llm_topology.metrics.routing import dragonfly_paths, fat_tree_paths, hyperx_paths
from llm_topology.topologies.common import ParallelismConfig
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic


def make_cfg() -> ParallelismConfig:
    return ParallelismConfig(total_gpus=32, tp=8, dp=2, pp=2, hbi_size=8)


def make_traffic(cfg: ParallelismConfig):
    return generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )


PROVIDERS = {
    "HyperX": lambda i, j, cfg: hyperx_paths(i, j, cfg),
    "Fat-tree": lambda i, j, cfg: fat_tree_paths(i, j, cfg, num_spines=2),
    "Dragonfly+": lambda i, j, cfg: dragonfly_paths(i, j, cfg, spines_per_group=2),
}


def test_compute_pair_latencies_one_row_per_active_pair():
    cfg = make_cfg()
    traffic = make_traffic(cfg)

    active_pair_count = int(np.count_nonzero(traffic > 0))

    for name, provider_factory in PROVIDERS.items():
        provider = lambda i, j, cfg=cfg, factory=provider_factory: factory(i, j, cfg)
        loads = compute_ecmp_link_loads(traffic, provider)
        max_load = summarize_link_loads(name, loads)["max_load"]
        capacity = max_load * 1.2

        latency_df = compute_pair_latencies(
            traffic,
            provider,
            loads,
            capacity=capacity,
        )

        assert len(latency_df) == active_pair_count
        assert (latency_df["latency_ms"] > 0).all()


def test_no_utilization_at_or_above_one():
    cfg = make_cfg()
    traffic = make_traffic(cfg)

    provider = lambda i, j: fat_tree_paths(i, j, cfg, num_spines=2)
    loads = compute_ecmp_link_loads(traffic, provider)
    max_load = summarize_link_loads("Fat-tree", loads)["max_load"]
    capacity = max_load * 1.2

    for load in loads.values():
        assert load / capacity < 1


def test_latency_summary_has_required_percentiles():
    cfg = make_cfg()
    traffic = make_traffic(cfg)

    provider = lambda i, j: fat_tree_paths(i, j, cfg, num_spines=2)
    loads = compute_ecmp_link_loads(traffic, provider)
    max_load = summarize_link_loads("Fat-tree", loads)["max_load"]
    capacity = max_load * 1.2

    latency_df = compute_pair_latencies(traffic, provider, loads, capacity=capacity)
    summary = summarize_latency("Fat-tree", latency_df)

    for key in ["p50_latency_ms", "p95_latency_ms", "p97_latency_ms", "p99_latency_ms", "p100_latency_ms"]:
        assert key in summary
