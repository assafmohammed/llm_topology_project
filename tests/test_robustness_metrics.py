from __future__ import annotations

from llm_topology.metrics.link_load import compute_ecmp_link_loads
from llm_topology.metrics.robustness import (
    summarize_critical_link_dependency,
    summarize_load_imbalance,
)
from llm_topology.metrics.routing import dragonfly_paths, fat_tree_paths, hyperx_paths
from llm_topology.topologies.common import ParallelismConfig
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic

PROVIDERS = {
    "HyperX": lambda i, j, cfg: hyperx_paths(i, j, cfg),
    "Fat-tree": lambda i, j, cfg: fat_tree_paths(i, j, cfg, num_spines=2),
    "Dragonfly+": lambda i, j, cfg: dragonfly_paths(i, j, cfg, spines_per_group=2),
}


def make_cfg() -> ParallelismConfig:
    return ParallelismConfig(total_gpus=32, tp=8, dp=2, pp=2, hbi_size=8)


def make_loads(name: str, cfg: ParallelismConfig):
    factory = PROVIDERS[name]
    provider = lambda i, j, cfg=cfg, factory=factory: factory(i, j, cfg)
    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )
    return compute_ecmp_link_loads(traffic, provider)


def test_critical_link_dependency_ranges_and_ordering():
    cfg = make_cfg()

    for name in PROVIDERS:
        loads = make_loads(name, cfg)
        summary = summarize_critical_link_dependency(name, loads)

        assert 0 <= summary["top_1_link_dependency"] <= 1
        assert 0 <= summary["top_5_percent_link_dependency"] <= 1
        assert 0 <= summary["top_10_percent_link_dependency"] <= 1
        assert (
            summary["top_10_percent_link_dependency"]
            >= summary["top_5_percent_link_dependency"]
            >= summary["top_1_link_dependency"]
        )


def test_load_imbalance_ranges():
    cfg = make_cfg()

    for name in PROVIDERS:
        loads = make_loads(name, cfg)
        summary = summarize_load_imbalance(name, loads)

        assert summary["load_imbalance_coefficient"] >= 0
        assert summary["max_to_mean_load_ratio"] >= 1
