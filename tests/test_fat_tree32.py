from __future__ import annotations

from llm_topology.metrics.hops import active_hop_values, hop_distribution
from llm_topology.topologies.common import ParallelismConfig
from llm_topology.topologies.fat_tree import (
    build_fat_tree,
    fat_tree_hop_distance,
    fat_tree_hop_matrix,
)
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic


def make_cfg() -> ParallelismConfig:
    return ParallelismConfig(total_gpus=32, tp=8, dp=2, pp=2, hbi_size=8)


def test_node_counts():
    graph = build_fat_tree(make_cfg())

    gpu_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "gpu")
    rail_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "rail")
    spine_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "spine")

    assert gpu_count == 32
    assert rail_count == 8
    assert spine_count == 2


def test_fat_tree_hop_distance():
    cfg = make_cfg()

    assert fat_tree_hop_distance(0, 0, cfg) == 0
    assert fat_tree_hop_distance(0, 7, cfg) == 1
    assert fat_tree_hop_distance(0, 8, cfg) == 2
    assert fat_tree_hop_distance(0, 16, cfg) == 2
    assert fat_tree_hop_distance(0, 9, cfg) == 4


def test_fat_tree_hop_matrix_shape():
    cfg = make_cfg()

    assert fat_tree_hop_matrix(cfg).shape == (32, 32)


def test_active_traffic_hops():
    cfg = make_cfg()

    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )

    hop_matrix = fat_tree_hop_matrix(cfg)
    active_values = active_hop_values(hop_matrix, traffic)

    assert active_values.max() <= 2
    assert hop_distribution(active_values) == {1: 0.5, 2: 0.5}
