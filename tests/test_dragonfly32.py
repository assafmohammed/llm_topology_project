from __future__ import annotations

from llm_topology.metrics.hops import active_hop_values, hop_distribution
from llm_topology.topologies.common import ParallelismConfig
from llm_topology.topologies.dragonfly_plus import (
    build_dragonfly_plus,
    dragonfly_hop_distance,
    dragonfly_hop_matrix,
)
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic


def make_cfg() -> ParallelismConfig:
    return ParallelismConfig(total_gpus=32, tp=8, dp=2, pp=2, hbi_size=8)


def test_node_counts():
    graph = build_dragonfly_plus(make_cfg())

    gpu_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "gpu")
    leaf_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "leaf")
    spine_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "spine")

    assert gpu_count == 32
    assert leaf_count == 4
    assert spine_count == 4


def test_dragonfly_hop_distance():
    cfg = make_cfg()

    assert dragonfly_hop_distance(0, 0, cfg) == 0
    assert dragonfly_hop_distance(0, 7, cfg) == 1
    assert dragonfly_hop_distance(0, 8, cfg) == 4
    assert dragonfly_hop_distance(0, 16, cfg) == 5
    assert dragonfly_hop_distance(0, 24, cfg) == 5


def test_dragonfly_hop_matrix_shape():
    cfg = make_cfg()

    assert dragonfly_hop_matrix(cfg).shape == (32, 32)


def test_active_traffic_hops():
    cfg = make_cfg()

    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )

    hop_matrix = dragonfly_hop_matrix(cfg)
    active_values = active_hop_values(hop_matrix, traffic)
    distribution = hop_distribution(active_values)

    assert active_values.max() <= 5
    assert set(distribution.keys()) == {1, 4, 5}
