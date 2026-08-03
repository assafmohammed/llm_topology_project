from __future__ import annotations

from llm_topology.topologies.common import ParallelismConfig
from llm_topology.topologies.hyperx import hyperx_hop_distance


def test_hyperx_formula_matches_known_32gpu_values():
    cfg = ParallelismConfig(total_gpus=32, tp=8, dp=2, pp=2, hbi_size=8)

    assert hyperx_hop_distance(0, 0, cfg) == 0
    assert hyperx_hop_distance(0, 7, cfg) == 1
    assert hyperx_hop_distance(0, 8, cfg) == 3
    assert hyperx_hop_distance(0, 16, cfg) == 3


def test_hyperx_formula_tp_ring_crossing_hbi_domain():
    # TP=16 with hbi_size=8 splits one TP group across two HBI sub-domains,
    # so the TP ring edge between local ranks 7 and 8 crosses that boundary.
    cfg = ParallelismConfig(total_gpus=16, tp=16, dp=1, pp=1, hbi_size=8)

    assert hyperx_hop_distance(7, 8, cfg) == 3
