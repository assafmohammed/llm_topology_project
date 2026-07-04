from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import run_compare32  # noqa: E402
from llm_topology.topologies.common import ParallelismConfig  # noqa: E402
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic  # noqa: E402


def make_cfg() -> ParallelismConfig:
    return ParallelismConfig(total_gpus=32, tp=8, dp=2, pp=2, hbi_size=8)


def test_run_compare32_imports():
    assert hasattr(run_compare32, "compare_topologies")
    assert hasattr(run_compare32, "main")


def test_compare_topologies_returns_three_results():
    cfg = make_cfg()
    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )

    results = run_compare32.compare_topologies(cfg, traffic)

    assert set(results.keys()) == {"HyperX", "Fat-tree", "Dragonfly+"}


def test_weighted_average_hops_ordering():
    cfg = make_cfg()
    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )

    results = run_compare32.compare_topologies(cfg, traffic)

    fat_tree_avg = results["Fat-tree"]["weighted_avg"]
    hyperx_avg = results["HyperX"]["weighted_avg"]
    dragonfly_avg = results["Dragonfly+"]["weighted_avg"]

    assert fat_tree_avg < hyperx_avg < dragonfly_avg


def test_active_hop_distributions():
    cfg = make_cfg()
    traffic = generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )

    results = run_compare32.compare_topologies(cfg, traffic)

    assert results["HyperX"]["distribution"] == {1: 0.5, 3: 0.5}
    assert results["Fat-tree"]["distribution"] == {1: 0.5, 2: 0.5}
    assert results["Dragonfly+"]["distribution"] == {1: 0.5, 4: 0.25, 5: 0.25}
