from __future__ import annotations

from llm_topology.metrics.link_load import compute_ecmp_link_loads
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


def test_ecmp_link_loads_nonempty_and_positive_for_all_topologies():
    cfg = make_cfg()
    traffic = make_traffic(cfg)

    providers = {
        "HyperX": lambda i, j: hyperx_paths(i, j, cfg),
        "Fat-tree": lambda i, j: fat_tree_paths(i, j, cfg, num_spines=2),
        "Dragonfly+": lambda i, j: dragonfly_paths(i, j, cfg, spines_per_group=2),
    }

    for provider in providers.values():
        loads = compute_ecmp_link_loads(traffic, provider)

        assert len(loads) > 0
        assert all(load > 0 for load in loads.values())
        assert max(loads.values()) > 0


def test_path_providers_return_paths_for_active_traffic():
    cfg = make_cfg()
    traffic = make_traffic(cfg)

    providers = [
        lambda i, j: hyperx_paths(i, j, cfg),
        lambda i, j: fat_tree_paths(i, j, cfg, num_spines=2),
        lambda i, j: dragonfly_paths(i, j, cfg, spines_per_group=2),
    ]

    active_pairs = [
        (i, j)
        for i in range(cfg.total_gpus)
        for j in range(cfg.total_gpus)
        if traffic[i, j] > 0
    ]

    for provider in providers:
        for src, dst in active_pairs:
            paths = provider(src, dst)
            assert len(paths) >= 1


def test_fat_tree_path_avoids_hbi_transit_shortcut():
    cfg = make_cfg()

    paths = fat_tree_paths(0, 9, cfg, num_spines=2)

    assert len(paths) == 2
    for path in paths:
        assert path[0] == "gpu:0"
        assert path[1] == "rail:0"
        assert path[2].startswith("spine:")
        assert path[3] == "rail:1"
        assert path[4] == "gpu:9"
        assert len(path) == 5


def test_dragonfly_path_cross_group():
    cfg = make_cfg()

    paths = dragonfly_paths(0, 16, cfg, spines_per_group=2)

    assert paths == [
        ["gpu:0", "df:0:leaf:0", "df:0:spine:0", "df:1:spine:0", "df:1:leaf:0", "gpu:16"]
    ]
