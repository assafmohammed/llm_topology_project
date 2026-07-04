from __future__ import annotations

from itertools import permutations

from ..topologies.common import ParallelismConfig, gpu_node, switch_node
from ..topologies.dragonfly_plus import (
    dragonfly_group_id,
    dragonfly_hbi_domain,
    dragonfly_leaf_id,
    leaf_node,
)
from ..topologies.dragonfly_plus import spine_node as dragonfly_spine_node
from ..topologies.fat_tree import fat_tree_hbi_domain, fat_tree_local_rank, rail_node
from ..topologies.fat_tree import spine_node as fat_tree_spine_node
from ..topologies.hyperx import hyperx_switch_coord


def hyperx_paths(src_gpu: int, dst_gpu: int, cfg: ParallelismConfig) -> list[list[str]]:
    """
    Topology-specific routing for HyperX, used instead of generic NetworkX
    shortest paths so link-load/latency analysis matches the real HyperX
    routing model (GPU -> switch -> ... -> switch -> GPU).
    """
    cfg.validate()

    if src_gpu == dst_gpu:
        return [[gpu_node(src_gpu)]]

    src_coord = hyperx_switch_coord(src_gpu, cfg)
    dst_coord = hyperx_switch_coord(dst_gpu, cfg)

    if src_coord == dst_coord:
        return [[gpu_node(src_gpu), gpu_node(dst_gpu)]]

    changed_dims = [i for i in range(3) if src_coord[i] != dst_coord[i]]

    paths: list[list[str]] = []
    for order in permutations(changed_dims):
        coord = list(src_coord)
        switches = [switch_node(tuple(coord))]
        for dim in order:
            coord[dim] = dst_coord[dim]
            switches.append(switch_node(tuple(coord)))
        paths.append([gpu_node(src_gpu), *switches, gpu_node(dst_gpu)])

    return paths


def fat_tree_paths(
    src_gpu: int,
    dst_gpu: int,
    cfg: ParallelismConfig,
    num_spines: int = 2,
) -> list[list[str]]:
    """
    Topology-specific routing for Fat-tree, mirroring fat_tree_hop_distance.

    HBI edges are never used as a transit hop into another HBI domain.
    """
    cfg.validate()

    if src_gpu == dst_gpu:
        return [[gpu_node(src_gpu)]]

    src_domain = fat_tree_hbi_domain(src_gpu, cfg)
    dst_domain = fat_tree_hbi_domain(dst_gpu, cfg)

    if src_domain == dst_domain:
        return [[gpu_node(src_gpu), gpu_node(dst_gpu)]]

    src_rank = fat_tree_local_rank(src_gpu, cfg)
    dst_rank = fat_tree_local_rank(dst_gpu, cfg)

    if src_rank == dst_rank:
        return [[gpu_node(src_gpu), rail_node(src_rank), gpu_node(dst_gpu)]]

    return [
        [
            gpu_node(src_gpu),
            rail_node(src_rank),
            fat_tree_spine_node(spine_id),
            rail_node(dst_rank),
            gpu_node(dst_gpu),
        ]
        for spine_id in range(num_spines)
    ]


def dragonfly_paths(
    src_gpu: int,
    dst_gpu: int,
    cfg: ParallelismConfig,
    spines_per_group: int = 2,
) -> list[list[str]]:
    """
    Topology-specific routing for Dragonfly+, mirroring dragonfly_hop_distance.

    HBI edges are never used as a transit hop into another HBI domain.
    """
    cfg.validate()

    if src_gpu == dst_gpu:
        return [[gpu_node(src_gpu)]]

    src_domain = dragonfly_hbi_domain(src_gpu, cfg)
    dst_domain = dragonfly_hbi_domain(dst_gpu, cfg)

    if src_domain == dst_domain:
        return [[gpu_node(src_gpu), gpu_node(dst_gpu)]]

    src_group = dragonfly_group_id(src_gpu, cfg)
    dst_group = dragonfly_group_id(dst_gpu, cfg)

    src_leaf = leaf_node(src_group, dragonfly_leaf_id(src_gpu, cfg))
    dst_leaf = leaf_node(dst_group, dragonfly_leaf_id(dst_gpu, cfg))

    if src_group == dst_group:
        return [
            [
                gpu_node(src_gpu),
                src_leaf,
                dragonfly_spine_node(src_group, spine_id),
                dst_leaf,
                gpu_node(dst_gpu),
            ]
            for spine_id in range(spines_per_group)
        ]

    return [
        [
            gpu_node(src_gpu),
            src_leaf,
            dragonfly_spine_node(src_group, 0),
            dragonfly_spine_node(dst_group, 0),
            dst_leaf,
            gpu_node(dst_gpu),
        ]
    ]
