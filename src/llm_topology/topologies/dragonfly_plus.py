from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

from .common import ParallelismConfig, add_link, gpu_ids, gpu_node


def leaf_node(group_id: int, leaf_id: int) -> str:
    return f"df:{group_id}:leaf:{leaf_id}"


def spine_node(group_id: int, spine_id: int) -> str:
    return f"df:{group_id}:spine:{spine_id}"


def dragonfly_hbi_domain(gpu_id: int, cfg: ParallelismConfig) -> int:
    cfg.validate()
    return gpu_id // cfg.hbi_size


def dragonfly_group_id(gpu_id: int, cfg: ParallelismConfig) -> int:
    """
    Dragonfly+ group corresponds to a pipeline stage.
    """
    cfg.validate()
    return gpu_id // (cfg.tp * cfg.dp)


def dragonfly_leaf_id(gpu_id: int, cfg: ParallelismConfig) -> int:
    """
    Within a pipeline stage/group, leaf corresponds to the DP index.
    """
    cfg.validate()
    return (gpu_id // cfg.tp) % cfg.dp


def dragonfly_local_rank(gpu_id: int, cfg: ParallelismConfig) -> int:
    cfg.validate()
    return gpu_id % cfg.hbi_size


def dragonfly_hop_distance(src_gpu: int, dst_gpu: int, cfg: ParallelismConfig) -> int:
    """
    Formula-based Dragonfly+ hop distance.

    Like Fat-tree, this avoids using NetworkX shortest paths on the physical
    graph, since HBI clique edges could otherwise be used as transit
    shortcuts between leaves/groups instead of only for intra-domain hops.
    """
    cfg.validate()

    if not (0 <= src_gpu < cfg.total_gpus):
        raise ValueError(f"src_gpu must be in [0, {cfg.total_gpus}), got {src_gpu}")
    if not (0 <= dst_gpu < cfg.total_gpus):
        raise ValueError(f"dst_gpu must be in [0, {cfg.total_gpus}), got {dst_gpu}")

    if src_gpu == dst_gpu:
        return 0

    src_domain = dragonfly_hbi_domain(src_gpu, cfg)
    dst_domain = dragonfly_hbi_domain(dst_gpu, cfg)

    if src_domain == dst_domain:
        return 1

    src_group = dragonfly_group_id(src_gpu, cfg)
    dst_group = dragonfly_group_id(dst_gpu, cfg)

    if src_group == dst_group:
        return 4

    return 5


def dragonfly_hop_matrix(cfg: ParallelismConfig) -> np.ndarray:
    cfg.validate()

    matrix = np.zeros((cfg.total_gpus, cfg.total_gpus), dtype=float)

    for src_gpu in gpu_ids(cfg.total_gpus):
        for dst_gpu in gpu_ids(cfg.total_gpus):
            matrix[src_gpu, dst_gpu] = dragonfly_hop_distance(src_gpu, dst_gpu, cfg)

    return matrix


def build_dragonfly_plus(cfg: ParallelismConfig, spines_per_group: int = 2) -> nx.Graph:
    """
    Build a simplified Dragonfly+ graph.

    Node types:
    - gpu:<id>
    - df:<group>:leaf:<leaf_id>
    - df:<group>:spine:<spine_id>

    Hop model:
    - GPUs inside the same HBI domain are directly connected (HBI clique), 1 hop.
    - GPU to its leaf switch is 1 hop.
    - Leaf to spine inside the same group is 1 hop (complete bipartite).
    - Spine to spine across groups is 1 hop.
    - Therefore GPUs in different HBI domains within the same group are 4 hops
      (GPU -> leaf -> spine -> leaf -> GPU), and GPUs in different groups are
      5 hops (GPU -> leaf -> spine -> spine -> leaf -> GPU).
    """
    cfg.validate()

    if spines_per_group <= 0:
        raise ValueError("spines_per_group must be a positive integer")

    graph = nx.Graph()

    num_domains = cfg.total_gpus // cfg.hbi_size
    num_groups = cfg.pp

    graph.graph["topology"] = "Dragonfly+"
    graph.graph["total_gpus"] = cfg.total_gpus
    graph.graph["tp"] = cfg.tp
    graph.graph["dp"] = cfg.dp
    graph.graph["pp"] = cfg.pp
    graph.graph["hbi_size"] = cfg.hbi_size
    graph.graph["num_hbi_domains"] = num_domains
    graph.graph["num_groups"] = num_groups
    graph.graph["spines_per_group"] = spines_per_group

    gpus_by_domain: dict[int, list[str]] = {}
    domain_leaf: dict[int, tuple[int, int]] = {}

    for gpu_id in gpu_ids(cfg.total_gpus):
        hbi_domain = dragonfly_hbi_domain(gpu_id, cfg)
        group_id = dragonfly_group_id(gpu_id, cfg)
        leaf_id = dragonfly_leaf_id(gpu_id, cfg)
        local_rank = dragonfly_local_rank(gpu_id, cfg)
        gnode = gpu_node(gpu_id)

        graph.add_node(
            gnode,
            role="gpu",
            gpu_id=gpu_id,
            hbi_domain=hbi_domain,
            dragonfly_group=group_id,
            leaf_id=leaf_id,
            local_rank=local_rank,
        )

        gpus_by_domain.setdefault(hbi_domain, []).append(gnode)
        domain_leaf[hbi_domain] = (group_id, leaf_id)

    for group_id in range(num_groups):
        for spine_id in range(spines_per_group):
            graph.add_node(
                spine_node(group_id, spine_id),
                role="spine",
                dragonfly_group=group_id,
                spine_id=spine_id,
            )

    for hbi_domain, (group_id, leaf_id) in domain_leaf.items():
        graph.add_node(
            leaf_node(group_id, leaf_id),
            role="leaf",
            dragonfly_group=group_id,
            leaf_id=leaf_id,
            hbi_domain=hbi_domain,
        )

    # HBI links: GPUs inside the same HBI domain are fully connected.
    for gpu_nodes in gpus_by_domain.values():
        for u, v in combinations(gpu_nodes, 2):
            add_link(graph, u, v, hop_weight=1, link_type="hbi")

    # GPU-to-leaf links.
    for hbi_domain, gpu_nodes in gpus_by_domain.items():
        group_id, leaf_id = domain_leaf[hbi_domain]
        lnode = leaf_node(group_id, leaf_id)
        for gnode in gpu_nodes:
            add_link(graph, gnode, lnode, hop_weight=1, link_type="gpu_leaf")

    # Leaf-spine links: complete bipartite inside each group.
    leaves_by_group: dict[int, set[int]] = {}
    for group_id, leaf_id in domain_leaf.values():
        leaves_by_group.setdefault(group_id, set()).add(leaf_id)

    for group_id, leaf_ids in leaves_by_group.items():
        for leaf_id in leaf_ids:
            for spine_id in range(spines_per_group):
                add_link(
                    graph,
                    leaf_node(group_id, leaf_id),
                    spine_node(group_id, spine_id),
                    hop_weight=1,
                    link_type="leaf_spine",
                )

    # Inter-group spine-spine links: spine 0 of each group connects to
    # spine 0 of the next group.
    for group_id in range(num_groups - 1):
        add_link(
            graph,
            spine_node(group_id, 0),
            spine_node(group_id + 1, 0),
            hop_weight=1,
            link_type="inter_group_spine",
        )

    return graph


def dragonfly_plus_summary(graph: nx.Graph) -> str:
    gpu_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "gpu")
    leaf_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "leaf")
    spine_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "spine")

    return (
        f"Dragonfly+ summary: GPUs={gpu_count}, leaves={leaf_count}, spines={spine_count}, "
        f"edges={graph.number_of_edges()}, "
        f"groups={graph.graph.get('num_groups')}"
    )
