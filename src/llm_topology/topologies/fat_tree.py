from __future__ import annotations

from itertools import combinations

import networkx as nx
import numpy as np

from .common import ParallelismConfig, add_link, gpu_ids, gpu_node


def rail_node(local_rank: int) -> str:
    return f"rail:{local_rank}"


def spine_node(index: int) -> str:
    return f"spine:{index}"


def build_fat_tree(cfg: ParallelismConfig, num_spines: int = 2) -> nx.Graph:
    """
    Build a simplified Fat-tree graph.

    Node types:
    - gpu:<id>
    - rail:<local_rank>
    - spine:<index>

    Hop model:
    - GPUs inside the same HBI domain are directly connected (HBI clique), 1 hop.
    - GPU to its rail switch is 1 hop.
    - Rail switch to spine switch is 1 hop.
    - Therefore GPUs with the same local rank in different HBI domains are 2 hops
      (GPU -> rail -> GPU), and GPUs with different local ranks in different HBI
      domains are 4 hops (GPU -> rail -> spine -> rail -> GPU).
    """
    cfg.validate()

    if num_spines <= 0:
        raise ValueError("num_spines must be a positive integer")

    graph = nx.Graph()

    num_domains = cfg.total_gpus // cfg.hbi_size

    graph.graph["topology"] = "FatTree"
    graph.graph["total_gpus"] = cfg.total_gpus
    graph.graph["tp"] = cfg.tp
    graph.graph["dp"] = cfg.dp
    graph.graph["pp"] = cfg.pp
    graph.graph["hbi_size"] = cfg.hbi_size
    graph.graph["num_hbi_domains"] = num_domains
    graph.graph["num_spines"] = num_spines

    for local_rank in range(cfg.hbi_size):
        graph.add_node(rail_node(local_rank), role="rail", rail_id=local_rank)

    for index in range(num_spines):
        graph.add_node(spine_node(index), role="spine", spine_id=index)

    gpus_by_domain: dict[int, list[str]] = {}

    for gpu_id in gpu_ids(cfg.total_gpus):
        hbi_domain = gpu_id // cfg.hbi_size
        local_rank = gpu_id % cfg.hbi_size
        gnode = gpu_node(gpu_id)

        graph.add_node(
            gnode,
            role="gpu",
            gpu_id=gpu_id,
            hbi_domain=hbi_domain,
            local_rank=local_rank,
        )

        add_link(
            graph,
            gnode,
            rail_node(local_rank),
            hop_weight=1,
            link_type="gpu_rail",
        )

        gpus_by_domain.setdefault(hbi_domain, []).append(gnode)

    # HBI links: GPUs inside the same HBI domain are fully connected.
    for gpu_nodes in gpus_by_domain.values():
        for u, v in combinations(gpu_nodes, 2):
            add_link(graph, u, v, hop_weight=1, link_type="hbi")

    # Rail-to-spine links: every rail switch connects to every spine switch.
    for local_rank in range(cfg.hbi_size):
        for index in range(num_spines):
            add_link(
                graph,
                rail_node(local_rank),
                spine_node(index),
                hop_weight=1,
                link_type="rail_spine",
            )

    return graph


def fat_tree_hbi_domain(gpu_id: int, cfg: ParallelismConfig) -> int:
    cfg.validate()
    return gpu_id // cfg.hbi_size


def fat_tree_local_rank(gpu_id: int, cfg: ParallelismConfig) -> int:
    cfg.validate()
    return gpu_id % cfg.hbi_size


def fat_tree_hop_distance(src_gpu: int, dst_gpu: int, cfg: ParallelismConfig) -> int:
    """
    Formula-based Fat-tree hop distance.

    NetworkX shortest paths on the physical graph can "shortcut" through an
    HBI clique edge as a transit hop (e.g. gpu0 -> rail0 -> gpu8 -> gpu9),
    which is not a valid Fat-tree routing path: HBI links only give 1-hop
    reachability between GPUs inside the same HBI domain, they are not used
    to transit traffic destined for another domain. This formula enforces
    the intended rail -> spine -> rail routing instead.
    """
    cfg.validate()

    if src_gpu == dst_gpu:
        return 0

    src_domain = fat_tree_hbi_domain(src_gpu, cfg)
    dst_domain = fat_tree_hbi_domain(dst_gpu, cfg)

    src_rank = fat_tree_local_rank(src_gpu, cfg)
    dst_rank = fat_tree_local_rank(dst_gpu, cfg)

    if src_domain == dst_domain:
        return 1

    if src_rank == dst_rank:
        return 2

    return 4


def fat_tree_hop_matrix(cfg: ParallelismConfig) -> np.ndarray:
    cfg.validate()

    matrix = np.zeros((cfg.total_gpus, cfg.total_gpus), dtype=float)

    for src_gpu in gpu_ids(cfg.total_gpus):
        for dst_gpu in gpu_ids(cfg.total_gpus):
            matrix[src_gpu, dst_gpu] = fat_tree_hop_distance(src_gpu, dst_gpu, cfg)

    return matrix


def fat_tree_summary(graph: nx.Graph) -> str:
    gpu_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "gpu")
    rail_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "rail")
    spine_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "spine")

    return (
        f"FatTree summary: GPUs={gpu_count}, rails={rail_count}, spines={spine_count}, "
        f"edges={graph.number_of_edges()}, "
        f"hbi_domains={graph.graph.get('num_hbi_domains')}"
    )
