from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product

import networkx as nx

from .common import (
    ParallelismConfig,
    add_link,
    gpu_ids,
    gpu_node,
    switch_node,
)


def hyperx_switch_coord(gpu_id: int, cfg: ParallelismConfig) -> tuple[int, int, int]:
    """
    Map GPU i to HyperX switch coordinate.

    Formula:
        phi(i) = (
            floor((i mod TP) / 8),
            floor(i / TP) mod DP,
            floor(i / (TP * DP))
        )

    This follows TP fastest, then DP, then PP.
    """
    cfg.validate()

    if gpu_id < 0 or gpu_id >= cfg.total_gpus:
        raise ValueError(f"gpu_id must be in [0, {cfg.total_gpus}), got {gpu_id}")

    s1 = (gpu_id % cfg.tp) // cfg.hbi_size
    s2 = (gpu_id // cfg.tp) % cfg.dp
    s3 = gpu_id // (cfg.tp * cfg.dp)

    dims = cfg.hyperx_dims
    if not (0 <= s1 < dims[0] and 0 <= s2 < dims[1] and 0 <= s3 < dims[2]):
        raise RuntimeError(f"Invalid HyperX coordinate {(s1, s2, s3)} for dims={dims}")

    return (s1, s2, s3)


def _all_switch_coords(dims: tuple[int, int, int]) -> list[tuple[int, int, int]]:
    return list(product(range(dims[0]), range(dims[1]), range(dims[2])))


def build_hyperx(cfg: ParallelismConfig) -> nx.Graph:
    """
    Build an undirected HyperX graph.

    Node types:
    - gpu:<id>
    - sw:(x,y,z)

    Hop model:
    - GPUs inside the same HBI domain have direct 1-hop HBI links.
    - GPU to its switch is 1 hop.
    - Switch to switch along one HyperX dimension is 1 hop.
    - Therefore, GPUs on switches differing in one dimension are 3 hops:
      GPU -> switch -> switch -> GPU.
    """
    cfg.validate()

    graph = nx.Graph()
    dims = cfg.hyperx_dims

    graph.graph["topology"] = "HyperX"
    graph.graph["total_gpus"] = cfg.total_gpus
    graph.graph["tp"] = cfg.tp
    graph.graph["dp"] = cfg.dp
    graph.graph["pp"] = cfg.pp
    graph.graph["hbi_size"] = cfg.hbi_size
    graph.graph["dims"] = dims

    coords = _all_switch_coords(dims)

    for coord in coords:
        graph.add_node(
            switch_node(coord),
            role="switch",
            coord=coord,
        )

    # HyperX switch links:
    # For each dimension, switches with the same other coordinates are fully connected.
    for dim_index, dim_size in enumerate(dims):
        if dim_size <= 1:
            continue

        other_ranges = [range(d) for d in dims]
        fixed_dims = [i for i in range(3) if i != dim_index]

        for fixed_values in product(*(other_ranges[i] for i in fixed_dims)):
            base = [0, 0, 0]
            for idx, value in zip(fixed_dims, fixed_values):
                base[idx] = value

            for a, b in combinations(range(dim_size), 2):
                coord_a = list(base)
                coord_b = list(base)
                coord_a[dim_index] = a
                coord_b[dim_index] = b

                add_link(
                    graph,
                    switch_node(tuple(coord_a)),
                    switch_node(tuple(coord_b)),
                    hop_weight=1,
                    link_type=f"hyperx_dim_{dim_index}",
                )

    gpus_by_switch: dict[tuple[int, int, int], list[str]] = defaultdict(list)

    for gpu_id in gpu_ids(cfg.total_gpus):
        coord = hyperx_switch_coord(gpu_id, cfg)
        gnode = gpu_node(gpu_id)
        snode = switch_node(coord)

        graph.add_node(
            gnode,
            role="gpu",
            gpu_id=gpu_id,
            switch=snode,
            coord=coord,
        )

        add_link(
            graph,
            gnode,
            snode,
            hop_weight=1,
            link_type="gpu_switch",
        )

        gpus_by_switch[coord].append(gnode)

    # HBI links: GPUs attached to the same switch/server are directly connected.
    for coord, gpu_nodes in gpus_by_switch.items():
        for u, v in combinations(gpu_nodes, 2):
            add_link(
                graph,
                u,
                v,
                hop_weight=1,
                link_type="hbi",
            )

    return graph


def hyperx_summary(graph: nx.Graph) -> str:
    gpu_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "gpu")
    switch_count = sum(1 for _, data in graph.nodes(data=True) if data.get("role") == "switch")

    return (
        f"HyperX summary: GPUs={gpu_count}, switches={switch_count}, "
        f"edges={graph.number_of_edges()}, dims={graph.graph.get('dims')}"
    )