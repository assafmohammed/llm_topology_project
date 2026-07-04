from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import networkx as nx


GPU_PREFIX = "gpu"
SWITCH_PREFIX = "sw"
HOP_WEIGHT = "hop_weight"


@dataclass(frozen=True)
class ParallelismConfig:
    """
    Parallelism configuration.

    total_gpus = TP * DP * PP
    hbi_size is 8 because each HBI domain/server contains 8 GPUs.
    """

    total_gpus: int
    tp: int
    dp: int
    pp: int
    hbi_size: int = 8

    def validate(self) -> None:
        values = {
            "total_gpus": self.total_gpus,
            "tp": self.tp,
            "dp": self.dp,
            "pp": self.pp,
            "hbi_size": self.hbi_size,
        }

        for name, value in values.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        expected = self.tp * self.dp * self.pp
        if expected != self.total_gpus:
            raise ValueError(
                f"Invalid config: total_gpus must equal TP*DP*PP. "
                f"Got total_gpus={self.total_gpus}, TP*DP*PP={expected}."
            )

    @property
    def stage_size(self) -> int:
        return self.tp * self.dp

    @property
    def hyperx_dims(self) -> tuple[int, int, int]:
        """
        HyperX dimensions:
        S1 = ceil(TP / 8), S2 = DP, S3 = PP.
        """
        self.validate()
        return (max(1, ceil(self.tp / self.hbi_size)), self.dp, self.pp)


def gpu_node(gpu_id: int) -> str:
    return f"{GPU_PREFIX}:{gpu_id}"


def switch_node(coord: tuple[int, int, int]) -> str:
    x, y, z = coord
    return f"{SWITCH_PREFIX}:({x},{y},{z})"


def gpu_ids(total_gpus: int) -> range:
    if total_gpus <= 0:
        raise ValueError("total_gpus must be positive")
    return range(total_gpus)


def add_link(
    graph: nx.Graph,
    u: str,
    v: str,
    *,
    hop_weight: int = 1,
    link_type: str,
) -> None:
    graph.add_edge(u, v, **{HOP_WEIGHT: hop_weight, "link_type": link_type})


def nodes_by_role(graph: nx.Graph, role: str) -> list[str]:
    return [node for node, data in graph.nodes(data=True) if data.get("role") == role]