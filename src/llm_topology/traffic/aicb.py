from __future__ import annotations

from pathlib import Path

import numpy as np

RECOGNIZED_ROW_FIELDS = 3  # src_rank, dst_rank, message_bytes


def aicb_to_matrix(path: str | Path, total_gpus: int) -> np.ndarray:
    """
    Parse an AICB/SimAI workload trace into a GPU x GPU traffic matrix.

    AICB/SimAI workload files vary by version and collective schedule, and
    we do not have a fixed schema to parse against here. Rather than guess
    and risk silently fabricating a wrong traffic matrix, this only accepts
    the one unambiguous line shape it can parse safely: plain rows of
    "src_rank dst_rank message_bytes" (comma or whitespace separated). Any
    other line is skipped, and if the file contains no recognizable rows at
    all, parsing fails loudly instead of returning fake data.

    Use traffic_mode="npy" or "csv" with matrix_loader.load_traffic_matrix
    for a real workload matrix in the meantime.
    """
    path = Path(path)
    matrix = np.zeros((total_gpus, total_gpus), dtype=float)
    recognized_rows = 0

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        fields = line.replace(",", " ").split()
        if len(fields) != RECOGNIZED_ROW_FIELDS:
            continue

        try:
            src_rank = int(fields[0])
            dst_rank = int(fields[1])
            message_bytes = float(fields[2])
        except ValueError:
            continue

        if not (0 <= src_rank < total_gpus and 0 <= dst_rank < total_gpus):
            continue

        matrix[src_rank, dst_rank] += message_bytes
        recognized_rows += 1

    if recognized_rows == 0:
        raise NotImplementedError(
            "AICB parser needs the exact workload format. Use .npy or .csv "
            "traffic matrix loader for now."
        )

    np.fill_diagonal(matrix, 0.0)
    return matrix
