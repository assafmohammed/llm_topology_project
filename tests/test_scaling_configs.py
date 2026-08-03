from __future__ import annotations

import pytest

from llm_topology.topologies.common import ParallelismConfig

SCALING_CONFIGS = [
    (128, 8, 4, 4),
    (1024, 8, 16, 8),
    (1024, 16, 8, 8),
    (1024, 32, 4, 8),
]


@pytest.mark.parametrize("total_gpus,tp,dp,pp", SCALING_CONFIGS)
def test_scaling_config_is_consistent(total_gpus, tp, dp, pp):
    assert total_gpus == tp * dp * pp

    cfg = ParallelismConfig(total_gpus=total_gpus, tp=tp, dp=dp, pp=pp, hbi_size=8)
    cfg.validate()

    assert cfg.total_gpus == total_gpus
