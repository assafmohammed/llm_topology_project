from __future__ import annotations

from llm_topology.metrics.traffic_analysis import analyze_traffic_matrix, traffic_analysis_dataframe
from llm_topology.topologies.common import ParallelismConfig
from llm_topology.traffic.synthetic import TrafficWeights, generate_llm_like_traffic


def make_cfg() -> ParallelismConfig:
    return ParallelismConfig(total_gpus=32, tp=8, dp=2, pp=2, hbi_size=8)


def make_traffic(cfg: ParallelismConfig):
    return generate_llm_like_traffic(
        cfg,
        TrafficWeights(tp_bytes=100.0, dp_bytes=5.0, pp_bytes=1.0),
    )


def test_analyze_traffic_matrix_basic_fields():
    cfg = make_cfg()
    traffic = make_traffic(cfg)

    analysis = analyze_traffic_matrix(traffic, cfg)

    assert analysis["active_pairs"] > 0
    assert analysis["total_traffic_volume"] > 0
    assert 0 < analysis["traffic_density"] <= 1


def test_tp_dp_pp_fractions_sum_to_one_when_no_other_traffic():
    # The synthetic generator only ever produces TP/DP/PP traffic, so
    # other_traffic_volume should be zero and the three fractions should
    # sum to (approximately) 1.
    cfg = make_cfg()
    traffic = make_traffic(cfg)

    analysis = analyze_traffic_matrix(traffic, cfg)

    assert analysis["other_traffic_volume"] == 0
    total_fraction = (
        analysis["tp_traffic_fraction"]
        + analysis["dp_traffic_fraction"]
        + analysis["pp_traffic_fraction"]
    )
    assert abs(total_fraction - 1.0) < 1e-9


def test_traffic_analysis_dataframe_has_experiment_column():
    cfg = make_cfg()
    traffic = make_traffic(cfg)
    analysis = analyze_traffic_matrix(traffic, cfg)

    df = traffic_analysis_dataframe("test32", analysis)

    assert len(df) == 1
    assert df.iloc[0]["experiment"] == "test32"
    assert df.iloc[0]["active_pairs"] == analysis["active_pairs"]
