from __future__ import annotations

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison
from llm_topology.metrics.path_diversity import compute_path_diversity_pairs, summarize_path_diversity
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


PROVIDERS = {
    "HyperX": lambda i, j, cfg: hyperx_paths(i, j, cfg),
    "Fat-tree": lambda i, j, cfg: fat_tree_paths(i, j, cfg, num_spines=2),
    "Dragonfly+": lambda i, j, cfg: dragonfly_paths(i, j, cfg, spines_per_group=2),
}


def _pair_df(name: str, cfg: ParallelismConfig):
    factory = PROVIDERS[name]
    provider = lambda i, j, cfg=cfg, factory=factory: factory(i, j, cfg)
    traffic = make_traffic(cfg)
    return compute_path_diversity_pairs(traffic, provider)


def test_path_diversity_pairs_non_empty_32():
    cfg = make_cfg()

    for name in PROVIDERS:
        pair_df = _pair_df(name, cfg)

        assert not pair_df.empty
        for column in ["src_gpu", "dst_gpu", "traffic", "path_count", "is_single_path"]:
            assert column in pair_df.columns

        assert (pair_df["path_count"] >= 1).all()
        assert (pair_df["traffic"] > 0).all()


def test_path_diversity_summary_fields():
    cfg = make_cfg()

    for name in PROVIDERS:
        pair_df = _pair_df(name, cfg)
        summary = summarize_path_diversity(name, pair_df)

        for key in [
            "traffic_weighted_path_diversity",
            "single_path_traffic_exposure",
            "single_path_pair_fraction",
            "max_path_count",
            "mean_path_count",
        ]:
            assert key in summary


def test_single_path_exposure_range():
    cfg = make_cfg()

    for name in PROVIDERS:
        pair_df = _pair_df(name, cfg)
        summary = summarize_path_diversity(name, pair_df)

        assert 0 <= summary["single_path_traffic_exposure"] <= 1
        assert 0 <= summary["single_path_pair_fraction"] <= 1


def test_weighted_path_diversity_reasonable():
    cfg = make_cfg()

    for name in PROVIDERS:
        pair_df = _pair_df(name, cfg)
        summary = summarize_path_diversity(name, pair_df)

        assert summary["traffic_weighted_path_diversity"] >= 1
        assert summary["max_path_count"] >= summary["traffic_weighted_path_diversity"]


def test_pipeline_outputs_routing_flexibility(tmp_path):
    spec = ExperimentSpec(
        name="test_routing_flexibility",
        total_gpus=32,
        tp=8,
        dp=2,
        pp=2,
        hbi_size=8,
        traffic_mode="synthetic",
        output_dir=tmp_path / "test_routing_flexibility",
    )

    run_comparison(spec)
    output_dir = spec.output_dir

    assert (output_dir / "routing_flexibility_summary.csv").exists()
    assert (output_dir / "routing_flexibility_path_diversity.png").exists()
    assert (output_dir / "routing_flexibility_single_path_exposure.png").exists()
    assert (output_dir / "routing_flexibility_path_count_cdf.png").exists()
