from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import run_compare32
import run_compare128
import run_compare1024
import run_paper_sweep

from llm_topology.experiments.pipeline import ExperimentSpec, run_comparison

DEFAULT_1024_CONFIGS = [
    {"name": "compare1024_tp8_dp16_pp8", "tp": 8, "dp": 16, "pp": 8},
    {"name": "compare1024_tp16_dp8_pp8", "tp": 16, "dp": 8, "pp": 8},
    {"name": "compare1024_tp32_dp4_pp8", "tp": 32, "dp": 4, "pp": 8},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 32/128/1024 GPU comparisons (and optionally the paper sweep) in one command."
    )
    parser.add_argument("--skip-32", action="store_true")
    parser.add_argument("--skip-128", action="store_true")
    parser.add_argument("--skip-1024", action="store_true")
    parser.add_argument("--include-sweep", action="store_true")
    parser.add_argument("--traffic-mode", type=str, default="synthetic")
    parser.add_argument("--output-root", type=str, default="results")
    return parser.parse_args()


def _run_custom_spec(
    name: str,
    total_gpus: int,
    tp: int,
    dp: int,
    pp: int,
    output_root: Path,
    traffic_mode: str,
) -> None:
    spec = ExperimentSpec(
        name=name,
        total_gpus=total_gpus,
        tp=tp,
        dp=dp,
        pp=pp,
        hbi_size=8,
        traffic_mode=traffic_mode,
        output_dir=output_root / name,
    )
    print(f"Running {spec.name}: TP={spec.tp}, DP={spec.dp}, PP={spec.pp} "
          f"(traffic_mode={traffic_mode}) ...")
    run_comparison(spec)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)

    # When the caller sticks to the defaults, delegate straight to each
    # script's own main() so `run_all.py` reproduces exactly what running
    # each script individually would produce. A custom --traffic-mode or
    # --output-root instead builds fresh specs here, since the individual
    # scripts hardcode synthetic traffic under results/.
    use_defaults = args.traffic_mode == "synthetic" and args.output_root == "results"

    if not args.skip_32:
        if use_defaults:
            run_compare32.main()
        else:
            _run_custom_spec("compare32", 32, 8, 2, 2, output_root, args.traffic_mode)

    if not args.skip_128:
        if use_defaults:
            run_compare128.main()
        else:
            _run_custom_spec(
                "compare128_tp8_dp4_pp4", 128, 8, 4, 4, output_root, args.traffic_mode
            )

    if not args.skip_1024:
        if use_defaults:
            run_compare1024.main()
        else:
            for config in DEFAULT_1024_CONFIGS:
                _run_custom_spec(
                    config["name"],
                    1024,
                    config["tp"],
                    config["dp"],
                    config["pp"],
                    output_root,
                    args.traffic_mode,
                )

    if args.include_sweep:
        if use_defaults:
            run_paper_sweep.run_sweep(1024, [8, 16, 32], [1, 2, 4, 8, 16])
        else:
            specs = run_paper_sweep.generate_valid_specs(1024, [8, 16, 32], [1, 2, 4, 8, 16])
            for spec in specs:
                _run_custom_spec(
                    spec.name, spec.total_gpus, spec.tp, spec.dp, spec.pp, output_root, args.traffic_mode
                )


if __name__ == "__main__":
    main()
