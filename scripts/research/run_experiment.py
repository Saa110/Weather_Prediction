#!/usr/bin/env python3
"""
CLI entry point for running a single research experiment.

Usage:
    python scripts/research/run_experiment.py \
        --station KDEN \
        --source historical_forecast \
        --arch catboost \
        --target MaxT \
        --train-start 2022-01-01

    python scripts/research/run_experiment.py \
        --station KLAX --source reanalysis --arch xgboost --target MinT \
        --train-start 2020-01-01 --verbose
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from src.research.experiment_runner import run_experiment
from src.research.stations import STATIONS_RESEARCH


def main():
    parser = argparse.ArgumentParser(description="Run a single research experiment")

    parser.add_argument("--station", required=True,
                        choices=sorted(STATIONS_RESEARCH.keys()),
                        help="Station ICAO code")
    parser.add_argument("--source", required=True,
                        choices=["historical_forecast", "reanalysis"],
                        help="Data source")
    parser.add_argument("--arch", required=True,
                        choices=["catboost", "xgboost", "lightgbm", "linear", "mlp"],
                        help="Model architecture")
    parser.add_argument("--target", required=True,
                        choices=["MaxT", "MinT"],
                        help="Target variable")
    parser.add_argument("--train-start", required=True,
                        help="Training period start (YYYY-MM-DD)")
    parser.add_argument("--train-end", default="2025-06-30",
                        help="Training period end (default: 2025-06-30)")
    parser.add_argument("--no-db", action="store_true",
                        help="Don't save results to database")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed progress")

    args = parser.parse_args()

    result = run_experiment(
        station_id=args.station,
        data_source=args.source,
        architecture=args.arch,
        target_variable=args.target,
        train_start=args.train_start,
        train_end=args.train_end,
        save_to_db=not args.no_db,
        verbose=args.verbose,
    )

    # Always print the summary line
    m = result["metrics"]
    ci = result["bootstrap_ci"]
    p = result["params"]
    print(
        f"{p['station_id']} | {p['data_source']:20s} | {p['architecture']:10s} | "
        f"{p['target_variable']:4s} | "
        f"MAE={m['mae']:.2f} [{ci['mae'][0]:.2f},{ci['mae'][1]:.2f}] | "
        f"RMSE={m['rmse']:.2f} | Bias={m['bias']:+.2f} | "
        f"n_train={p['n_train']} n_test={m['n_test']}"
    )


if __name__ == "__main__":
    main()
