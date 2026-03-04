#!/usr/bin/env python3
"""
Batch runner for the full research experiment matrix.

Reads the experiment manifest and runs all experiments sequentially,
storing results in the SQLite database.  Tracks progress and can
resume from where it left off (skips experiments already in the DB).

Usage:
    python scripts/research/run_all_experiments.py --phase 2
    python scripts/research/run_all_experiments.py --phase 3 --verbose
    python scripts/research/run_all_experiments.py --phase all
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
import argparse
import traceback
from src.research.experiment_runner import run_experiment
from src.research.database import query_results, count_experiments


def load_manifest(phase: str) -> list:
    """Load experiment list for a given phase from manifest."""
    manifest_path = Path("configs/experiment_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    if phase == "all":
        experiments = []
        for p in sorted(manifest.keys()):
            experiments.extend(manifest[p])
        return experiments
    elif phase in manifest:
        return manifest[phase]
    else:
        raise ValueError(f"Unknown phase: {phase}. Available: {list(manifest.keys())}")


def experiment_exists(exp: dict) -> bool:
    """Check if this exact experiment has already been run."""
    df = query_results(
        station_id=exp["station"],
        data_source=exp["source"],
        architecture=exp["arch"],
        target_variable=exp["target"],
    )
    if df.empty:
        return False
    # Check training window matches
    matches = df[
        (df["train_start"] <= exp["train_start"]) &
        (df["train_end"] >= exp.get("train_end", "2025-06-30"))
    ]
    return len(matches) > 0


def main():
    parser = argparse.ArgumentParser(description="Run all experiments for a phase")
    parser.add_argument("--phase", required=True,
                        help="Phase to run (2, 3, 4, 5, or 'all')")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print experiments without running")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Skip experiments already in the database (default: True)")
    args = parser.parse_args()

    experiments = load_manifest(args.phase)
    total = len(experiments)
    print(f"Phase {args.phase}: {total} experiments to run")
    print(f"Database has {count_experiments()} existing results\n")

    skipped = 0
    completed = 0
    failed = 0

    for i, exp in enumerate(experiments, 1):
        label = (
            f"[{i}/{total}] {exp['station']} | {exp['source']} | "
            f"{exp['arch']} | {exp['target']}"
        )

        if args.skip_existing and experiment_exists(exp):
            skipped += 1
            if args.verbose:
                print(f"  SKIP {label} (already in DB)")
            continue

        if args.dry_run:
            print(f"  WOULD RUN {label}")
            continue

        try:
            result = run_experiment(
                station_id=exp["station"],
                data_source=exp["source"],
                architecture=exp["arch"],
                target_variable=exp["target"],
                train_start=exp["train_start"],
                train_end=exp.get("train_end", "2025-06-30"),
                save_to_db=True,
                verbose=args.verbose,
            )
            mae = result["metrics"]["mae"]
            print(f"  OK   {label} | MAE={mae:.2f}")
            completed += 1

        except Exception as e:
            print(f"  FAIL {label} | {e}")
            if args.verbose:
                traceback.print_exc()
            failed += 1

    print(f"\nDone: {completed} completed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
