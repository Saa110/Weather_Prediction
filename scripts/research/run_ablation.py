#!/usr/bin/env python3
"""
Feature ablation study: for each (station, architecture, target), train with
one feature group dropped and compare MAE to the full-feature baseline.

Output: docs/analysis/ablation_results.csv and a short summary.

Usage:
  python scripts/research/run_ablation.py              # full run
  python scripts/research/run_ablation.py --dry-run   # print count only
  python scripts/research/run_ablation.py --stations KDEN KLAX  # subset
"""

import sys
from pathlib import Path
import argparse
import time
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sqlite3

from src.research.experiment_runner import run_experiment
from src.research.stations import STATIONS_RESEARCH
from src.research.ablation_config import ABLATION_GROUP_IDS, FEATURE_GROUPS
from src.research.database import insert_ablation_results

DB_PATH = Path("data/research/results.db")
OUT_CSV = Path("docs/analysis/ablation_results.csv")
TRAIN_START = "2022-07-01"  # 3yr


def load_baseline_mae() -> pd.DataFrame:
    """Baseline MAE (full features) per station, architecture, target (3yr HF)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT station_id, architecture, target_variable, mae
        FROM experiments
        WHERE data_source = 'historical_forecast'
          AND train_start LIKE '2022-07%'
    """, conn)
    conn.close()
    # One row per (station, arch, target) — keep min MAE if duplicates
    baseline = df.groupby(["station_id", "architecture", "target_variable"])["mae"].min().reset_index()
    baseline = baseline.rename(columns={"mae": "mae_baseline"})
    return baseline


def main():
    parser = argparse.ArgumentParser(description="Run feature ablation study")
    parser.add_argument("--dry-run", action="store_true", help="Print experiment count only")
    parser.add_argument("--stations", nargs="+", default=None, help="Subset of station IDs (default: all 28)")
    parser.add_argument("--groups", nargs="+", default=None, help="Subset of ablation groups (default: all 7). Use e.g. --groups NWP_primary for sanity-only run.")
    args = parser.parse_args()

    stations = args.stations or list(STATIONS_RESEARCH.keys())
    groups = args.groups or ABLATION_GROUP_IDS
    invalid = [g for g in groups if g not in FEATURE_GROUPS]
    if invalid:
        print(f"Unknown ablation group(s): {invalid}. Valid: {list(FEATURE_GROUPS.keys())}")
        return

    baseline_df = load_baseline_mae()
    total = len(stations) * 5 * 2 * len(groups)  # 5 arch, 2 targets
    print(f"Ablation study: {len(stations)} stations × 5 arch × 2 targets × {len(groups)} groups = {total} runs")
    if args.dry_run:
        return

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    done = 0
    failed = 0
    t0 = time.time()

    for si, station_id in enumerate(stations):
        if station_id not in STATIONS_RESEARCH:
            continue
        for target in ["MaxT", "MinT"]:
            for arch in ["catboost", "xgboost", "lightgbm", "linear", "mlp"]:
                bl = baseline_df[
                    (baseline_df["station_id"] == station_id) &
                    (baseline_df["architecture"] == arch) &
                    (baseline_df["target_variable"] == target)
                ]
                mae_baseline = bl["mae_baseline"].iloc[0] if len(bl) else None

                for ablation_group in groups:
                    done += 1
                    try:
                        r = run_experiment(
                            station_id,
                            "historical_forecast",
                            arch,
                            target,
                            TRAIN_START,
                            save_to_db=False,
                            verbose=False,
                            ablation_drop_groups=[ablation_group],
                        )
                        mae = r["metrics"]["mae"]
                        n_feat = r["extra"].get("n_features", 0)
                        delta = (mae - mae_baseline) if mae_baseline is not None else np.nan
                        rows.append({
                            "station_id": station_id,
                            "architecture": arch,
                            "target_variable": target,
                            "ablation_group": ablation_group,
                            "mae": mae,
                            "mae_baseline": mae_baseline,
                            "delta_mae": delta,
                            "n_features": n_feat,
                        })
                    except Exception as e:
                        failed += 1
                        rows.append({
                            "station_id": station_id,
                            "architecture": arch,
                            "target_variable": target,
                            "ablation_group": ablation_group,
                            "mae": np.nan,
                            "mae_baseline": mae_baseline,
                            "delta_mae": np.nan,
                            "n_features": np.nan,
                        })
                        if done <= 3 or failed <= 5:
                            print(f"  FAIL {station_id} {arch} {target} {ablation_group}: {e}")

        elapsed = time.time() - t0
        eta = (elapsed / (si + 1)) * (len(stations) - si - 1) if si < len(stations) - 1 else 0
        pct = done / total * 100
        print(f"  [{si+1}/{len(stations)}] {station_id} | {done}/{total} ({pct:.0f}%) | ETA: {eta/60:.1f} min")

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    # If we ran only a subset of groups, merge with existing results so we don't overwrite
    if args.groups and OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV)
        existing = existing[~existing["ablation_group"].isin(args.groups)]
        df = pd.concat([existing, df], ignore_index=True)
        print(f"  Merged with existing: {len(df)} total rows (replaced groups: {args.groups})")
    df.to_csv(OUT_CSV, index=False)
    # Replace these groups in SQLite so we don't duplicate on re-runs
    conn = sqlite3.connect(DB_PATH)
    placeholders = ",".join("?" * len(groups))
    conn.execute(f"DELETE FROM ablation_runs WHERE ablation_group IN ({placeholders})", groups)
    conn.commit()
    conn.close()
    n_inserted = insert_ablation_results(rows)
    elapsed = time.time() - t0
    print(f"\nDone: {done} runs in {elapsed/60:.1f} min ({failed} failed)")
    print(f"  CSV: {OUT_CSV}")
    print(f"  DB:  {n_inserted} rows inserted in ablation_runs table")

    # Summary: mean delta MAE by (architecture, target, ablation_group)
    summary = df.groupby(["architecture", "target_variable", "ablation_group"])["delta_mae"].agg(["mean", "median", "count"]).reset_index()
    summary = summary.sort_values(["target_variable", "architecture", "mean"], ascending=[True, True, False])
    print("\n--- Mean ΔMAE (positive = worse without that group) ---")
    for target in ["MaxT", "MinT"]:
        print(f"\n  {target}:")
        s = summary[summary["target_variable"] == target]
        pivot = s.pivot(index="ablation_group", columns="architecture", values="mean").round(3)
        print(pivot.to_string())
    return df


if __name__ == "__main__":
    main()
