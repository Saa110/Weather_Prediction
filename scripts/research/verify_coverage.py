#!/usr/bin/env python3
"""
Verify experiment coverage in the database.

Prints a gap report: which experiments exist, which are missing,
grouped by phase (data source, architecture, training window).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import sqlite3
import pandas as pd

DB_PATH = Path("data/research/results.db")
ABLATION_CSV = Path("docs/analysis/ablation_results.csv")

ALL_STATIONS = [
    "KORD", "KMDW", "KMSP", "KDTW", "KDEN", "KOKC",
    "KNYC", "KLGA", "KBOS", "KPHL", "KDCA",
    "KATL", "KCLT", "KBNA", "KJAX", "KTPA", "KMIA",
    "KHOU", "KMSY", "KDFW", "KDAL", "KAUS", "KSAT",
    "KLAX", "KSFO", "KSEA",
    "KPHX", "KLAS",
]

ARCHITECTURES = ["catboost", "xgboost", "lightgbm", "linear", "mlp"]
TARGETS = ["MaxT", "MinT"]
SOURCES = ["historical_forecast", "reanalysis"]
WINDOWS = {
    "1yr": "2024-07",
    "2yr": "2023-07",
    "3yr": "2022-07",
    "4yr": "2021-07",
    "5yr": "2020-07",
    "8yr": "2018-01",
}


def load_experiments():
    if not DB_PATH.exists():
        print(f"ERROR: {DB_PATH} not found")
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM experiments", conn)
    except Exception as e:
        print(f"ERROR reading experiments table: {e}")
        df = pd.DataFrame()
    conn.close()
    return df


def load_ablation():
    if not DB_PATH.exists():
        return pd.DataFrame()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT * FROM ablation_runs", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if df.empty and ABLATION_CSV.exists():
        df = pd.read_csv(ABLATION_CSV)
    return df


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    df = load_experiments()
    abl = load_ablation()

    # --- Overview ---
    section("DATABASE OVERVIEW")
    print(f"  experiments table:   {len(df)} rows")
    print(f"  ablation_runs:       {len(abl)} rows")
    if len(df) > 0:
        print(f"  Unique stations:     {df['station_id'].nunique()} / 28")
        print(f"  Unique architectures:{df['architecture'].nunique()} / 5")
        print(f"  Unique targets:      {df['target_variable'].nunique()}")
        print(f"  Unique data_sources: {df['data_source'].nunique()}")
        print(f"  Date range:          {df['run_timestamp'].min()[:10]} to {df['run_timestamp'].max()[:10]}")

    if len(df) == 0:
        print("\n  *** No experiments in DB. All phases need to run. ***")
        print_phase_summary_no_data()
        return

    # --- Phase 2: Data Source Comparison ---
    section("PHASE 2 — Data Source Comparison (RQ1+RQ2)")
    print("  Need: 28 stations × 2 sources × 2 targets × 5 architectures = 560 runs")
    for target in TARGETS:
        for source in SOURCES:
            for arch in ARCHITECTURES:
                sub = df[(df["target_variable"] == target) & (df["data_source"] == source) &
                         (df["architecture"] == arch) & (df["train_start"].str.contains("2022"))]
                stations_done = set(sub["station_id"].unique())
                missing = [s for s in ALL_STATIONS if s not in stations_done]
                status = "COMPLETE" if len(missing) == 0 else f"MISSING {len(missing)}"
                print(f"  {target:5s} + {source:20s} + {arch:10s}: {len(stations_done):2d}/28 [{status}]")
                if missing and len(missing) <= 5:
                    print(f"    Missing: {', '.join(missing)}")

    # --- Phase 3: Architecture Comparison ---
    section("PHASE 3 — Architecture Comparison (RQ3)")
    print("  Need: 28 stations × 5 arch × 2 targets × HF (3yr) = 280 runs")
    p3 = df[(df["data_source"] == "historical_forecast") & (df["train_start"].str.contains("2022"))]
    for target in TARGETS:
        print(f"\n  {target}:")
        for arch in ARCHITECTURES:
            sub = p3[(p3["target_variable"] == target) & (p3["architecture"] == arch)]
            stations_done = set(sub["station_id"].unique())
            missing = [s for s in ALL_STATIONS if s not in stations_done]
            status = "COMPLETE" if len(missing) == 0 else f"MISSING {len(missing)}"
            print(f"    {arch:10s}: {len(stations_done):2d}/28 [{status}]")

    # --- Phase 4: Training Window Sensitivity ---
    section("PHASE 4 — Training Window Sensitivity (RQ4)")
    print("  Need: 28 stations × 6 windows × CatBoost × MaxT = 168 runs")
    print("  (Optionally: all 5 arch × 2 targets = 1,680 for full Tables 4-5)")
    p4 = df[(df["data_source"] == "historical_forecast")]
    for label, prefix in WINDOWS.items():
        sub = p4[(p4["architecture"] == "catboost") & (p4["target_variable"] == "MaxT") & (p4["train_start"].str.startswith(prefix))]
        stations_done = set(sub["station_id"].unique())
        missing = [s for s in ALL_STATIONS if s not in stations_done]
        status = "COMPLETE" if len(missing) == 0 else f"MISSING {len(missing)}"
        print(f"  {label} (start {prefix}): {len(stations_done):2d}/28 [{status}]")

    # Extended window check: all arch × all targets
    print("\n  Extended (all 5 arch × 2 targets):")
    for label, prefix in WINDOWS.items():
        sub = p4[p4["train_start"].str.startswith(prefix)]
        combos = set(zip(sub["station_id"], sub["architecture"], sub["target_variable"]))
        total_expected = 28 * 5 * 2
        print(f"  {label}: {len(combos):4d}/{total_expected} combos")

    # --- Baseline MAE (raw NWP) ---
    section("BASELINE (Raw NWP MAE)")
    has_nwp = df[df["raw_nwp_mae"].notna()]
    print(f"  Experiments with raw_nwp_mae: {len(has_nwp)}/{len(df)}")

    # --- Bootstrap CIs ---
    section("BOOTSTRAP CONFIDENCE INTERVALS")
    has_ci = df[df["bootstrap_ci"].notna() & (df["bootstrap_ci"] != "")]
    print(f"  Experiments with bootstrap_ci: {len(has_ci)}/{len(df)}")

    # --- Ablation ---
    section("ABLATION STUDY")
    if len(abl) > 0:
        print(f"  Total ablation runs: {len(abl)}")
        print(f"  Unique stations:     {abl['station_id'].nunique()}")
        print(f"  Groups tested:       {sorted(abl['ablation_group'].unique())}")
        for target in TARGETS:
            sub = abl[abl["target_variable"] == target]
            print(f"  {target}: {len(sub)} runs")
    else:
        print("  No ablation data found in DB; check CSV")

    # --- Summary ---
    section("GAP SUMMARY")
    gaps = []

    # Phase 2 gaps
    for target in TARGETS:
        for source in SOURCES:
            sub = df[(df["architecture"] == "catboost") & (df["target_variable"] == target) & (df["data_source"] == source)]
            n = len(set(sub["station_id"].unique()))
            if n < 28:
                gaps.append(f"Phase 2: {target} + {source} — {28-n} stations missing")

    # Phase 3 gaps
    p3 = df[(df["data_source"] == "historical_forecast") & (df["train_start"].str.contains("2022"))]
    for target in TARGETS:
        for arch in ARCHITECTURES:
            sub = p3[(p3["target_variable"] == target) & (p3["architecture"] == arch)]
            n = len(set(sub["station_id"].unique()))
            if n < 28:
                gaps.append(f"Phase 3: {target} + {arch} + HF 3yr — {28-n} stations missing")

    # Phase 4 gaps
    p4 = df[(df["data_source"] == "historical_forecast")]
    for label, prefix in WINDOWS.items():
        sub = p4[(p4["architecture"] == "catboost") & (p4["target_variable"] == "MaxT") & (p4["train_start"].str.startswith(prefix))]
        n = len(set(sub["station_id"].unique()))
        if n < 28:
            gaps.append(f"Phase 4: MaxT + catboost + {label} — {28-n} stations missing")

    if not gaps:
        print("  No gaps found — all core experiments are covered!")
    else:
        print(f"  {len(gaps)} gaps found:\n")
        for g in gaps:
            print(f"    - {g}")

    print()


def print_phase_summary_no_data():
    print(f"""
  Experiments needed:
    Phase 2 (data source):      28 × 2 × 2 × CatBoost  = 112
    Phase 3 (architecture):     28 × 5 × 2 × HF         = 280
    Phase 4 (training window):  28 × 6 × CatBoost × MaxT = 168
    ---
    Total:                                                  560

  Run commands:
    python3 scripts/research/run_all_experiments.py --phase 2
    python3 scripts/research/run_all_experiments.py --phase 3
    python3 scripts/research/run_all_experiments.py --phase 4
""")


if __name__ == "__main__":
    main()
