#!/usr/bin/env python3
"""
Generate per-station and per-climate baseline comparison tables + skill score figure.

Computes raw GFS MAE, ECMWF MAE, blend MAE, and best model MAE per station,
then outputs:
  - table6b_baseline_per_station.csv  (per-station)
  - table6c_baseline_per_climate.csv  (per-climate-type)
  - fig5b_skill_score_by_climate.{pdf,png}

Usage:
  python3 scripts/research/generate_baseline_comparison.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sqlite3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.research.experiment_runner import load_forecast_data, load_actuals
from src.models.preprocessing import engineer_features
from src.research.stations import STATIONS_RESEARCH

DB_PATH = Path("data/research/results.db")
OUT_TABLES = Path("docs/paper/tables")
OUT_FIGURES = Path("docs/paper/figures")

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")

TEST_START = "2025-07-01"
TEST_END = "2026-02-12"


def get_best_model_mae():
    """Best model MAE per (station, target) from the DB."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT station_id, target_variable, MIN(mae) as model_mae, climate_type
        FROM experiments
        WHERE data_source = 'historical_forecast'
          AND train_start LIKE '2022%'
        GROUP BY station_id, target_variable
    """, conn)
    conn.close()
    return df


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    best_df = get_best_model_mae()
    rows = []
    stations = list(STATIONS_RESEARCH.keys())
    total = len(stations)

    for i, station_id in enumerate(stations):
        info = STATIONS_RESEARCH[station_id]
        try:
            forecasts = load_forecast_data(station_id, "historical_forecast", "2022-07-01", TEST_END)
            actuals = load_actuals(station_id)
            merged = actuals.join(forecasts, how="inner")
            merged = merged[merged["MaxT"].notna() & merged["MinT"].notna()]
            df_feat = engineer_features(merged, drop_redundant=True, mode=None, verbose=False)
            test = df_feat[(df_feat.index >= TEST_START) & (df_feat.index <= TEST_END)]
            if len(test) < 10:
                continue

            for target in ["MaxT", "MinT"]:
                y = test[target]
                gfs_col = "GFS_MaxT" if target == "MaxT" else "GFS_MinT"
                ec_col = "EC_MaxT" if target == "MaxT" else "EC_MinT"
                blend_col = f"Forecast_{target}"

                gfs_mae = float(np.abs(test[gfs_col] - y).mean())
                blend_mae = float(np.abs(test[blend_col] - y).mean())

                ec_valid = test[[ec_col, target]].dropna()
                ec_mae = float(np.abs(ec_valid[ec_col] - ec_valid[target]).mean()) if len(ec_valid) > 10 else np.nan

                m = best_df[(best_df["station_id"] == station_id) & (best_df["target_variable"] == target)]
                model_mae = float(m["model_mae"].iloc[0]) if len(m) else np.nan

                skill_vs_gfs = 1 - (model_mae / gfs_mae) if gfs_mae > 0 else np.nan
                skill_vs_ec = 1 - (model_mae / ec_mae) if ec_mae and ec_mae > 0 else np.nan
                improv_vs_gfs_pct = ((gfs_mae - model_mae) / gfs_mae * 100) if gfs_mae > 0 else np.nan

                rows.append({
                    "station_id": station_id,
                    "climate_type": info["climate"],
                    "target": target,
                    "gfs_mae": round(gfs_mae, 3),
                    "ecmwf_mae": round(ec_mae, 3) if not np.isnan(ec_mae) else np.nan,
                    "blend_mae": round(blend_mae, 3),
                    "model_mae": round(model_mae, 3),
                    "skill_vs_gfs": round(skill_vs_gfs, 3),
                    "skill_vs_ecmwf": round(skill_vs_ec, 3) if not np.isnan(skill_vs_ec) else np.nan,
                    "improvement_vs_gfs_pct": round(improv_vs_gfs_pct, 1),
                })
        except Exception as e:
            print(f"  SKIP {station_id}: {e}")
        print(f"  [{i+1}/{total}] {station_id}")

    df = pd.DataFrame(rows)

    # Per-station table
    df.to_csv(OUT_TABLES / "table6b_baseline_per_station.csv", index=False)
    print(f"\n  Per-station: {OUT_TABLES / 'table6b_baseline_per_station.csv'}")

    # Per-climate table
    climate = df.groupby(["climate_type", "target"]).agg({
        "gfs_mae": "mean", "ecmwf_mae": "mean", "blend_mae": "mean",
        "model_mae": "mean", "skill_vs_gfs": "mean", "improvement_vs_gfs_pct": "mean",
    }).round(3).reset_index()
    climate.to_csv(OUT_TABLES / "table6c_baseline_per_climate.csv", index=False)
    print(f"  Per-climate: {OUT_TABLES / 'table6c_baseline_per_climate.csv'}")
    print(climate.to_string(index=False))

    # Skill score figure by climate type
    climate_order = ["Continental", "NE Coastal", "SE Subtropical", "Gulf/SC", "Pacific", "Arid"]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"MaxT": "#e74c3c", "MinT": "#3498db"}
    x = np.arange(len(climate_order))
    w = 0.35
    for i, target in enumerate(["MaxT", "MinT"]):
        sub = climate[climate["target"] == target].set_index("climate_type")
        vals = [sub.loc[c, "skill_vs_gfs"] if c in sub.index else 0 for c in climate_order]
        ax.bar(x + i * w, vals, w, label=target, color=colors[target], edgecolor="white")
    ax.set_ylabel("Skill Score (1 - model/GFS)")
    ax.set_xlabel("Climate Type")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(climate_order, rotation=20, ha="right")
    ax.legend()
    ax.set_title("MOS Skill Score vs Raw GFS by Climate Type")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig5b_skill_score_by_climate.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  Figure 5b: {OUT_FIGURES / 'fig5b_skill_score_by_climate.pdf'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
