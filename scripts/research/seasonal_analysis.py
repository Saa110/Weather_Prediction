#!/usr/bin/env python3
"""
Seasonal breakdown of MAE: compute per-month absolute errors for the best
model configuration (linear, 3yr HF) across all 28 stations.

Output:
  docs/paper/tables/table9_seasonal_mae.csv
  docs/paper/figures/fig10_seasonal_mae.{pdf,png}

Usage:
  python3 scripts/research/seasonal_analysis.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.research.experiment_runner import run_experiment
from src.research.stations import STATIONS_RESEARCH

OUT_TABLES = Path("docs/paper/tables")
OUT_FIGURES = Path("docs/paper/figures")

SEASON_MAP = {7: "Summer", 8: "Summer", 9: "Summer",
              10: "Fall", 11: "Fall", 12: "Fall",
              1: "Winter", 2: "Winter"}
SEASON_ORDER = ["Summer", "Fall", "Winter"]

plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)

    stations = list(STATIONS_RESEARCH.keys())
    all_rows = []
    total = len(stations) * 2
    done = 0

    for station_id in stations:
        for target in ["MaxT", "MinT"]:
            done += 1
            try:
                r = run_experiment(
                    station_id, "historical_forecast", "linear", target,
                    "2022-07-01", save_to_db=False, verbose=False,
                    n_bootstrap=50, return_predictions=True,
                )
                preds = r["predictions"]
                actuals = r["actuals"]
                abs_err = np.abs(preds["q50"].values - actuals.values)
                dates = actuals.index
                for date, err in zip(dates, abs_err):
                    month = date.month
                    season = SEASON_MAP.get(month, "Other")
                    all_rows.append({
                        "station_id": station_id,
                        "target": target,
                        "date": str(date.date()),
                        "month": month,
                        "season": season,
                        "abs_error": err,
                    })
                print(f"  [{done}/{total}] {station_id} {target} OK")
            except Exception as e:
                print(f"  [{done}/{total}] {station_id} {target} FAIL: {e}")

    df = pd.DataFrame(all_rows)

    # Table 9: mean MAE by season and target
    table = df.groupby(["target", "season"])["abs_error"].agg(["mean", "std", "count"]).reset_index()
    table.columns = ["target", "season", "mean_mae", "std", "n_days"]
    table["se"] = table["std"] / np.sqrt(table["n_days"])
    table = table.round(3)
    table.to_csv(OUT_TABLES / "table9_seasonal_mae.csv", index=False)
    print(f"\n  Table 9: {OUT_TABLES / 'table9_seasonal_mae.csv'}")
    print(table.to_string(index=False))

    # Figure 10: grouped bar chart — MAE by season, split by target
    fig, ax = plt.subplots(figsize=(6, 4))
    pivot_mean = df.groupby(["target", "season"])["abs_error"].mean().unstack()
    pivot_se = df.groupby(["target", "season"])["abs_error"].sem().unstack()
    pivot_mean = pivot_mean[SEASON_ORDER]
    pivot_se = pivot_se[SEASON_ORDER]
    x = np.arange(len(SEASON_ORDER))
    w = 0.3
    colors = {"MaxT": "#e74c3c", "MinT": "#3498db"}
    for i, target in enumerate(["MaxT", "MinT"]):
        if target in pivot_mean.index:
            ax.bar(x + i * w, pivot_mean.loc[target], w,
                   yerr=pivot_se.loc[target], capsize=4,
                   color=colors[target], label=target, edgecolor="white")
    ax.set_ylabel("MAE (°F)")
    ax.set_xlabel("Season")
    ax.set_xticks(x + w / 2)
    ax.set_xticklabels(SEASON_ORDER)
    ax.legend(title="Target")
    ax.set_title("Seasonal MAE breakdown (Linear, 3yr HF, 28 stations)")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig10_seasonal_mae.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  Figure 10: {OUT_FIGURES / 'fig10_seasonal_mae.pdf'}")

    # Also generate per-month figure
    fig, ax = plt.subplots(figsize=(7, 4))
    month_order = [7, 8, 9, 10, 11, 12, 1, 2]
    month_labels = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb"]
    for target in ["MaxT", "MinT"]:
        sub = df[df["target"] == target]
        monthly = sub.groupby("month")["abs_error"].agg(["mean", "sem"]).reindex(month_order)
        ax.plot(range(len(month_order)), monthly["mean"].values, marker="o",
                label=target, color=colors[target])
        ax.fill_between(range(len(month_order)),
                        (monthly["mean"] - monthly["sem"]).values,
                        (monthly["mean"] + monthly["sem"]).values,
                        alpha=0.15, color=colors[target])
    ax.set_ylabel("MAE (°F)")
    ax.set_xlabel("Month")
    ax.set_xticks(range(len(month_order)))
    ax.set_xticklabels(month_labels)
    ax.legend(title="Target")
    ax.set_title("Monthly MAE (Linear, 3yr HF, 28 stations)")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig10b_monthly_mae.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  Figure 10b: {OUT_FIGURES / 'fig10b_monthly_mae.pdf'}")

    print("\nDone.")


if __name__ == "__main__":
    main()
