#!/usr/bin/env python3
"""
Generate a comprehensive per-station analysis document.

Produces docs/paper/STATION_ANALYSIS.md covering all 28 stations individually,
grouped by climate zone. Includes model performance, GFS deviation, feature
ablation, seasonal MAE, and auto-generated insights.

Usage:
  .venv/bin/python scripts/research/generate_station_report.py
  .venv/bin/python scripts/research/generate_station_report.py --skip-seasonal
"""

import sys
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.research.stations import STATIONS_RESEARCH, CLIMATE_TYPES
from src.research.ablation_config import FEATURE_GROUPS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DB_PATH = PROJECT_ROOT / "data/research/results.db"
BASELINE_CSV = PROJECT_ROOT / "docs/paper/tables/table6b_baseline_per_station.csv"
ABLATION_CSV = PROJECT_ROOT / "docs/analysis/ablation_results.csv"
SEASONAL_CACHE = PROJECT_ROOT / "docs/paper/tables/table9_seasonal_per_station.csv"
OUT_PATH = PROJECT_ROOT / "docs/paper/STATION_ANALYSIS.md"

CLIMATE_ORDER = [
    "Continental",
    "NE Coastal",
    "SE Subtropical",
    "Gulf/SC",
    "Pacific",
    "Arid",
]

STATION_ORDER_BY_CLIMATE = {
    "Continental": ["KORD", "KMDW", "KMSP", "KDTW", "KDEN", "KOKC"],
    "NE Coastal": ["KNYC", "KLGA", "KBOS", "KPHL", "KDCA"],
    "SE Subtropical": ["KATL", "KCLT", "KBNA", "KJAX", "KTPA", "KMIA"],
    "Gulf/SC": ["KHOU", "KMSY", "KDFW", "KDAL", "KAUS", "KSAT"],
    "Pacific": ["KLAX", "KSFO", "KSEA"],
    "Arid": ["KPHX", "KLAS"],
}

ARCH_ORDER = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
ARCH_LABELS = {
    "linear": "Linear",
    "mlp": "MLP",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

SEASON_MAP = {7: "Summer", 8: "Summer", 9: "Summer",
              10: "Fall", 11: "Fall", 12: "Fall",
              1: "Winter", 2: "Winter"}
SEASON_ORDER = ["Summer", "Fall", "Winter"]

ABLATION_GROUP_ORDER = ["Bias", "ECMWF", "Lags", "Rolling",
                        "NWP_atmosphere", "Physics", "Time", "NWP_primary"]


# ===================================================================
# DATA LOADING
# ===================================================================

def load_experiments() -> pd.DataFrame:
    """Load all experiments from the SQLite DB, deduplicated."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM experiments ORDER BY run_timestamp", conn)
    conn.close()
    key = ["station_id", "data_source", "architecture", "target_variable", "train_start"]
    df = df.drop_duplicates(subset=key, keep="last")
    return df


def load_baseline() -> pd.DataFrame:
    """Load per-station GFS/ECMWF/Blend baseline comparison."""
    return pd.read_csv(BASELINE_CSV)


def load_ablation() -> pd.DataFrame:
    """Load per-station ablation results."""
    return pd.read_csv(ABLATION_CSV)


def compute_seasonal_mae(skip: bool = False) -> pd.DataFrame:
    """
    Compute per-station seasonal MAE. Caches to CSV.

    Runs linear model (3yr HF) for each station/target, collects daily errors,
    splits by season.
    """
    if SEASONAL_CACHE.exists() and not skip:
        df = pd.read_csv(SEASONAL_CACHE)
        if len(df) > 0:
            print(f"  Loaded cached seasonal data: {len(df)} rows")
            return df

    if skip:
        print("  Skipping seasonal computation (--skip-seasonal)")
        return pd.DataFrame()

    from src.research.experiment_runner import run_experiment

    all_rows = []
    stations = list(STATIONS_RESEARCH.keys())
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
                    season = SEASON_MAP.get(month)
                    if season is None:
                        continue
                    all_rows.append({
                        "station_id": station_id,
                        "target": target,
                        "date": str(date.date()),
                        "month": month,
                        "season": season,
                        "abs_error": err,
                    })
                print(f"  [{done}/{total}] {station_id} {target} OK "
                      f"(MAE={r['metrics']['mae']:.3f})")
            except Exception as e:
                print(f"  [{done}/{total}] {station_id} {target} FAIL: {e}")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        SEASONAL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SEASONAL_CACHE, index=False)
        print(f"  Cached seasonal data: {SEASONAL_CACHE}")
    return df


def get_3yr_experiments(df_exp: pd.DataFrame) -> pd.DataFrame:
    """Filter experiments to 3yr training window for both data sources."""
    hf = df_exp[
        (df_exp["data_source"] == "historical_forecast") &
        (df_exp["train_start"].str.contains("2022-07", na=False))
    ]
    era = df_exp[
        (df_exp["data_source"] == "reanalysis") &
        (df_exp["train_start"].str.contains("2022", na=False))
    ]
    return pd.concat([hf, era], ignore_index=True)


# ===================================================================
# INSIGHT GENERATION
# ===================================================================

def compute_climate_means(df_exp_3yr: pd.DataFrame) -> pd.DataFrame:
    """Compute mean MAE per climate_type / target / data_source."""
    best_per_station = (
        df_exp_3yr.groupby(["station_id", "climate_type", "target_variable", "data_source"])
        ["mae"].min().reset_index()
    )
    climate_means = (
        best_per_station.groupby(["climate_type", "target_variable", "data_source"])
        ["mae"].mean().reset_index()
    )
    return climate_means


def station_insights(
    station_id: str,
    climate_type: str,
    station_exp: pd.DataFrame,
    baseline_row_maxt: Optional[pd.Series],
    baseline_row_mint: Optional[pd.Series],
    ablation_station: pd.DataFrame,
    seasonal_station: pd.DataFrame,
    climate_means: pd.DataFrame,
) -> List[str]:
    """Generate auto-commentary for a single station."""
    insights = []

    for target in ["MaxT", "MinT"]:
        hf_sub = station_exp[
            (station_exp["target_variable"] == target) &
            (station_exp["data_source"] == "historical_forecast")
        ]
        if hf_sub.empty:
            continue

        best_mae = hf_sub["mae"].min()
        best_arch = hf_sub.loc[hf_sub["mae"].idxmin(), "architecture"]

        cm = climate_means[
            (climate_means["climate_type"] == climate_type) &
            (climate_means["target_variable"] == target) &
            (climate_means["data_source"] == "historical_forecast")
        ]
        if not cm.empty:
            group_mean = cm["mae"].values[0]
            diff = best_mae - group_mean
            pct = (diff / group_mean) * 100 if group_mean > 0 else 0
            direction = "above" if diff > 0 else "below"
            if abs(pct) > 10:
                insights.append(
                    f"**{target}**: Best MAE ({best_mae:.3f} F, {ARCH_LABELS[best_arch]}) "
                    f"is {abs(pct):.1f}% {direction} the {climate_type} group mean "
                    f"({group_mean:.3f} F)."
                )

    brow_maxt = baseline_row_maxt
    if brow_maxt is not None and not brow_maxt.empty:
        imp = brow_maxt.get("improvement_vs_gfs_pct")
        if imp is not None:
            val = float(imp.iloc[0]) if hasattr(imp, "iloc") else float(imp)
            if val > 50:
                insights.append(
                    f"MaxT post-processing delivers exceptional {val:.1f}% improvement over raw GFS."
                )
            elif val < 5:
                insights.append(
                    f"MaxT shows minimal improvement ({val:.1f}%) over raw GFS -- "
                    "GFS already performs well here."
                )

    brow_mint = baseline_row_mint
    if brow_mint is not None and not brow_mint.empty:
        imp = brow_mint.get("improvement_vs_gfs_pct")
        if imp is not None:
            val = float(imp.iloc[0]) if hasattr(imp, "iloc") else float(imp)
            if val > 50:
                insights.append(
                    f"MinT post-processing delivers exceptional {val:.1f}% improvement over raw GFS."
                )
            elif val < 5:
                insights.append(
                    f"MinT shows minimal improvement ({val:.1f}%) over raw GFS."
                )

    if not ablation_station.empty:
        for target in ["MaxT", "MinT"]:
            ab_t = ablation_station[ablation_station["target_variable"] == target]
            if ab_t.empty:
                continue
            avg_delta = ab_t.groupby("ablation_group")["delta_mae"].mean()
            most_important = avg_delta.idxmax()
            if avg_delta[most_important] > 0.05:
                insights.append(
                    f"**{target} ablation**: Removing **{most_important}** causes the "
                    f"largest degradation (+{avg_delta[most_important]:.3f} F avg across architectures)."
                )
            harmful = avg_delta[avg_delta < -0.02]
            if not harmful.empty:
                groups = ", ".join(harmful.index.tolist())
                insights.append(
                    f"**{target} ablation**: Removing {groups} actually *improves* "
                    "performance (possible overfitting from these features)."
                )

    if not seasonal_station.empty:
        for target in ["MaxT", "MinT"]:
            s_t = seasonal_station[seasonal_station["target"] == target]
            if s_t.empty:
                continue
            season_mae = s_t.groupby("season")["abs_error"].mean()
            if len(season_mae) >= 2:
                best_season = season_mae.idxmin()
                worst_season = season_mae.idxmax()
                if season_mae[worst_season] > season_mae[best_season] * 1.3:
                    insights.append(
                        f"**{target} seasonality**: {worst_season} MAE "
                        f"({season_mae[worst_season]:.3f} F) is "
                        f"{(season_mae[worst_season]/season_mae[best_season] - 1)*100:.0f}% "
                        f"worse than {best_season} ({season_mae[best_season]:.3f} F)."
                    )

    return insights


# ===================================================================
# MARKDOWN GENERATION
# ===================================================================

def _md_table(headers: List[str], rows: List[List[str]], bold_min_col: Optional[int] = None) -> str:
    """Build a Markdown table. Optionally bold the minimum numeric value in a column per-row group."""
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _fmt(v, decimals=3):
    """Format a float, returning '--' for NaN/None."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}"


def generate_executive_summary(
    df_exp: pd.DataFrame,
    df_3yr: pd.DataFrame,
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
    seasonal: pd.DataFrame,
) -> str:
    """Generate the executive summary section."""
    n_exp = len(df_exp)
    n_stations = df_exp["station_id"].nunique()
    n_arch = df_exp["architecture"].nunique()

    hf_3yr = df_3yr[df_3yr["data_source"] == "historical_forecast"]
    best_per_station = hf_3yr.groupby(["station_id", "target_variable"])["mae"].min().reset_index()

    overall_maxt = best_per_station[best_per_station["target_variable"] == "MaxT"]["mae"]
    overall_mint = best_per_station[best_per_station["target_variable"] == "MinT"]["mae"]

    best_arch_maxt = (
        hf_3yr[hf_3yr["target_variable"] == "MaxT"]
        .groupby("architecture")["mae"].mean().idxmin()
    )
    best_arch_mint = (
        hf_3yr[hf_3yr["target_variable"] == "MinT"]
        .groupby("architecture")["mae"].mean().idxmin()
    )

    gfs_imp_maxt = baseline[baseline["target"] == "MaxT"]["improvement_vs_gfs_pct"].mean()
    gfs_imp_mint = baseline[baseline["target"] == "MinT"]["improvement_vs_gfs_pct"].mean()

    lines = [
        "# Comprehensive Per-Station Analysis",
        "",
        "*Auto-generated on " + datetime.now().strftime("%Y-%m-%d %H:%M") + "*",
        "",
        "## Executive Summary",
        "",
        f"This document presents a detailed, station-by-station analysis of MOS "
        f"(Model Output Statistics) post-processing performance across all "
        f"**{n_stations} Kalshi-tradeable U.S. temperature stations**, spanning "
        f"**{len(CLIMATE_ORDER)} climate zones**. The analysis draws from "
        f"**{n_exp:,} experiments** covering {n_arch} ML architectures "
        f"(Linear, MLP, XGBoost, LightGBM, CatBoost), 2 training data sources "
        f"(Historical Forecasts vs. ERA5 Reanalysis), and a fixed test period "
        f"of July 2025 -- February 2026 (~227 days).",
        "",
        "### Key Findings",
        "",
        f"- **Overall MaxT MAE**: {overall_maxt.mean():.3f} F "
        f"(range: {overall_maxt.min():.3f} -- {overall_maxt.max():.3f} F). "
        f"Best architecture on average: **{ARCH_LABELS[best_arch_maxt]}**.",
        f"- **Overall MinT MAE**: {overall_mint.mean():.3f} F "
        f"(range: {overall_mint.min():.3f} -- {overall_mint.max():.3f} F). "
        f"Best architecture on average: **{ARCH_LABELS[best_arch_mint]}**.",
        f"- **GFS improvement**: Post-processing reduces MaxT error by "
        f"**{gfs_imp_maxt:.1f}%** and MinT error by **{gfs_imp_mint:.1f}%** "
        f"on average vs. raw GFS.",
    ]

    if not ablation.empty:
        avg_delta = ablation.groupby("ablation_group")["delta_mae"].mean()
        top_group = avg_delta.idxmax()
        lines.append(
            f"- **Most critical feature group**: **{top_group}** "
            f"(avg +{avg_delta[top_group]:.3f} F when removed)."
        )

    if not seasonal.empty:
        season_mae = seasonal.groupby("season")["abs_error"].mean()
        worst = season_mae.idxmax()
        best = season_mae.idxmin()
        lines.append(
            f"- **Seasonal pattern**: {worst} is the hardest season "
            f"(MAE {season_mae[worst]:.3f} F), {best} the easiest "
            f"({season_mae[best]:.3f} F)."
        )

    lines += [
        "",
        "### Document Structure",
        "",
        "Stations are grouped by climate zone. For each station we present:",
        "",
        "1. **Model Performance** -- all 5 architectures on both Historical Forecast "
        "and ERA5 datasets (MAE, RMSE, Bias)",
        "2. **GFS Deviation** -- raw NWP baselines vs. best MOS model",
        "3. **Feature Ablation** -- impact of removing each feature group",
        "4. **Seasonal Performance** -- Summer / Fall / Winter MAE breakdown",
        "5. **Station Insights** -- auto-generated commentary on notable patterns",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def generate_station_section(
    station_id: str,
    station_info: dict,
    station_exp: pd.DataFrame,
    baseline: pd.DataFrame,
    ablation_station: pd.DataFrame,
    seasonal_station: pd.DataFrame,
    climate_means: pd.DataFrame,
    section_num: str,
) -> str:
    """Generate the full Markdown section for one station."""
    lines = []
    city = station_info.get("city", station_id)
    climate = station_info["climate"]

    lines.append(f"### {section_num} {station_id} -- {city}")
    lines.append("")
    lines.append(f"**Climate zone**: {climate} | "
                 f"**Coordinates**: {station_info['lat']:.2f} N, "
                 f"{abs(station_info['lon']):.2f} W | "
                 f"**Timezone**: {station_info.get('timezone', 'N/A')}")
    if station_info.get("note"):
        lines.append(f"  \n*Note: {station_info['note']}*")
    lines.append("")

    # --- Model performance table ---
    lines.append("#### Model Performance (All Architectures, Both Datasets)")
    lines.append("")

    for target in ["MaxT", "MinT"]:
        lines.append(f"**{target}**")
        lines.append("")
        headers = ["Architecture", "HF MAE", "HF RMSE", "HF Bias",
                    "ERA5 MAE", "ERA5 RMSE", "ERA5 Bias"]
        rows = []
        best_hf_mae = float("inf")
        best_hf_arch = ""

        for arch in ARCH_ORDER:
            hf = station_exp[
                (station_exp["architecture"] == arch) &
                (station_exp["data_source"] == "historical_forecast") &
                (station_exp["target_variable"] == target)
            ]
            era = station_exp[
                (station_exp["architecture"] == arch) &
                (station_exp["data_source"] == "reanalysis") &
                (station_exp["target_variable"] == target)
            ]

            hf_mae = hf["mae"].min() if not hf.empty else np.nan
            hf_rmse = hf.loc[hf["mae"].idxmin(), "rmse"] if not hf.empty else np.nan
            hf_bias = hf.loc[hf["mae"].idxmin(), "bias"] if not hf.empty else np.nan

            era_mae = era["mae"].min() if not era.empty else np.nan
            era_rmse = era.loc[era["mae"].idxmin(), "rmse"] if not era.empty else np.nan
            era_bias = era.loc[era["mae"].idxmin(), "bias"] if not era.empty else np.nan

            if not np.isnan(hf_mae) and hf_mae < best_hf_mae:
                best_hf_mae = hf_mae
                best_hf_arch = arch

            rows.append([
                ARCH_LABELS[arch],
                _fmt(hf_mae), _fmt(hf_rmse), _fmt(hf_bias, 2),
                _fmt(era_mae), _fmt(era_rmse), _fmt(era_bias, 2),
            ])

        for row in rows:
            if row[0] == ARCH_LABELS.get(best_hf_arch, ""):
                row[0] = f"**{row[0]}**"
                row[1] = f"**{row[1]}**"

        lines.append(_md_table(headers, rows))
        lines.append("")

    # --- GFS Deviation ---
    lines.append("#### GFS Deviation Analysis")
    lines.append("")

    bl_station = baseline[baseline["station_id"] == station_id]
    if not bl_station.empty:
        headers = ["Target", "GFS MAE", "ECMWF MAE", "Blend MAE",
                    "Model MAE", "Skill vs GFS", "Improvement %"]
        rows = []
        for target in ["MaxT", "MinT"]:
            bl_t = bl_station[bl_station["target"] == target]
            if bl_t.empty:
                continue
            r = bl_t.iloc[0]
            rows.append([
                target,
                _fmt(r["gfs_mae"]),
                _fmt(r["ecmwf_mae"]),
                _fmt(r["blend_mae"]),
                _fmt(r["model_mae"]),
                _fmt(r["skill_vs_gfs"], 2),
                f"{r['improvement_vs_gfs_pct']:.1f}%",
            ])
        lines.append(_md_table(headers, rows))
    else:
        lines.append("*Baseline data not available for this station.*")
    lines.append("")

    # --- Feature Ablation ---
    lines.append("#### Feature Ablation")
    lines.append("")

    if not ablation_station.empty:
        for target in ["MaxT", "MinT"]:
            ab_t = ablation_station[ablation_station["target_variable"] == target]
            if ab_t.empty:
                continue
            lines.append(f"**{target}** -- Mean delta-MAE when feature group is removed "
                         "(positive = group is helpful)")
            lines.append("")

            headers = ["Feature Group"] + [ARCH_LABELS[a] for a in ARCH_ORDER]
            rows = []
            groups_present = [g for g in ABLATION_GROUP_ORDER if g in ab_t["ablation_group"].values]
            for group in groups_present:
                row = [group]
                for arch in ARCH_ORDER:
                    cell = ab_t[
                        (ab_t["ablation_group"] == group) & (ab_t["architecture"] == arch)
                    ]
                    if not cell.empty:
                        val = cell["delta_mae"].values[0]
                        formatted = f"{val:+.3f}"
                        row.append(formatted)
                    else:
                        row.append("--")
                rows.append(row)

            lines.append(_md_table(headers, rows))
            lines.append("")
    else:
        lines.append("*Ablation data not available for this station.*")
        lines.append("")

    # --- Seasonal Performance ---
    lines.append("#### Seasonal Performance")
    lines.append("")

    if not seasonal_station.empty:
        headers = ["Target", "Summer MAE", "Fall MAE", "Winter MAE",
                    "Best Season", "Worst Season"]
        rows = []
        for target in ["MaxT", "MinT"]:
            s_t = seasonal_station[seasonal_station["target"] == target]
            if s_t.empty:
                rows.append([target, "--", "--", "--", "--", "--"])
                continue
            season_mae = s_t.groupby("season")["abs_error"].mean()
            summer = season_mae.get("Summer", np.nan)
            fall = season_mae.get("Fall", np.nan)
            winter = season_mae.get("Winter", np.nan)
            valid = {k: v for k, v in [("Summer", summer), ("Fall", fall),
                                         ("Winter", winter)] if not np.isnan(v)}
            best_s = min(valid, key=valid.get) if valid else "--"
            worst_s = max(valid, key=valid.get) if valid else "--"
            rows.append([
                target, _fmt(summer), _fmt(fall), _fmt(winter),
                best_s, worst_s,
            ])
        lines.append(_md_table(headers, rows))
    else:
        lines.append("*Seasonal data not available. Run without --skip-seasonal to compute.*")
    lines.append("")

    # --- Insights ---
    bl_maxt = bl_station[bl_station["target"] == "MaxT"] if not bl_station.empty else None
    bl_mint = bl_station[bl_station["target"] == "MinT"] if not bl_station.empty else None

    insight_list = station_insights(
        station_id, climate,
        station_exp, bl_maxt, bl_mint,
        ablation_station, seasonal_station, climate_means,
    )

    lines.append("#### Station Insights")
    lines.append("")
    if insight_list:
        for ins in insight_list:
            lines.append(f"- {ins}")
    else:
        lines.append("No notable deviations from climate-group patterns.")
    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def generate_climate_zone_header(climate: str, stations: List[str], num: int) -> str:
    """Generate climate zone section header with summary."""
    n = len(stations)
    station_list = ", ".join(stations)
    return "\n".join([
        f"## {num}. {climate} ({n} stations)",
        "",
        f"Stations: {station_list}",
        "",
    ])


def generate_cross_cutting_insights(
    df_3yr: pd.DataFrame,
    baseline: pd.DataFrame,
    ablation: pd.DataFrame,
    seasonal: pd.DataFrame,
) -> str:
    """Generate the final cross-cutting insights section."""
    lines = [
        "## Cross-Cutting Insights",
        "",
    ]

    # 1. Architecture winners by climate
    lines.append("### Architecture Recommendations by Climate Zone")
    lines.append("")

    hf = df_3yr[df_3yr["data_source"] == "historical_forecast"]
    headers = ["Climate Zone", "Best MaxT Arch", "Best MaxT MAE",
               "Best MinT Arch", "Best MinT MAE"]
    rows = []
    for climate in CLIMATE_ORDER:
        c_data = hf[hf["climate_type"] == climate]
        row = [climate]
        for target in ["MaxT", "MinT"]:
            t_data = c_data[c_data["target_variable"] == target]
            if t_data.empty:
                row += ["--", "--"]
                continue
            arch_means = t_data.groupby("architecture")["mae"].mean()
            best = arch_means.idxmin()
            row += [ARCH_LABELS[best], _fmt(arch_means[best])]
        rows.append(row)
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 2. HF vs ERA5 advantage by climate
    lines.append("### Historical Forecast vs. ERA5 Advantage by Climate Zone")
    lines.append("")

    headers = ["Climate Zone", "MaxT HF MAE", "MaxT ERA5 MAE", "MaxT Advantage",
               "MinT HF MAE", "MinT ERA5 MAE", "MinT Advantage"]
    rows = []
    for climate in CLIMATE_ORDER:
        row = [climate]
        for target in ["MaxT", "MinT"]:
            hf_c = df_3yr[
                (df_3yr["climate_type"] == climate) &
                (df_3yr["data_source"] == "historical_forecast") &
                (df_3yr["target_variable"] == target)
            ]
            era_c = df_3yr[
                (df_3yr["climate_type"] == climate) &
                (df_3yr["data_source"] == "reanalysis") &
                (df_3yr["target_variable"] == target)
            ]
            hf_mean = hf_c.groupby("station_id")["mae"].min().mean() if not hf_c.empty else np.nan
            era_mean = era_c.groupby("station_id")["mae"].min().mean() if not era_c.empty else np.nan
            if not np.isnan(hf_mean) and not np.isnan(era_mean):
                adv = ((era_mean - hf_mean) / era_mean * 100)
                row += [_fmt(hf_mean), _fmt(era_mean), f"{adv:.1f}%"]
            else:
                row += [_fmt(hf_mean), _fmt(era_mean), "--"]
        rows.append(row)
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 3. GFS improvement range by climate
    lines.append("### GFS Post-Processing Improvement by Climate Zone")
    lines.append("")

    headers = ["Climate Zone", "MaxT Avg Improvement", "MaxT Range",
               "MinT Avg Improvement", "MinT Range"]
    rows = []
    for climate in CLIMATE_ORDER:
        bl_c = baseline[baseline["climate_type"] == climate]
        row = [climate]
        for target in ["MaxT", "MinT"]:
            bl_t = bl_c[bl_c["target"] == target]
            if bl_t.empty:
                row += ["--", "--"]
                continue
            imp = bl_t["improvement_vs_gfs_pct"]
            row += [
                f"{imp.mean():.1f}%",
                f"{imp.min():.1f}% -- {imp.max():.1f}%",
            ]
        rows.append(row)
    lines.append(_md_table(headers, rows))
    lines.append("")

    # 4. Ablation patterns by climate
    if not ablation.empty:
        lines.append("### Feature Group Importance by Climate Zone")
        lines.append("")
        lines.append("Mean delta-MAE when each group is removed, averaged across "
                      "architectures and stations within each climate zone.")
        lines.append("")

        for target in ["MaxT", "MinT"]:
            lines.append(f"**{target}**")
            lines.append("")
            ab_t = ablation[ablation["target_variable"] == target]
            headers = ["Climate Zone"] + [g for g in ABLATION_GROUP_ORDER
                                          if g in ab_t["ablation_group"].values]
            rows = []
            for climate in CLIMATE_ORDER:
                stations_in = STATION_ORDER_BY_CLIMATE[climate]
                ab_c = ab_t[ab_t["station_id"].isin(stations_in)]
                row = [climate]
                for group in ABLATION_GROUP_ORDER:
                    if group not in ab_t["ablation_group"].values:
                        continue
                    val = ab_c[ab_c["ablation_group"] == group]["delta_mae"].mean()
                    row.append(f"{val:+.3f}" if not np.isnan(val) else "--")
                rows.append(row)
            lines.append(_md_table(headers, rows))
            lines.append("")

    # 5. Seasonal vulnerability by climate
    if not seasonal.empty:
        lines.append("### Seasonal Vulnerability by Climate Zone")
        lines.append("")
        lines.append("Mean MAE by season for each climate zone (Linear model, 3yr HF).")
        lines.append("")

        headers = ["Climate Zone", "Target", "Summer", "Fall", "Winter", "Worst/Best Ratio"]
        rows = []
        for climate in CLIMATE_ORDER:
            stations_in = STATION_ORDER_BY_CLIMATE[climate]
            s_c = seasonal[seasonal["station_id"].isin(stations_in)]
            for target in ["MaxT", "MinT"]:
                s_t = s_c[s_c["target"] == target]
                if s_t.empty:
                    rows.append([climate, target, "--", "--", "--", "--"])
                    continue
                season_mae = s_t.groupby("season")["abs_error"].mean()
                vals = {s: season_mae.get(s, np.nan) for s in SEASON_ORDER}
                valid = {k: v for k, v in vals.items() if not np.isnan(v)}
                ratio = (max(valid.values()) / min(valid.values())
                         if len(valid) >= 2 and min(valid.values()) > 0 else np.nan)
                rows.append([
                    climate, target,
                    _fmt(vals["Summer"]), _fmt(vals["Fall"]), _fmt(vals["Winter"]),
                    _fmt(ratio, 2) if not np.isnan(ratio) else "--",
                ])
        lines.append(_md_table(headers, rows))
        lines.append("")

    # 6. Hardest and easiest stations
    lines.append("### Hardest and Easiest Stations to Predict")
    lines.append("")

    best_per = hf.groupby(["station_id", "target_variable"])["mae"].min().reset_index()
    for target in ["MaxT", "MinT"]:
        bp_t = best_per[best_per["target_variable"] == target].sort_values("mae")
        top5 = bp_t.head(5)
        bot5 = bp_t.tail(5).sort_values("mae", ascending=False)

        lines.append(f"**{target} -- Easiest (lowest MAE)**")
        lines.append("")
        for _, r in top5.iterrows():
            sid = r["station_id"]
            city = STATIONS_RESEARCH[sid].get("city", sid)
            lines.append(f"1. {sid} ({city}): {r['mae']:.3f} F")
        lines.append("")

        lines.append(f"**{target} -- Hardest (highest MAE)**")
        lines.append("")
        for _, r in bot5.iterrows():
            sid = r["station_id"]
            city = STATIONS_RESEARCH[sid].get("city", sid)
            lines.append(f"1. {sid} ({city}): {r['mae']:.3f} F")
        lines.append("")

    return "\n".join(lines)


# ===================================================================
# MAIN
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate per-station analysis document")
    parser.add_argument("--skip-seasonal", action="store_true",
                        help="Skip seasonal MAE computation (use cached or leave blank)")
    args = parser.parse_args()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Comprehensive Per-Station Analysis Generator")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading experiment data from database...")
    df_exp = load_experiments()
    print(f"  {len(df_exp)} experiments loaded")

    df_3yr = get_3yr_experiments(df_exp)
    print(f"  {len(df_3yr)} experiments in 3yr window")

    print("\n[2/5] Loading baseline (GFS deviation) data...")
    baseline = load_baseline()
    print(f"  {len(baseline)} baseline rows")

    print("\n[3/5] Loading ablation data...")
    ablation = load_ablation()
    print(f"  {len(ablation)} ablation rows")

    print("\n[4/5] Computing per-station seasonal MAE...")
    seasonal = compute_seasonal_mae(skip=args.skip_seasonal)
    print(f"  {len(seasonal)} seasonal rows")

    climate_means = compute_climate_means(df_3yr)

    # 2. Generate document
    print("\n[5/5] Generating Markdown document...")
    doc_parts = []

    doc_parts.append(generate_executive_summary(df_exp, df_3yr, baseline, ablation, seasonal))

    climate_num = 0
    station_counter = 0
    for climate in CLIMATE_ORDER:
        climate_num += 1
        stations = STATION_ORDER_BY_CLIMATE[climate]
        doc_parts.append(generate_climate_zone_header(climate, stations, climate_num))

        for sid in stations:
            station_counter += 1
            info = STATIONS_RESEARCH[sid]
            section_num = f"{climate_num}.{stations.index(sid) + 1}"

            station_exp = df_3yr[df_3yr["station_id"] == sid]
            abl_station = ablation[ablation["station_id"] == sid]
            seas_station = seasonal[seasonal["station_id"] == sid] if not seasonal.empty else pd.DataFrame()

            doc_parts.append(generate_station_section(
                sid, info, station_exp, baseline, abl_station, seas_station,
                climate_means, section_num,
            ))

            if station_counter % 7 == 0:
                print(f"  ... {station_counter}/28 stations")

    doc_parts.append(generate_cross_cutting_insights(df_3yr, baseline, ablation, seasonal))

    # 3. Write output
    full_doc = "\n".join(doc_parts)
    OUT_PATH.write_text(full_doc, encoding="utf-8")

    n_lines = full_doc.count("\n") + 1
    print(f"\n  Output: {OUT_PATH}")
    print(f"  Size:   {n_lines} lines, {len(full_doc):,} characters")
    print("\nDone.")


if __name__ == "__main__":
    main()
