#!/usr/bin/env python3
"""
Generate all tables and figures for the paper from the experiment database.

Output:
  docs/paper/tables/  — CSV and LaTeX tables
  docs/paper/figures/ — PDF and PNG figures (300 dpi)

Run from project root:
  python scripts/research/generate_paper_assets.py
"""

import sys
from pathlib import Path
import json
import sqlite3

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as sp_stats
from src.research.stations import STATIONS_RESEARCH

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data/research/results.db"
FEATURE_IMPORTANCE_CSV = PROJECT_ROOT / "docs/analysis/feature_importance_raw.csv"
ABLATION_CSV = PROJECT_ROOT / "docs/analysis/ablation_results.csv"
OUT_TABLES = PROJECT_ROOT / "docs/paper/tables"
OUT_FIGURES = PROJECT_ROOT / "docs/paper/figures"

# Style for publication
plt.rcParams["font.size"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")
COLORS = {"catboost": "#2ecc71", "xgboost": "#3498db", "lightgbm": "#9b59b6", "linear": "#e74c3c", "mlp": "#f39c12"}


def load_deduped_experiments():
    """Load experiments, one row per (station, data_source, architecture, target_variable, train_start)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM experiments ORDER BY run_timestamp", conn)
    conn.close()
    # Keep last run per unique combo (or min MAE — last is simpler)
    key = ["station_id", "data_source", "architecture", "target_variable", "train_start"]
    df = df.drop_duplicates(subset=key, keep="last")
    return df


def table1_stations():
    """Table 1: Station list."""
    rows = []
    for sid, info in STATIONS_RESEARCH.items():
        rows.append({
            "station_id": sid,
            "city": info.get("city", sid),
            "climate_type": info["climate"],
            "lat": info["lat"],
            "lon": info["lon"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLES / "table1_stations.csv", index=False)
    return df


def table2_data_source(df):
    """Table 2: HF vs ERA5 across all 5 architectures (3yr window) with paired tests.

    HF data comes from Phase 3 runs (train_start 2022-07).
    ERA5 data comes from Phase 2 runs (train_start 2022-01 for catboost, 2022-01 for others).
    Both use ~3yr windows so the comparison is fair.
    """
    arch_order = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
    rows = []
    for target in ["MaxT", "MinT"]:
        for arch in arch_order:
            hf_sub = df[(df["architecture"] == arch) &
                        (df["data_source"] == "historical_forecast") &
                        (df["train_start"].str.contains("2022")) &
                        (df["target_variable"] == target)]
            era_sub = df[(df["architecture"] == arch) &
                         (df["data_source"] == "reanalysis") &
                         (df["train_start"].str.contains("2022")) &
                         (df["target_variable"] == target)]
            hf = hf_sub.groupby("station_id")["mae"].min()
            era = era_sub.groupby("station_id")["mae"].min()
            common = hf.index.intersection(era.index)
            if len(common) == 0:
                continue
            hf_vals, era_vals = hf.loc[common].values, era.loc[common].values
            n = len(common)
            hf_mean, era_mean = hf_vals.mean(), era_vals.mean()
            hf_se = hf_vals.std() / np.sqrt(n)
            era_se = era_vals.std() / np.sqrt(n)
            if n >= 10:
                _, p = sp_stats.wilcoxon(era_vals - hf_vals, alternative="greater")
            else:
                p = np.nan
            improvement_pct = ((era_mean - hf_mean) / era_mean * 100) if era_mean > 0 else 0
            rows.append({
                "target": target, "architecture": arch,
                "hf_mean_mae": round(hf_mean, 3), "hf_se": round(hf_se, 3),
                "era5_mean_mae": round(era_mean, 3), "era5_se": round(era_se, 3),
                "improvement_pct": round(improvement_pct, 1),
                "wilcoxon_p": round(p, 6) if not np.isnan(p) else np.nan,
                "n_stations": n,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "table2_data_source_comparison.csv", index=False)
    return out


def table3_architecture(df):
    """Table 3: Architecture comparison (3yr HF) with SE and pairwise tests."""
    sub = df[(df["data_source"] == "historical_forecast") & (df["train_start"].str.contains("2022-07"))]
    arch_order = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
    rows = []
    for target in ["MaxT", "MinT"]:
        t = sub[sub["target_variable"] == target]
        for arch in arch_order:
            a = t[t["architecture"] == arch]
            vals = a.set_index("station_id")["mae"]
            rows.append({
                "target": target, "architecture": arch,
                "mean_mae": round(vals.mean(), 3),
                "se": round(vals.std() / np.sqrt(len(vals)), 3),
                "n_stations": len(vals),
            })
    out = pd.DataFrame(rows)
    # Pairwise Wilcoxon: best arch vs each other (per target)
    pairwise_rows = []
    for target in ["MaxT", "MinT"]:
        t = sub[sub["target_variable"] == target]
        means = t.groupby("architecture")["mae"].mean()
        best_arch = means.idxmin()
        best = t[t["architecture"] == best_arch].set_index("station_id")["mae"]
        for arch in arch_order:
            if arch == best_arch:
                pairwise_rows.append({"target": target, "comparison": f"{best_arch} vs {arch}", "p_value": np.nan, "note": "reference"})
                continue
            other = t[t["architecture"] == arch].set_index("station_id")["mae"]
            common = best.index.intersection(other.index)
            if len(common) >= 10:
                _, p = sp_stats.wilcoxon(other.loc[common].values - best.loc[common].values, alternative="greater")
            else:
                p = np.nan
            pairwise_rows.append({"target": target, "comparison": f"{best_arch} vs {arch}", "p_value": round(p, 6)})
    out.to_csv(OUT_TABLES / "table3_architecture_comparison.csv", index=False)
    pd.DataFrame(pairwise_rows).to_csv(OUT_TABLES / "table3_pairwise_tests.csv", index=False)
    return out


def table4_5_window_sensitivity(df):
    """Tables 4 & 5: Window sensitivity MaxT / MinT."""
    sub = df[df["data_source"] == "historical_forecast"].copy()
    sub["window"] = sub["train_start"].map({
        "2024-07-01": "1yr", "2023-07-01": "2yr", "2022-07-01": "3yr",
        "2021-07-01": "4yr", "2020-07-01": "5yr", "2018-01-01": "8yr",
    })
    sub = sub[sub["window"].notna()]

    for target in ["MaxT", "MinT"]:
        s = sub[sub["target_variable"] == target]
        mat = s.groupby(["architecture", "window"])["mae"].mean().unstack()
        mat = mat.reindex(columns=["1yr", "2yr", "3yr", "4yr", "5yr", "8yr"])
        fname = "table4_window_sensitivity_maxt.csv" if target == "MaxT" else "table5_window_sensitivity_mint.csv"
        mat.to_csv(OUT_TABLES / fname)
    return sub


def table6_baseline(df):
    """Table 6: GFS, ECMWF, blend, best model — computed from raw forecast data.

    If raw data files are not available (e.g. download_all_stations.py has not been
    run), falls back to the pre-bundled CSV tables if they exist.
    """
    existing = OUT_TABLES / "table6_baseline_comparison.csv"

    import warnings
    warnings.filterwarnings("ignore")
    try:
        from src.research.experiment_runner import load_forecast_data, load_actuals
        from src.models.preprocessing import engineer_features
    except ImportError:
        if existing.exists():
            print("  (table6: using pre-bundled CSV — raw data loaders unavailable)")
            return pd.read_csv(existing, index_col=0)
        print("  (Skip table6: raw data unavailable)")
        return None

    test_start, test_end = "2025-07-01", "2026-02-12"
    rows = []
    for station_id in STATIONS_RESEARCH:
        try:
            forecasts = load_forecast_data(station_id, "historical_forecast", "2022-07-01", test_end)
            actuals = load_actuals(station_id)
            merged = actuals.join(forecasts, how="inner")
            merged = merged[merged["MaxT"].notna() & merged["MinT"].notna()]
            df_feat = engineer_features(merged, drop_redundant=True, mode=None, verbose=False)
            test = df_feat[(df_feat.index >= test_start) & (df_feat.index <= test_end)]
            if len(test) < 10:
                continue
            for target in ["MaxT", "MinT"]:
                y = test[target]
                gfs_col = "GFS_MaxT" if target == "MaxT" else "GFS_MinT"
                ec_col = "EC_MaxT" if target == "MaxT" else "EC_MinT"
                blend_col = f"Forecast_{target}"
                gfs_mae = np.abs(test[gfs_col] - y).mean()
                ec_valid = test[[ec_col, target]].dropna()
                ec_mae = np.abs(ec_valid[ec_col] - ec_valid[target]).mean() if len(ec_valid) > 10 else np.nan
                blend_mae = np.abs(test[blend_col] - y).mean()
                rows.append({"station": station_id, "target": target, "gfs_mae": gfs_mae, "ecmwf_mae": ec_mae, "blend_mae": blend_mae})
        except Exception:
            pass

    if not rows:
        if existing.exists():
            print("  (table6: using pre-bundled CSV — raw data not downloaded)")
            return pd.read_csv(existing, index_col=0)
        print("  (Skip table6: no raw data available)")
        return None

    base_df = pd.DataFrame(rows)
    best = df[(df["data_source"] == "historical_forecast") & (df["train_start"].str.contains("2022-07"))]
    best = best.groupby(["station_id", "target_variable"])["mae"].min().reset_index()
    best = best.rename(columns={"station_id": "station", "target_variable": "target", "mae": "model_mae"})
    merged = base_df.merge(best, on=["station", "target"], how="left")
    summary = merged.groupby("target").agg({"gfs_mae": "mean", "ecmwf_mae": "mean", "blend_mae": "mean", "model_mae": "mean"}).round(3)
    sig_rows = []
    for target in ["MaxT", "MinT"]:
        m = merged[merged["target"] == target].dropna(subset=["blend_mae", "model_mae"])
        if len(m) >= 10:
            _, p = sp_stats.wilcoxon(m["blend_mae"].values - m["model_mae"].values, alternative="greater")
        else:
            p = np.nan
        sig_rows.append({"target": target, "comparison": "blend_vs_model", "p_value": round(p, 6),
                         "mean_improvement": round((m["blend_mae"] - m["model_mae"]).mean(), 3),
                         "n_stations": len(m)})
    summary.to_csv(OUT_TABLES / "table6_baseline_comparison.csv")
    pd.DataFrame(sig_rows).to_csv(OUT_TABLES / "table6_significance.csv", index=False)
    return summary


def table7_feature_importance():
    """Table 7: Top features from feature_importance_raw.csv."""
    if not FEATURE_IMPORTANCE_CSV.exists():
        print("  (Skip table7: feature_importance_raw.csv not found)")
        return None
    df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    top = df.groupby(["target", "feature"])["avg"].mean().reset_index()
    top = top.sort_values(["target", "avg"], ascending=[True, False])
    top = top.groupby("target").head(15)
    top.to_csv(OUT_TABLES / "table7_feature_importance.csv", index=False)
    return top


def table_supp_per_station(df):
    """Supplementary: per-station best MAE."""
    sub = df[(df["data_source"] == "historical_forecast") & (df["train_start"].str.contains("2022-07"))]
    best = sub.loc[sub.groupby(["station_id", "target_variable"])["mae"].idxmin()]
    best = best[["station_id", "climate_type", "target_variable", "architecture", "mae"]]
    best.to_csv(OUT_TABLES / "table_supp_per_station_mae.csv", index=False)
    return best


def table_supp_rmse(df):
    """Supplementary: RMSE by architecture and data source (mirrors Table 2 layout)."""
    arch_order = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
    arch_labels = {"linear": "Linear", "mlp": "MLP", "xgboost": "XGBoost",
                   "lightgbm": "LightGBM", "catboost": "CatBoost"}
    rows = []
    for target in ["MaxT", "MinT"]:
        for arch in arch_order:
            hf_sub = df[(df["architecture"] == arch) &
                        (df["data_source"] == "historical_forecast") &
                        (df["train_start"].str.contains("2022")) &
                        (df["target_variable"] == target)]
            era_sub = df[(df["architecture"] == arch) &
                         (df["data_source"] == "reanalysis") &
                         (df["train_start"].str.contains("2022")) &
                         (df["target_variable"] == target)]
            hf = hf_sub.groupby("station_id")["rmse"].min()
            era = era_sub.groupby("station_id")["rmse"].min()
            if hf.empty or era.empty:
                continue
            hf_mean, era_mean = hf.mean(), era.mean()
            imp = ((era_mean - hf_mean) / era_mean * 100) if era_mean > 0 else 0
            rows.append({
                "target": target, "architecture": arch,
                "hf_rmse": round(hf_mean, 3),
                "era5_rmse": round(era_mean, 3),
                "improvement_pct": round(imp, 1),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_TABLES / "table_supp_rmse.csv", index=False)

    # Generate LaTeX fragment
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Supplementary: mean RMSE (\textdegree{}F) by architecture and data source",
        r"(3-year training window, 28~stations). Layout mirrors Table~2.}",
        r"\label{tab:supp_rmse}",
        r"\small",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Target & Architecture & HF RMSE & ERA5 RMSE & Improv.\ (\%) \\",
        r"\midrule",
    ]
    for target in ["MaxT", "MinT"]:
        t_rows = [r for r in rows if r["target"] == target]
        for i, r in enumerate(t_rows):
            prefix = r"\multirow{5}{*}{" + target + "}" if i == 0 else ""
            lines.append(
                f"  {prefix} & {arch_labels[r['architecture']]} & "
                f"{r['hf_rmse']:.3f} & {r['era5_rmse']:.3f} & "
                f"{r['improvement_pct']:.1f} \\\\"
            )
        if target == "MaxT":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT_TABLES / "table_supp_rmse.tex").write_text("\n".join(lines) + "\n")
    return out


# --- Figures ---
def fig1_data_source(df):
    """Figure 1: HF vs ERA5 grouped bar chart for all architectures with SE error bars."""
    arch_order = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
    arch_labels = {"linear": "Linear", "mlp": "MLP", "xgboost": "XGBoost",
                   "lightgbm": "LightGBM", "catboost": "CatBoost"}
    hf_color, era_color = "#3498db", "#e74c3c"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, target in zip(axes, ["MaxT", "MinT"]):
        hf_means, era_means, hf_ses, era_ses = [], [], [], []
        labels = []
        for arch in arch_order:
            hf_sub = df[(df["architecture"] == arch) &
                        (df["data_source"] == "historical_forecast") &
                        (df["train_start"].str.contains("2022")) &
                        (df["target_variable"] == target)]
            era_sub = df[(df["architecture"] == arch) &
                         (df["data_source"] == "reanalysis") &
                         (df["train_start"].str.contains("2022")) &
                         (df["target_variable"] == target)]
            hf_vals = hf_sub.groupby("station_id")["mae"].min().values
            era_vals = era_sub.groupby("station_id")["mae"].min().values
            if len(hf_vals) == 0 or len(era_vals) == 0:
                continue
            labels.append(arch_labels[arch])
            hf_means.append(np.mean(hf_vals))
            era_means.append(np.mean(era_vals))
            hf_ses.append(np.std(hf_vals) / np.sqrt(len(hf_vals)))
            era_ses.append(np.std(era_vals) / np.sqrt(len(era_vals)))
        x = np.arange(len(labels))
        w = 0.35
        ax.bar(x - w / 2, hf_means, w, yerr=hf_ses, capsize=3,
               color=hf_color, label="Historical Forecast", edgecolor="white")
        ax.bar(x + w / 2, era_means, w, yerr=era_ses, capsize=3,
               color=era_color, label="ERA5 Reanalysis", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(target)
        ax.set_ylabel("MAE (\u00b0F)" if target == "MaxT" else "")
    axes[0].legend(title="Data source", fontsize=8, loc="upper left")
    fig.suptitle("Data source comparison across architectures (3yr window, 28 stations)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig1_data_source_comparison.{ext}",
                    dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()


def fig2_architecture(df):
    """Figure 2: Architecture comparison with error bars."""
    sub = df[(df["data_source"] == "historical_forecast") & (df["train_start"].str.contains("2022-07"))]
    arch_order = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
    mean = sub.groupby(["target_variable", "architecture"])["mae"].mean().unstack()
    se = sub.groupby(["target_variable", "architecture"])["mae"].sem().unstack()
    mean = mean[[c for c in arch_order if c in mean.columns]]
    se = se[[c for c in arch_order if c in se.columns]]
    mean.columns = [c.capitalize() for c in mean.columns]
    se.columns = [c.capitalize() for c in se.columns]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(mean))
    n_arch = len(mean.columns)
    w = 0.8 / n_arch
    for i, col in enumerate(mean.columns):
        ax.bar(x + i * w, mean[col], w, yerr=se[col], capsize=3,
               color=COLORS.get(col.lower(), "#333"), label=col, edgecolor="white")
    ax.set_ylabel("MAE (°F)")
    ax.set_xlabel("Target")
    ax.set_xticks(x + w * (n_arch - 1) / 2)
    ax.set_xticklabels(mean.index)
    ax.legend(title="Architecture", fontsize=8)
    ax.set_title("Architecture comparison (3yr, Historical Forecast)")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig2_architecture_comparison.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()


def fig3_4_window_sensitivity(df):
    """Figures 3 & 4: MAE vs training window."""
    sub = df[(df["data_source"] == "historical_forecast")].copy()
    sub["window"] = sub["train_start"].map({
        "2024-07-01": "1yr", "2023-07-01": "2yr", "2022-07-01": "3yr",
        "2021-07-01": "4yr", "2020-07-01": "5yr", "2018-01-01": "8yr",
    })
    sub = sub[sub["window"].notna()]
    order = ["1yr", "2yr", "3yr", "4yr", "5yr", "8yr"]
    sub["window"] = pd.Categorical(sub["window"], categories=order, ordered=True)

    for target in ["MaxT", "MinT"]:
        s = sub[sub["target_variable"] == target]
        mean_df = s.groupby(["window", "architecture"])["mae"].mean().reset_index()
        se_df = s.groupby(["window", "architecture"])["mae"].sem().reset_index().rename(columns={"mae": "se"})
        agg = mean_df.merge(se_df, on=["window", "architecture"])
        fig, ax = plt.subplots(figsize=(6, 4))
        for arch in ["linear", "mlp", "xgboost", "lightgbm", "catboost"]:
            a = agg[agg["architecture"] == arch].sort_values("window")
            if len(a):
                xvals = np.arange(len(a))
                color = COLORS.get(arch, "#333")
                ax.plot(xvals, a["mae"].values, marker="o", label=arch.capitalize(), color=color)
                ax.fill_between(xvals,
                                (a["mae"] - a["se"]).values,
                                (a["mae"] + a["se"]).values,
                                alpha=0.15, color=color)
        ax.set_ylabel("MAE (°F)")
        ax.set_xlabel("Training window")
        ax.set_title(f"Training window sensitivity: {target}")
        ax.legend(fontsize=8)
        ax.set_xticks(range(6))
        ax.set_xticklabels(order)
        plt.tight_layout()
        n = "3" if target == "MaxT" else "4"
        for ext in ["pdf", "png"]:
            plt.savefig(OUT_FIGURES / f"fig{n}_window_sensitivity_{target.lower()}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        plt.close()


def fig5_baseline(df):
    """Figure 5: GFS, ECMWF, blend, best model — use per-station data for error bars."""
    summary_path = OUT_TABLES / "table6_baseline_comparison.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path, index_col=0)
    else:
        summary = table6_baseline(df)
    if summary is None or summary.empty:
        return
    plot_df = summary.reset_index()
    plot_df = plot_df.melt(id_vars="target", var_name="Baseline", value_name="MAE")
    plot_df["Baseline"] = plot_df["Baseline"].replace({"model_mae": "Best MOS", "gfs_mae": "GFS", "ecmwf_mae": "ECMWF", "blend_mae": "Blend"})
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(data=plot_df, x="target", y="MAE", hue="Baseline", palette="Set2", ax=ax, errwidth=1.5, capsize=0.08)
    ax.set_ylabel("MAE (°F)")
    ax.set_title("Raw NWP vs best MOS model (28 stations)")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig5_baseline_comparison.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()


def fig6_7_feature_importance():
    """Figures 6 & 7: Feature importance."""
    if not FEATURE_IMPORTANCE_CSV.exists():
        return
    df = pd.read_csv(FEATURE_IMPORTANCE_CSV)
    for target in ["MaxT", "MinT"]:
        sub = df[df["target"] == target].groupby("feature")["avg"].mean().sort_values(ascending=True).tail(15)
        fig, ax = plt.subplots(figsize=(6, 4))
        sub.plot(kind="barh", ax=ax, color="steelblue", width=0.7)
        ax.set_xlabel("Mean importance (%)")
        ax.set_title(f"Feature importance: {target}")
        plt.tight_layout()
        n = "6" if target == "MaxT" else "7"
        for ext in ["pdf", "png"]:
            plt.savefig(OUT_FIGURES / f"fig{n}_feature_importance_{target.lower()}.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
        plt.close()


def fig8_mae_by_climate(df):
    """Figure 8: MAE by climate type."""
    sub = df[(df["data_source"] == "historical_forecast") & (df["train_start"].str.contains("2022-07"))]
    best = sub.loc[sub.groupby(["station_id", "target_variable"])["mae"].idxmin()]
    best = best[["station_id", "climate_type", "target_variable", "mae"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=best, x="climate_type", y="mae", hue="target_variable", ax=ax, palette="Set1")
    ax.set_ylabel("MAE (°F)")
    ax.set_xlabel("Climate type")
    ax.tick_params(axis="x", rotation=25)
    ax.set_title("Best-model MAE by climate type")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig8_mae_by_climate_type.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()


def load_ablation_results():
    """Load ablation runs from DB or CSV."""
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT station_id, architecture, target_variable, ablation_group, mae, mae_baseline, delta_mae, n_features FROM ablation_runs", conn)
    except Exception:
        df = pd.DataFrame()
    conn.close()
    if df.empty and ABLATION_CSV.exists():
        df = pd.read_csv(ABLATION_CSV)
    return df


def table8_ablation(ablation_df):
    """Table 8: Mean ΔMAE by ablation group and architecture (MaxT and MinT) with significance."""
    if ablation_df is None or ablation_df.empty:
        return
    arch_order = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
    sig_rows = []
    for target in ["MaxT", "MinT"]:
        sub = ablation_df[ablation_df["target_variable"] == target]
        pivot = sub.groupby(["ablation_group", "architecture"])["delta_mae"].mean().unstack()
        pivot = pivot[[c for c in arch_order if c in pivot.columns]]
        pivot = pivot.round(3)
        pivot.to_csv(OUT_TABLES / f"table8_ablation_{target.lower()}.csv")
        # Per-cell Wilcoxon: is ΔMAE significantly > 0 across 28 stations?
        for group in pivot.index:
            for arch in pivot.columns:
                cell = sub[(sub["ablation_group"] == group) & (sub["architecture"] == arch)]["delta_mae"].dropna()
                if len(cell) >= 10:
                    try:
                        _, p = sp_stats.wilcoxon(cell.values, alternative="greater")
                    except ValueError:
                        p = np.nan
                else:
                    p = np.nan
                sig_rows.append({"target": target, "group": group, "architecture": arch,
                                 "mean_delta": round(cell.mean(), 3), "p_value": round(p, 6) if not np.isnan(p) else np.nan})
    pd.DataFrame(sig_rows).to_csv(OUT_TABLES / "table8_ablation_significance.csv", index=False)
    summary = ablation_df.groupby(["target_variable", "ablation_group"])["delta_mae"].mean().unstack().round(3)
    summary.to_csv(OUT_TABLES / "table8_ablation_summary.csv")


def fig9_ablation(ablation_df):
    """Figure 9: Heatmap of mean ΔMAE with significance markers."""
    if ablation_df is None or ablation_df.empty:
        return
    arch_order = ["linear", "mlp", "xgboost", "lightgbm", "catboost"]
    group_order = ["Bias", "ECMWF", "Lags", "Rolling", "NWP_atmosphere", "Physics", "Time", "NWP_primary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for ax, target in zip([ax1, ax2], ["MaxT", "MinT"]):
        sub = ablation_df[ablation_df["target_variable"] == target]
        pivot = sub.groupby(["ablation_group", "architecture"])["delta_mae"].mean().unstack()
        pivot = pivot.reindex([g for g in group_order if g in pivot.index])
        pivot = pivot[[c for c in arch_order if c in pivot.columns]]
        # Build annotation matrix with significance markers
        annot = pivot.copy().round(2).astype(str)
        for group in pivot.index:
            for arch in pivot.columns:
                cell = sub[(sub["ablation_group"] == group) & (sub["architecture"] == arch)]["delta_mae"].dropna()
                if len(cell) >= 10:
                    try:
                        _, p = sp_stats.wilcoxon(cell.values, alternative="greater")
                        stars = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
                    except ValueError:
                        stars = ""
                else:
                    stars = ""
                annot.loc[group, arch] = f"{pivot.loc[group, arch]:.2f}{stars}"
        pivot.columns = [c.capitalize() for c in pivot.columns]
        annot.columns = [c.capitalize() for c in annot.columns]
        sns.heatmap(pivot, ax=ax, cmap="RdYlGn_r", center=0, annot=annot, fmt="",
                    cbar_kws={"label": "Mean ΔMAE (°F)"}, annot_kws={"fontsize": 8})
        ax.set_title(f"{target}")
        ax.set_xlabel("")
    fig.suptitle("Feature ablation: mean ΔMAE when group removed (* p<.05, ** p<.01, *** p<.001)")
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(OUT_FIGURES / f"fig9_ablation_delta_mae.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()


def main():
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    print("Loading experiments (deduplicated)...")
    df = load_deduped_experiments()
    print(f"  {len(df)} unique experiments")

    print("Generating tables...")
    table1_stations()
    table2_data_source(df)
    table3_architecture(df)
    table4_5_window_sensitivity(df)
    table6_baseline(df)
    table7_feature_importance()
    table_supp_per_station(df)
    table_supp_rmse(df)
    ablation_df = load_ablation_results()
    if ablation_df is not None and not ablation_df.empty:
        table8_ablation(ablation_df)
    print(f"  → {OUT_TABLES}")

    print("Generating figures...")
    fig1_data_source(df)
    fig2_architecture(df)
    fig3_4_window_sensitivity(df)
    fig5_baseline(df)
    fig6_7_feature_importance()
    fig8_mae_by_climate(df)
    if ablation_df is not None and not ablation_df.empty:
        fig9_ablation(ablation_df)
    print(f"  → {OUT_FIGURES}")

    print("Done. Attach docs/paper/tables/* and docs/paper/figures/* to your paper.")


if __name__ == "__main__":
    main()
