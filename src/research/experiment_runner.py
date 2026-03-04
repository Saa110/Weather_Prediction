"""
Experiment runner -- the core engine of the research pipeline.

Takes a single experiment specification (station, source, architecture,
training window, target variable) and:
1. Loads and merges the data
2. Engineers features
3. Trains the model
4. Evaluates on the fixed test set
5. Computes bootstrap confidence intervals
6. Stores results in the SQLite database

Can be called programmatically or via the CLI script.
"""

import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.research.stations import STATIONS_RESEARCH
from src.research.models import build_model
from src.research.ablation_config import columns_to_drop_for_ablation
from src.research.statistics import compute_metrics, bootstrap_all_metrics, skill_score
from src.research.database import insert_result
from src.models.preprocessing import engineer_features
from src.calibration.calibration_conformal import ConformalCalibrator

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
HF_DIR = Path("data/research/historical_forecasts")
RE_DIR = Path("data/research/reanalysis")
ACTUALS_DIR = Path("data/research/actuals")

# Also check the existing pilot-study directories as fallback
HF_PILOT_DIR = Path("data/historical_forecasts")
ACTUALS_PILOT_DIR = Path("data/asos_historical")

# Fixed test period (same for every experiment)
TEST_START = "2025-07-01"
TEST_END = "2026-02-12"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _find_csv(directory: Path, pattern: str) -> Optional[Path]:
    """Find a CSV file matching a glob pattern, checking research dir then pilot dir."""
    matches = sorted(directory.glob(pattern))
    if matches:
        return matches[-1]  # latest file if multiple
    return None


def load_forecast_data(station_id: str, source: str, start: str, end: str) -> pd.DataFrame:
    """
    Load GFS and ECMWF forecast data for a station.

    Args:
        station_id: ICAO station code
        source: 'historical_forecast' or 'reanalysis'
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        Merged DataFrame with GFS and ECMWF columns
    """
    if source == "historical_forecast":
        base_dir = HF_DIR
        fallback_dir = HF_PILOT_DIR
    elif source == "reanalysis":
        base_dir = RE_DIR
        fallback_dir = None
    else:
        raise ValueError(f"Unknown source: {source}")

    # Try loading from multiple file chunks via glob matching
    gfs_parts, ec_parts = [], []

    # Expected file date ranges from our download scripts
    _KNOWN_CHUNKS = [
        ("2018-01-01", "2021-12-31"),
        ("2022-01-01", "2024-12-31"),
        ("2025-01-01", "2026-02-12"),
        ("2026-02-13", "2026-03-04"),
    ]

    search_dirs = [base_dir]
    if fallback_dir is not None:
        search_dirs.append(fallback_dir)

    if source == "reanalysis":
        # Reanalysis files are named {station}_REANALYSIS_{chunk}.csv
        # Load as a single source (no separate GFS/ECMWF)
        for search_dir in search_dirs:
            for chunk_start, chunk_end in _KNOWN_CHUNKS:
                fpath = search_dir / f"{station_id}_REANALYSIS_{chunk_start}_{chunk_end}.csv"
                if fpath.exists():
                    df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                    gfs_parts.append(df)
    else:
        # Historical forecast: load separate GFS and ECMWF files
        for search_dir in search_dirs:
            for model_type, parts_list in [("GFS", gfs_parts), ("ECMWF", ec_parts)]:
                for chunk_start, chunk_end in _KNOWN_CHUNKS:
                    fpath = search_dir / f"{station_id}_{model_type}_{chunk_start}_{chunk_end}.csv"
                    if fpath.exists():
                        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
                        parts_list.append(df)

    if not gfs_parts:
        raise FileNotFoundError(
            f"No {source} data found for {station_id}. "
            f"Checked: {search_dirs}"
        )

    gfs_all = pd.concat(gfs_parts).sort_index()
    gfs_all = gfs_all[~gfs_all.index.duplicated(keep="last")]
    gfs_all = gfs_all[start:end]

    # ECMWF is optional
    if ec_parts:
        ec_all = pd.concat(ec_parts).sort_index()
        ec_all = ec_all[~ec_all.index.duplicated(keep="last")]
        ec_all = ec_all[start:end]
    else:
        ec_all = pd.DataFrame(index=gfs_all.index)

    # Build merged DataFrame
    merged = pd.DataFrame(index=gfs_all.index)
    merged["GFS_MaxT"] = gfs_all.get("Forecast_MaxT")
    merged["GFS_MinT"] = gfs_all.get("Forecast_MinT")
    merged["Forecast_Wind"] = gfs_all.get("Forecast_Wind")
    merged["Forecast_Dir"] = gfs_all.get("Forecast_Dir")
    merged["Forecast_Solar"] = gfs_all.get("Forecast_Solar")
    merged["Forecast_Clouds"] = gfs_all.get("Forecast_Clouds")
    merged["Forecast_DewPoint"] = gfs_all.get("Forecast_DewPoint")
    merged["GFS_MeanT"] = gfs_all.get("Forecast_AirMass")

    if "Forecast_MaxT" in ec_all.columns:
        ec_aligned = ec_all.reindex(gfs_all.index)
        merged["EC_MaxT"] = ec_aligned.get("Forecast_MaxT")
        merged["EC_MinT"] = ec_aligned.get("Forecast_MinT")
        merged["EC_MeanT"] = ec_aligned.get("Forecast_AirMass")
        merged["Forecast_MaxT"] = merged[["GFS_MaxT", "EC_MaxT"]].mean(axis=1)
        merged["Forecast_MinT"] = merged[["GFS_MinT", "EC_MinT"]].mean(axis=1)
        merged["Forecast_Uncertainty"] = (merged["GFS_MaxT"] - merged["EC_MaxT"]).abs()
        merged["Forecast_MeanT"] = merged[["GFS_MeanT", "EC_MeanT"]].mean(axis=1)
    else:
        merged["Forecast_MaxT"] = merged["GFS_MaxT"]
        merged["Forecast_MinT"] = merged["GFS_MinT"]

    return merged


def load_actuals(station_id: str) -> pd.DataFrame:
    """Load actual MaxT/MinT observations from ACIS."""
    for directory in [ACTUALS_DIR, ACTUALS_PILOT_DIR]:
        fpath = directory / f"{station_id}_daily.csv"
        if fpath.exists():
            df = pd.read_csv(fpath, index_col=0, parse_dates=True)
            return df[["MaxT", "MinT"]].dropna()

    raise FileNotFoundError(
        f"No actuals found for {station_id}. "
        f"Checked: {ACTUALS_DIR}, {ACTUALS_PILOT_DIR}"
    )


# ---------------------------------------------------------------------------
# Main experiment function
# ---------------------------------------------------------------------------
def run_experiment(
    station_id: str,
    data_source: str,
    architecture: str,
    target_variable: str,
    train_start: str,
    train_end: str = "2025-06-30",
    test_start: str = TEST_START,
    test_end: str = TEST_END,
    n_bootstrap: int = 1000,
    save_to_db: bool = True,
    verbose: bool = False,
    ablation_drop_groups: Optional[List[str]] = None,
    return_predictions: bool = False,
) -> Dict:
    """
    Run a single experiment end-to-end.

    Args:
        station_id: ICAO code (e.g. 'KDEN')
        data_source: 'historical_forecast' or 'reanalysis'
        architecture: 'catboost', 'xgboost', 'lightgbm', 'linear', 'mlp'
        target_variable: 'MaxT' or 'MinT'
        train_start: Training period start (YYYY-MM-DD)
        train_end: Training period end
        test_start: Test period start (fixed: 2025-07-01)
        test_end: Test period end (fixed: 2026-02-12)
        n_bootstrap: Number of bootstrap resamples for CIs
        save_to_db: Whether to store results in SQLite
        verbose: Print progress
        ablation_drop_groups: If set (e.g. ["ECMWF"]), drop these feature groups
            before training (for ablation studies). Do not save to DB when used.

    Returns:
        Dict with all parameters, metrics, and bootstrap CIs
    """
    t0 = time.time()
    station_info = STATIONS_RESEARCH[station_id]

    if verbose:
        print(f"\n{'='*60}")
        print(f"  {station_id} | {data_source} | {architecture} | {target_variable}")
        print(f"  Train: {train_start} to {train_end}")
        print(f"  Test:  {test_start} to {test_end}")
        print(f"{'='*60}")

    # 1. Load data
    forecasts = load_forecast_data(station_id, data_source, train_start, test_end)
    actuals = load_actuals(station_id)
    merged = actuals.join(forecasts, how="inner")
    merged = merged[merged["MaxT"].notna() & merged["MinT"].notna()]

    if verbose:
        print(f"  Merged: {len(merged)} days ({merged.index.min().date()} to {merged.index.max().date()})")

    # 2. Engineer features
    df_feat = engineer_features(merged, drop_redundant=True, mode=None, verbose=verbose)

    # 2b. Ablation: drop feature groups if requested
    if ablation_drop_groups:
        all_drop = []
        for grp in ablation_drop_groups:
            all_drop.extend(columns_to_drop_for_ablation(grp, list(df_feat.columns)))
        cols = [c for c in set(all_drop) if c in df_feat.columns]
        if cols:
            df_feat = df_feat.drop(columns=cols)

    # 3. Split into train+val and test by date
    train_val_mask = df_feat.index < test_start
    test_mask = (df_feat.index >= test_start) & (df_feat.index <= test_end)

    df_train_val = df_feat[train_val_mask]
    df_test = df_feat[test_mask]

    if len(df_test) < 10:
        raise ValueError(
            f"Only {len(df_test)} test days for {station_id}. "
            f"Need data through {test_end}."
        )

    # 4. Train model (uses its own 60/20/20 split within train_val)
    model = build_model(architecture, station_id, target_variable)
    X_val_internal, y_val_internal = model.train(df_train_val, verbose=verbose)

    # 5. Calibrate on internal validation set
    val_preds = model.predict(X_val_internal)
    calibrator = ConformalCalibrator(target_coverage=0.50, verbose=False)
    calibrator.fit(val_preds, y_val_internal)

    # 6. Predict on fixed test set
    drop_cols = ["MaxT", "MinT", "Pcpn", "SnowDepth", "day_of_year"]
    X_test = df_test.drop(columns=[c for c in drop_cols if c in df_test.columns])
    y_test = df_test[target_variable]

    raw_preds = model.predict(X_test)
    cal_preds = calibrator.transform(raw_preds)

    # 7. Compute metrics
    metrics = compute_metrics(cal_preds, y_test)
    boot_cis = bootstrap_all_metrics(cal_preds, y_test, n_boot=n_bootstrap)

    # 8. Raw NWP baseline (forecast vs actual, no model)
    forecast_col = f"Forecast_{target_variable}" if f"Forecast_{target_variable}" in df_test.columns else None
    raw_nwp_mae = None
    ss = None
    if forecast_col:
        raw_nwp_mae = float(np.mean(np.abs(df_test[forecast_col] - y_test)))
        ss = skill_score(metrics["mae"], raw_nwp_mae)

    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Results:")
        print(f"    MAE:  {metrics['mae']:.2f} [{boot_cis['mae'][1]:.2f}, {boot_cis['mae'][2]:.2f}]")
        print(f"    RMSE: {metrics['rmse']:.2f}")
        print(f"    Bias: {metrics['bias']:+.2f}")
        if raw_nwp_mae:
            print(f"    Raw NWP MAE: {raw_nwp_mae:.2f}  |  Skill: {ss:.2%}")
        print(f"    Time: {elapsed:.1f}s")

    # 9. Store results
    result = {
        "params": {
            "station_id": station_id,
            "climate_type": station_info["climate"],
            "data_source": data_source,
            "architecture": architecture,
            "target_variable": target_variable,
            "train_start": str(df_train_val.index.min().date()),
            "train_end": str(df_train_val.index.max().date()),
            "test_start": str(y_test.index.min().date()),
            "test_end": str(y_test.index.max().date()),
            "n_train": len(df_train_val),
        },
        "metrics": metrics,
        "bootstrap_ci": {k: [v[1], v[2]] for k, v in boot_cis.items()},
        "extra": {
            "raw_nwp_mae": raw_nwp_mae,
            "skill_score": ss,
            "n_features": len(model.feature_names),
            "training_secs": elapsed,
            "bootstrap_ci": {k: [v[1], v[2]] for k, v in boot_cis.items()},
        },
    }

    if return_predictions:
        result["predictions"] = cal_preds
        result["actuals"] = y_test

    if save_to_db:
        insert_result(result["params"], result["metrics"], result["extra"])

    return result
