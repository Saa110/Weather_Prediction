"""
SQLite results database for the research experiments.

Stores every experiment run as a single row with all parameters and metrics,
enabling easy querying, aggregation, and comparison across conditions.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

DB_PATH = Path("data/research/results.db")


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the results table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT    NOT NULL,

            -- experiment parameters
            station_id      TEXT    NOT NULL,
            climate_type    TEXT    NOT NULL,
            data_source     TEXT    NOT NULL,   -- 'historical_forecast' or 'reanalysis'
            architecture    TEXT    NOT NULL,   -- 'catboost', 'xgboost', etc.
            target_variable TEXT    NOT NULL,   -- 'MaxT', 'MinT', 'Wind', 'DewPoint'
            train_start     TEXT    NOT NULL,
            train_end       TEXT    NOT NULL,
            test_start      TEXT    NOT NULL,
            test_end        TEXT    NOT NULL,
            n_train         INTEGER NOT NULL,
            n_test          INTEGER NOT NULL,

            -- primary metrics
            mae             REAL    NOT NULL,
            rmse            REAL    NOT NULL,
            bias            REAL    NOT NULL,
            coverage_50     REAL,
            interval_width  REAL,

            -- bootstrap CIs (JSON: {"mae": [lo, hi], "rmse": [lo, hi], ...})
            bootstrap_ci    TEXT,

            -- raw NWP baseline (for skill score)
            raw_nwp_mae     REAL,
            skill_score     REAL,

            -- metadata
            n_features      INTEGER,
            training_secs   REAL,
            notes           TEXT
        )
    """)

    # Index for common queries
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_station_source
        ON experiments (station_id, data_source, architecture, target_variable)
    """)

    conn.commit()
    conn.close()
    init_ablation_table()


def init_ablation_table():
    """Create the ablation_runs table if it doesn't exist."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ablation_runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            run_timestamp   TEXT    NOT NULL,
            station_id      TEXT    NOT NULL,
            architecture    TEXT    NOT NULL,
            target_variable TEXT    NOT NULL,
            ablation_group  TEXT    NOT NULL,
            mae             REAL,
            mae_baseline    REAL,
            delta_mae       REAL,
            n_features      INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ablation_lookup
        ON ablation_runs (station_id, architecture, target_variable, ablation_group)
    """)
    conn.commit()
    conn.close()


def insert_ablation_results(rows: List[Dict]) -> int:
    """
    Bulk insert ablation results. Each row: station_id, architecture,
    target_variable, ablation_group, mae, mae_baseline, delta_mae, n_features.
    """
    if not rows:
        return 0
    conn = _get_conn()
    ts = datetime.now().isoformat()
    conn.executemany(
        """
        INSERT INTO ablation_runs (
            run_timestamp, station_id, architecture, target_variable,
            ablation_group, mae, mae_baseline, delta_mae, n_features
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                ts,
                str(r["station_id"]),
                str(r["architecture"]),
                str(r["target_variable"]),
                str(r["ablation_group"]),
                float(r["mae"]) if r.get("mae") is not None and pd.notna(r.get("mae")) else None,
                float(r["mae_baseline"]) if r.get("mae_baseline") is not None and pd.notna(r.get("mae_baseline")) else None,
                float(r["delta_mae"]) if r.get("delta_mae") is not None and pd.notna(r.get("delta_mae")) else None,
                int(r["n_features"]) if r.get("n_features") is not None and pd.notna(r.get("n_features")) else None,
            )
            for r in rows
        ],
    )
    conn.commit()
    n = len(rows)
    conn.close()
    return n


def insert_result(params: Dict, metrics: Dict, extra: Optional[Dict] = None):
    """
    Insert one experiment result into the database.

    Args:
        params: dict with keys station_id, climate_type, data_source,
                architecture, target_variable, train_start, train_end,
                test_start, test_end, n_train
        metrics: dict with keys mae, rmse, bias, coverage_50,
                 interval_width, n_test
        extra: optional dict with bootstrap_ci, raw_nwp_mae, skill_score,
               n_features, training_secs, notes
    """
    conn = _get_conn()
    extra = extra or {}

    bootstrap_ci_json = None
    if "bootstrap_ci" in extra:
        bootstrap_ci_json = json.dumps(extra["bootstrap_ci"])

    conn.execute("""
        INSERT INTO experiments (
            run_timestamp, station_id, climate_type, data_source,
            architecture, target_variable, train_start, train_end,
            test_start, test_end, n_train, n_test,
            mae, rmse, bias, coverage_50, interval_width,
            bootstrap_ci, raw_nwp_mae, skill_score,
            n_features, training_secs, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        params["station_id"],
        params["climate_type"],
        params["data_source"],
        params["architecture"],
        params["target_variable"],
        params["train_start"],
        params["train_end"],
        params["test_start"],
        params["test_end"],
        params["n_train"],
        metrics["n_test"],
        metrics["mae"],
        metrics["rmse"],
        metrics["bias"],
        metrics.get("coverage_50"),
        metrics.get("interval_width"),
        bootstrap_ci_json,
        extra.get("raw_nwp_mae"),
        extra.get("skill_score"),
        extra.get("n_features"),
        extra.get("training_secs"),
        extra.get("notes"),
    ))

    conn.commit()
    conn.close()


def query_results(
    station_id: Optional[str] = None,
    data_source: Optional[str] = None,
    architecture: Optional[str] = None,
    target_variable: Optional[str] = None,
    climate_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Query experiment results with optional filters.

    Returns:
        DataFrame of matching results
    """
    conn = _get_conn()

    clauses = []
    params = []

    if station_id:
        clauses.append("station_id = ?")
        params.append(station_id)
    if data_source:
        clauses.append("data_source = ?")
        params.append(data_source)
    if architecture:
        clauses.append("architecture = ?")
        params.append(architecture)
    if target_variable:
        clauses.append("target_variable = ?")
        params.append(target_variable)
    if climate_type:
        clauses.append("climate_type = ?")
        params.append(climate_type)

    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM experiments{where} ORDER BY run_timestamp"

    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


def get_all_results() -> pd.DataFrame:
    """Return all experiment results as a DataFrame."""
    conn = _get_conn()
    df = pd.read_sql_query("SELECT * FROM experiments ORDER BY run_timestamp", conn)
    conn.close()
    return df


def count_experiments() -> int:
    """Return total number of experiments in the database."""
    conn = _get_conn()
    cur = conn.execute("SELECT COUNT(*) FROM experiments")
    n = cur.fetchone()[0]
    conn.close()
    return n


# Initialise on import
init_db()
