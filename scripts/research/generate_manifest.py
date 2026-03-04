#!/usr/bin/env python3
"""
Generate the experiment manifest JSON for all phases.

Run this once to create configs/experiment_manifest.json.
Re-run if station list or experiment design changes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from src.research.stations import get_all_station_ids

STATIONS = get_all_station_ids()
TRAIN_END = "2025-06-30"

manifest = {}

# ---------------------------------------------------------------------------
# Phase 2: Source Comparison (RQ1 + RQ2)
# 28 stations x 2 sources x 4 variables = 224
# ---------------------------------------------------------------------------
phase2 = []
for station in STATIONS:
    for source in ["historical_forecast", "reanalysis"]:
        for target in ["MaxT", "MinT"]:
            phase2.append({
                "station": station,
                "source": source,
                "arch": "catboost",
                "target": target,
                "train_start": "2022-01-01",
                "train_end": TRAIN_END,
            })
manifest["2"] = phase2

# ---------------------------------------------------------------------------
# Phase 3: Architecture Comparison (RQ3)
# 28 stations x 2 sources x 5 architectures x 1 variable = 280
# (CatBoost MaxT overlaps with Phase 2 -- runner will skip duplicates)
# ---------------------------------------------------------------------------
phase3 = []
for station in STATIONS:
    for source in ["historical_forecast", "reanalysis"]:
        for arch in ["catboost", "xgboost", "lightgbm", "linear", "mlp"]:
            phase3.append({
                "station": station,
                "source": source,
                "arch": arch,
                "target": "MaxT",
                "train_start": "2022-01-01",
                "train_end": TRAIN_END,
            })
manifest["3"] = phase3

# ---------------------------------------------------------------------------
# Phase 4: Training Data Volume (RQ4)
# 28 stations x 6 windows x 1 arch x 1 variable = 168
# ---------------------------------------------------------------------------
phase4 = []
windows = [
    ("2024-07-01", "1yr"),
    ("2023-07-01", "2yr"),
    ("2022-07-01", "3yr"),
    ("2021-07-01", "4yr"),
    ("2020-07-01", "5yr"),
    ("2018-01-01", "8yr"),
]
for station in STATIONS:
    for start, label in windows:
        phase4.append({
            "station": station,
            "source": "historical_forecast",
            "arch": "catboost",
            "target": "MaxT",
            "train_start": start,
            "train_end": TRAIN_END,
        })
manifest["4"] = phase4

# ---------------------------------------------------------------------------
# Phase 5: Variable Generality (RQ5)
# 28 stations x 2 sources x 2 new variables = 112
# (MaxT/MinT covered by Phase 2)
# ---------------------------------------------------------------------------
# NOTE: Wind and DewPoint targets require separate actuals data.
# These experiments will be generated but may fail until Phase 5.1-5.2 are done.
phase5 = []
for station in STATIONS:
    for source in ["historical_forecast", "reanalysis"]:
        for target in ["MaxT", "MinT"]:
            phase5.append({
                "station": station,
                "source": source,
                "arch": "catboost",
                "target": target,
                "train_start": "2022-01-01",
                "train_end": TRAIN_END,
            })
manifest["5"] = phase5

# ---------------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------------
out_path = Path("configs/experiment_manifest.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)

# Summary
total = sum(len(v) for v in manifest.values())
print(f"Manifest written to {out_path}")
for phase, exps in sorted(manifest.items()):
    print(f"  Phase {phase}: {len(exps)} experiments")
print(f"  Total: {total} experiments")
