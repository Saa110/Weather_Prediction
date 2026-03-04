#!/usr/bin/env python3
"""
Phase D: Training Window Sensitivity for MinT
28 stations × 5 architectures × 6 windows × Historical Forecast × MinT = 840 experiments

Progress is printed in real-time:
  - Per-experiment status (OK/FAIL with MAE)
  - Per-station summary table after each station completes
  - Running totals, elapsed time, and ETA
  - Final cross-station summary tables
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import warnings, time
warnings.filterwarnings('ignore')

from src.research.experiment_runner import run_experiment
from src.research.stations import STATIONS_RESEARCH

# ── Configuration ──────────────────────────────────────────────────────────
TARGET = "MinT"
SOURCE = "historical_forecast"
ARCHITECTURES = ["catboost", "xgboost", "lightgbm", "linear", "mlp"]
WINDOWS = [
    ("2024-07-01", "1yr"),
    ("2023-07-01", "2yr"),
    ("2022-07-01", "3yr"),
    ("2021-07-01", "4yr"),
    ("2020-07-01", "5yr"),
    ("2018-01-01", "8yr"),
]
STATIONS = list(STATIONS_RESEARCH.keys())

# ── Bookkeeping ────────────────────────────────────────────────────────────
total_experiments = len(STATIONS) * len(ARCHITECTURES) * len(WINDOWS)
completed = 0
failed = 0
t_start = time.time()
results = {}   # (station, arch, window_label) -> MAE

# ── Banner ─────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  PHASE D: Training Window Sensitivity | Historical Forecast | {TARGET}")
print(f"  {len(STATIONS)} stations × {len(ARCHITECTURES)} architectures × {len(WINDOWS)} windows = {total_experiments} experiments")
print(f"{'='*80}\n")
sys.stdout.flush()

# ── Main loop ──────────────────────────────────────────────────────────────
for si, station_id in enumerate(STATIONS):
    climate = STATIONS_RESEARCH[station_id]["climate"]
    city    = STATIONS_RESEARCH[station_id].get("name", station_id)
    station_start = time.time()
    station_ok = 0
    station_fail = 0

    for train_start, wlabel in WINDOWS:
        for arch in ARCHITECTURES:
            completed += 1
            try:
                r = run_experiment(
                    station_id, SOURCE, arch, TARGET,
                    train_start, save_to_db=True, verbose=False
                )
                mae = r["metrics"]["mae"]
                results[(station_id, arch, wlabel)] = mae
                station_ok += 1
            except Exception as e:
                failed += 1
                station_fail += 1
                results[(station_id, arch, wlabel)] = None
                print(f"  [FAIL] {station_id} {arch} {wlabel}: {str(e)[:80]}")
                sys.stdout.flush()

    # ── Per-station summary ────────────────────────────────────────────────
    station_elapsed = time.time() - station_start
    total_elapsed   = time.time() - t_start
    remaining       = (total_elapsed / (si + 1)) * (len(STATIONS) - si - 1)

    pct = completed / total_experiments * 100
    bar_len = 30
    filled  = int(bar_len * completed / total_experiments)
    bar     = '█' * filled + '░' * (bar_len - filled)

    print(f"\n┌─ [{si+1}/{len(STATIONS)}] {station_id} ({city}, {climate})")
    print(f"│  {bar} {pct:.0f}%  |  {completed}/{total_experiments}  |  "
          f"{station_elapsed:.0f}s  |  Elapsed: {total_elapsed/60:.1f}min  |  ETA: {remaining/60:.1f}min")

    # Table header
    header = f"│  {'Window':<8}"
    for arch in ARCHITECTURES:
        header += f" {arch:>9}"
    header += "  | Best"
    print(header)
    print(f"│  {'─'*8}" + "─"*10*len(ARCHITECTURES) + "──┼──────────────")

    for _, wlabel in WINDOWS:
        row = f"│  {wlabel:<8}"
        best_arch, best_val = None, 999
        for arch in ARCHITECTURES:
            mae = results.get((station_id, arch, wlabel))
            if mae is not None:
                row += f" {mae:>9.2f}"
                if mae < best_val:
                    best_val, best_arch = mae, arch
            else:
                row += f" {'FAIL':>9}"
        if best_arch:
            row += f"  | {best_arch} {best_val:.2f}"
        print(row)

    # Best overall for this station
    valid = [(a, w, results[(station_id, a, w)])
             for _, w in WINDOWS for a in ARCHITECTURES
             if results.get((station_id, a, w)) is not None]
    if valid:
        best = min(valid, key=lambda x: x[2])
        print(f"│  ★ Station best: {best[0]} + {best[1]} = {best[2]:.2f}°F MAE")

    fail_note = f"  ({station_fail} failed)" if station_fail else ""
    print(f"└─ {station_ok} OK{fail_note}")
    sys.stdout.flush()

# ── Final summary ──────────────────────────────────────────────────────────
total_elapsed = time.time() - t_start
print(f"\n{'='*80}")
print(f"  PHASE D COMPLETE: {completed} experiments in {total_elapsed/60:.1f}min ({failed} failed)")
print(f"{'='*80}")

# Cross-station average MAE per architecture × window
print(f"\n  AVERAGE MAE (°F) ACROSS 28 STATIONS — {TARGET}:")
header = f"  {'Window':<8}"
for arch in ARCHITECTURES:
    header += f" {arch:>9}"
header += "  | Best"
print(header)
print(f"  {'─'*8}" + "─"*10*len(ARCHITECTURES) + "──┼──────────────")

for _, wlabel in WINDOWS:
    row = f"  {wlabel:<8}"
    best_arch, best_avg = None, 999
    for arch in ARCHITECTURES:
        maes = [results[(s, arch, wlabel)] for s in STATIONS
                if results.get((s, arch, wlabel)) is not None]
        if maes:
            avg = sum(maes) / len(maes)
            row += f" {avg:>9.2f}"
            if avg < best_avg:
                best_avg, best_arch = avg, arch
        else:
            row += f" {'N/A':>9}"
    if best_arch:
        row += f"  | {best_arch} {best_avg:.2f}"
    print(row)

# Best window per architecture
print(f"\n  BEST WINDOW PER ARCHITECTURE:")
for arch in ARCHITECTURES:
    best_w, best_mae = None, 999
    for _, wlabel in WINDOWS:
        maes = [results[(s, arch, wlabel)] for s in STATIONS
                if results.get((s, arch, wlabel)) is not None]
        if maes:
            avg = sum(maes) / len(maes)
            if avg < best_mae:
                best_mae, best_w = avg, wlabel
    print(f"    {arch:<12} → {best_w} ({best_mae:.2f}°F)")

# Overall best combo
print(f"\n  OVERALL BEST COMBO:")
best_combo = min(
    [(a, w) for a in ARCHITECTURES for _, w in WINDOWS],
    key=lambda aw: sum(results.get((s, aw[0], aw[1]) , 999) for s in STATIONS) / len(STATIONS)
)
avg = sum(results.get((s, best_combo[0], best_combo[1]), 999) for s in STATIONS) / len(STATIONS)
print(f"    {best_combo[0]} + {best_combo[1]} window = {avg:.2f}°F avg MAE\n")
