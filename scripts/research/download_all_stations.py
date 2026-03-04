#!/usr/bin/env python3
"""
Download all data for the research experiment: GFS forecasts, ECMWF forecasts,
reanalysis data, and ACIS actuals for all 28 stations.

Usage:
    python scripts/research/download_all_stations.py --what all
    python scripts/research/download_all_stations.py --what actuals
    python scripts/research/download_all_stations.py --what gfs_forecast
    python scripts/research/download_all_stations.py --what ecmwf_forecast
    python scripts/research/download_all_stations.py --what reanalysis
    python scripts/research/download_all_stations.py --station KDEN --what all
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import requests
import pandas as pd
import numpy as np
import time
import json
import logging
from datetime import datetime, timedelta

from src.research.stations import STATIONS_RESEARCH

# Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HF_DIR = Path("data/research/historical_forecasts")
RE_DIR = Path("data/research/reanalysis")
ACTUALS_DIR = Path("data/research/actuals")

for d in [HF_DIR, RE_DIR, ACTUALS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

ACIS_URL = "http://data.rcc-acis.org/StnData"


# ---------------------------------------------------------------------------
# ACIS Actuals
# ---------------------------------------------------------------------------
def download_actuals(station_id: str):
    """Download MaxT/MinT actuals from ACIS for a single station."""
    outfile = ACTUALS_DIR / f"{station_id}_daily.csv"
    if outfile.exists():
        logger.info(f"  {station_id} actuals already exist, skipping")
        return

    payload = {
        "sid": station_id,
        "sdate": "2017-01-01",
        "edate": datetime.now().strftime("%Y-%m-%d"),
        "elems": [
            {"name": "maxt", "interval": "dly"},
            {"name": "mint", "interval": "dly"},
        ],
        "meta": "name,state",
    }

    try:
        resp = requests.post(ACIS_URL, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        if "data" not in data:
            logger.error(f"  {station_id}: no 'data' key in ACIS response")
            return

        df = pd.DataFrame(data["data"], columns=["Date", "MaxT", "MinT"])
        df["Date"] = pd.to_datetime(df["Date"])
        for col in ["MaxT", "MinT"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.set_index("Date", inplace=True)
        df.dropna(subset=["MaxT"], inplace=True)

        df.to_csv(outfile)
        logger.info(f"  {station_id}: {len(df)} days saved to {outfile}")

    except Exception as e:
        logger.error(f"  {station_id} ACIS download failed: {e}")


# ---------------------------------------------------------------------------
# Open-Meteo Forecast / Reanalysis
# ---------------------------------------------------------------------------
def download_openmeteo(
    station_id: str,
    lat: float, lon: float, timezone: str,
    model: str, model_label: str,
    start_date: str, end_date: str,
    api_type: str,  # 'forecast' or 'reanalysis'
    output_dir: Path,
):
    """Download hourly data from Open-Meteo, aggregate to daily, save as CSV."""
    outfile = output_dir / f"{station_id}_{model_label}_{start_date}_{end_date}.csv"
    if outfile.exists():
        logger.info(f"  {station_id} {model_label} {start_date}-{end_date} already exists, skipping")
        return

    if api_type == "forecast":
        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"

    # Download in 1-year chunks
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_dfs = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=365), end)

        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": [
                "temperature_2m", "windspeed_10m", "winddirection_10m",
                "shortwave_radiation", "cloudcover", "dewpoint_2m",
            ],
            "timezone": timezone,
        }
        if api_type == "forecast":
            params["models"] = model
        else:
            # CRITICAL: Must explicitly request ERA5 for reanalysis.
            # Without this, the archive API defaults to "best_match" which
            # returns ECMWF IFS data — identical to the historical forecast API.
            params["models"] = "era5"

        try:
            resp = requests.get(url, params=params, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            if "hourly" not in data:
                logger.warning(f"  {station_id} {model_label}: no hourly data for {current.date()}-{chunk_end.date()}")
                current = chunk_end + timedelta(days=1)
                time.sleep(1)
                continue

            hourly = pd.DataFrame({
                "timestamp": pd.to_datetime(data["hourly"]["time"]),
                "temperature_2m": data["hourly"].get("temperature_2m"),
                "windspeed_10m": data["hourly"].get("windspeed_10m"),
                "winddirection_10m": data["hourly"].get("winddirection_10m"),
                "shortwave_radiation": data["hourly"].get("shortwave_radiation"),
                "cloudcover": data["hourly"].get("cloudcover"),
                "dewpoint_2m": data["hourly"].get("dewpoint_2m"),
            })
            hourly.set_index("timestamp", inplace=True)

            daily = pd.DataFrame({
                "Forecast_MaxT": hourly["temperature_2m"].resample("D").max(),
                "Forecast_MinT": hourly["temperature_2m"].resample("D").min(),
                "Forecast_AirMass": hourly["temperature_2m"].resample("D").mean(),
                "Forecast_Wind": hourly["windspeed_10m"].resample("D").mean(),
                "Forecast_Dir": hourly["winddirection_10m"].resample("D").mean(),
                "Forecast_Solar": hourly["shortwave_radiation"].resample("D").mean(),
                "Forecast_Clouds": hourly["cloudcover"].resample("D").mean(),
                "Forecast_DewPoint": hourly["dewpoint_2m"].resample("D").mean(),
            })
            all_dfs.append(daily)
            logger.info(f"    {current.date()} to {chunk_end.date()}: {len(daily)} days")

        except Exception as e:
            logger.error(f"    {current.date()}-{chunk_end.date()} failed: {e}")

        time.sleep(1)  # rate limit
        current = chunk_end + timedelta(days=1)

    if all_dfs:
        full = pd.concat(all_dfs).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        full.to_csv(outfile)
        logger.info(f"  {station_id} {model_label}: {len(full)} days -> {outfile}")
    else:
        logger.error(f"  {station_id} {model_label}: NO DATA downloaded")


def download_station_forecasts(station_id: str, info: dict, what: str):
    """Download all forecast/reanalysis data for one station."""
    lat, lon, tz = info["lat"], info["lon"], info["timezone"]

    # Time periods
    periods = [
        ("2018-01-01", "2021-12-31"),
        ("2022-01-01", "2024-12-31"),
        ("2025-01-01", "2026-02-12"),
    ]

    if what in ("all", "gfs_forecast"):
        logger.info(f"  Downloading GFS historical forecasts for {station_id}...")
        for start, end in periods:
            download_openmeteo(
                station_id, lat, lon, tz,
                model="gfs_seamless", model_label="GFS",
                start_date=start, end_date=end,
                api_type="forecast", output_dir=HF_DIR,
            )

    if what in ("all", "ecmwf_forecast"):
        logger.info(f"  Downloading ECMWF historical forecasts for {station_id}...")
        for start, end in periods:
            # Use ecmwf_ifs for pre-2024, ecmwf_ifs025 for 2024+
            if start < "2024-01-01":
                ecmwf_model = "ecmwf_ifs"
            else:
                ecmwf_model = "ecmwf_ifs025"
            download_openmeteo(
                station_id, lat, lon, tz,
                model=ecmwf_model, model_label="ECMWF",
                start_date=start, end_date=end,
                api_type="forecast", output_dir=HF_DIR,
            )

    if what in ("all", "reanalysis"):
        logger.info(f"  Downloading reanalysis data for {station_id}...")
        for start, end in periods:
            download_openmeteo(
                station_id, lat, lon, tz,
                model="", model_label="REANALYSIS",
                start_date=start, end_date=end,
                api_type="reanalysis", output_dir=RE_DIR,
            )


def main():
    parser = argparse.ArgumentParser(description="Download research data for all stations")
    parser.add_argument("--what", required=True,
                        choices=["all", "actuals", "gfs_forecast", "ecmwf_forecast", "reanalysis"],
                        help="What to download")
    parser.add_argument("--station", default=None,
                        help="Download only this station (default: all 28)")
    args = parser.parse_args()

    if args.station:
        stations = {args.station: STATIONS_RESEARCH[args.station]}
    else:
        stations = STATIONS_RESEARCH

    logger.info(f"Downloading {args.what} for {len(stations)} stations")

    for station_id, info in stations.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"  {station_id}: {info['city']}")
        logger.info(f"{'='*60}")

        if args.what in ("all", "actuals"):
            download_actuals(station_id)

        if args.what != "actuals":
            download_station_forecasts(station_id, info, args.what)

    logger.info("\nAll downloads complete.")


if __name__ == "__main__":
    main()
