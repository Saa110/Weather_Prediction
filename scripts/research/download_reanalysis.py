
import sys, time, requests
sys.path.insert(0, '.')
from pathlib import Path
from src.research.stations import STATIONS_RESEARCH
import pandas as pd
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RE_DIR = Path("data/research/reanalysis")

# Test one request first
print("Testing if rate limit has reset...")
test = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
    "latitude": 32.9, "longitude": -97.04,
    "start_date": "2026-01-01", "end_date": "2026-01-02",
    "hourly": "temperature_2m", "timezone": "America/Chicago"
}, timeout=30)
print(f"  Status: {test.status_code}")
if test.status_code == 429:
    print("  Rate limit still active. Try again later.")
    sys.exit(1)
print("  Rate limit reset! Proceeding with downloads...")

def download_chunk(station_id, lat, lon, tz, start_date, end_date, output_dir):
    """Download one chunk with retry and longer sleep."""
    outfile = output_dir / f"{station_id}_REANALYSIS_{start_date}_{end_date}.csv"
    if outfile.exists():
        logger.info(f"  {station_id} {start_date}-{end_date} already exists")
        return True
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_dfs = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=365), end)
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": current.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "hourly": ["temperature_2m","windspeed_10m","winddirection_10m",
                       "shortwave_radiation","cloudcover","dewpoint_2m"],
            "timezone": tz,
        }
        
        for attempt in range(3):
            try:
                resp = requests.get(url, params=params, timeout=300)
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    logger.warning(f"  429 rate limit, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                if "hourly" not in data:
                    logger.warning(f"  No data for {current.date()}-{chunk_end.date()}")
                    break
                
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
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(10)
                else:
                    logger.error(f"    FAILED: {e}")
        
        time.sleep(3)  # 3s between sub-chunks
        current = chunk_end + timedelta(days=1)
    
    if all_dfs:
        full = pd.concat(all_dfs).sort_index()
        full = full[~full.index.duplicated(keep="last")]
        full.to_csv(outfile)
        logger.info(f"  {station_id} REANALYSIS: {len(full)} days -> {outfile}")
        return True
    return False

# Download missing stations one at a time with generous delays
missing_stations = ["KDFW", "KDAL", "KAUS", "KSAT"]
periods = [
    ("2018-01-01", "2021-12-31"),
    ("2022-01-01", "2024-12-31"),
    ("2025-01-01", "2026-02-12"),
]

for station_id in missing_stations:
    info = STATIONS_RESEARCH[station_id]
    lat, lon, tz = info["lat"], info["lon"], info["timezone"]
    logger.info(f"\n{'='*40}")
    logger.info(f"  {station_id}: {info['city']}")
    
    for start, end in periods:
        download_chunk(station_id, lat, lon, tz, start, end, RE_DIR)
        time.sleep(5)  # 5s between chunks
    
    time.sleep(10)  # 10s between stations

# Final count
count = len(list(RE_DIR.glob("*_REANALYSIS_*.csv")))
print(f"\nFinal reanalysis files: {count} / 84 expected")