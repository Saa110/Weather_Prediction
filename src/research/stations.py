"""
All 28 Kalshi-tradeable temperature stations.

Grouped by climate type for analysis. Every station settles on the
NWS Daily Climate Report (CLI). Station IDs are ICAO codes matching
ACIS and Open-Meteo coordinate lookups.
"""

CLIMATE_TYPES = [
    "Continental",
    "NE Coastal",
    "SE Subtropical",
    "Gulf/SC",
    "Pacific",
    "Arid",
]

STATIONS_RESEARCH = {
    # ---- Continental (6) ----
    "KORD": {
        "city": "Chicago O'Hare, IL",
        "lat": 41.960,
        "lon": -87.932,
        "timezone": "America/Chicago",
        "climate": "Continental",
    },
    "KMDW": {
        "city": "Chicago Midway, IL",
        "lat": 41.784,
        "lon": -87.755,
        "timezone": "America/Chicago",
        "climate": "Continental",
    },
    "KMSP": {
        "city": "Minneapolis, MN",
        "lat": 44.885,
        "lon": -93.222,
        "timezone": "America/Chicago",
        "climate": "Continental",
    },
    "KDTW": {
        "city": "Detroit, MI",
        "lat": 42.212,
        "lon": -83.353,
        "timezone": "America/Detroit",
        "climate": "Continental",
    },
    "KDEN": {
        "city": "Denver, CO",
        "lat": 39.856,
        "lon": -104.674,
        "timezone": "America/Denver",
        "climate": "Continental",
    },
    "KOKC": {
        "city": "Oklahoma City, OK",
        "lat": 35.393,
        "lon": -97.601,
        "timezone": "America/Chicago",
        "climate": "Continental",
    },

    # ---- Northeast Coastal (5) ----
    "KNYC": {
        "city": "New York Central Park, NY",
        "lat": 40.779,
        "lon": -73.969,
        "timezone": "America/New_York",
        "climate": "NE Coastal",
        "note": "Co-op station, not ASOS. Verify ACIS data.",
    },
    "KLGA": {
        "city": "LaGuardia, NY",
        "lat": 40.777,
        "lon": -73.873,
        "timezone": "America/New_York",
        "climate": "NE Coastal",
    },
    "KBOS": {
        "city": "Boston, MA",
        "lat": 42.366,
        "lon": -71.010,
        "timezone": "America/New_York",
        "climate": "NE Coastal",
    },
    "KPHL": {
        "city": "Philadelphia, PA",
        "lat": 39.872,
        "lon": -75.241,
        "timezone": "America/New_York",
        "climate": "NE Coastal",
    },
    "KDCA": {
        "city": "Washington D.C.",
        "lat": 38.851,
        "lon": -77.040,
        "timezone": "America/New_York",
        "climate": "NE Coastal",
    },

    # ---- Southeast Subtropical (6) ----
    "KATL": {
        "city": "Atlanta, GA",
        "lat": 33.630,
        "lon": -84.422,
        "timezone": "America/New_York",
        "climate": "SE Subtropical",
    },
    "KCLT": {
        "city": "Charlotte, NC",
        "lat": 35.214,
        "lon": -80.947,
        "timezone": "America/New_York",
        "climate": "SE Subtropical",
    },
    "KBNA": {
        "city": "Nashville, TN",
        "lat": 36.124,
        "lon": -86.678,
        "timezone": "America/Chicago",
        "climate": "SE Subtropical",
    },
    "KJAX": {
        "city": "Jacksonville, FL",
        "lat": 30.494,
        "lon": -81.688,
        "timezone": "America/New_York",
        "climate": "SE Subtropical",
    },
    "KTPA": {
        "city": "Tampa, FL",
        "lat": 27.976,
        "lon": -82.533,
        "timezone": "America/New_York",
        "climate": "SE Subtropical",
    },
    "KMIA": {
        "city": "Miami, FL",
        "lat": 25.788,
        "lon": -80.317,
        "timezone": "America/New_York",
        "climate": "SE Subtropical",
    },

    # ---- Gulf / South Central (6) ----
    "KHOU": {
        "city": "Houston, TX",
        "lat": 29.645,
        "lon": -95.279,
        "timezone": "America/Chicago",
        "climate": "Gulf/SC",
    },
    "KMSY": {
        "city": "New Orleans, LA",
        "lat": 29.993,
        "lon": -90.258,
        "timezone": "America/Chicago",
        "climate": "Gulf/SC",
    },
    "KDFW": {
        "city": "Dallas DFW, TX",
        "lat": 32.900,
        "lon": -97.040,
        "timezone": "America/Chicago",
        "climate": "Gulf/SC",
    },
    "KDAL": {
        "city": "Dallas Love, TX",
        "lat": 32.847,
        "lon": -96.852,
        "timezone": "America/Chicago",
        "climate": "Gulf/SC",
    },
    "KAUS": {
        "city": "Austin, TX",
        "lat": 30.210,
        "lon": -97.681,
        "timezone": "America/Chicago",
        "climate": "Gulf/SC",
    },
    "KSAT": {
        "city": "San Antonio, TX",
        "lat": 29.534,
        "lon": -98.470,
        "timezone": "America/Chicago",
        "climate": "Gulf/SC",
    },

    # ---- Pacific Maritime (3) ----
    "KLAX": {
        "city": "Los Angeles, CA",
        "lat": 33.938,
        "lon": -118.387,
        "timezone": "America/Los_Angeles",
        "climate": "Pacific",
    },
    "KSFO": {
        "city": "San Francisco, CA",
        "lat": 37.621,
        "lon": -122.379,
        "timezone": "America/Los_Angeles",
        "climate": "Pacific",
    },
    "KSEA": {
        "city": "Seattle, WA",
        "lat": 47.450,
        "lon": -122.309,
        "timezone": "America/Los_Angeles",
        "climate": "Pacific",
    },

    # ---- Arid / Desert (2) ----
    "KPHX": {
        "city": "Phoenix, AZ",
        "lat": 33.437,
        "lon": -112.008,
        "timezone": "America/Phoenix",
        "climate": "Arid",
    },
    "KLAS": {
        "city": "Las Vegas, NV",
        "lat": 36.084,
        "lon": -115.154,
        "timezone": "America/Los_Angeles",
        "climate": "Arid",
    },
}


def get_stations_by_climate(climate_type: str) -> dict:
    """Return stations matching a climate type."""
    return {
        k: v for k, v in STATIONS_RESEARCH.items()
        if v["climate"] == climate_type
    }


def get_all_station_ids() -> list:
    """Return sorted list of all 28 station IDs."""
    return sorted(STATIONS_RESEARCH.keys())
