# preprocessor.py
import numpy as np
import pandas as pd


# ============================================================================
# ALIAS → CANONICAL NAME MAPPING
# These aliases exist for backward compatibility in CSVs and loaders,
# but should not be separate model input features (they duplicate canonical cols).
# ============================================================================
_ALIAS_TO_CANONICAL = {
    'Forecast_AirMass': 'Forecast_MaxT',   # In consolidated pipeline: same value
    'Forecast_MinTemp': 'Forecast_MinT',   # In all pipelines: same value
    'GFS_AirMass':      'GFS_MaxT',        # Always a copy
    'GFS_MinTemp':      'GFS_MinT',        # Always a copy
    'EC_AirMass':       'EC_MaxT',         # Always a copy
    'EC_MinTemp':       'EC_MinT',         # Always a copy
}

# Columns safe to drop after all derived features are computed
_REDUNDANT_ALIASES = [
    'GFS_AirMass', 'EC_AirMass',
    'GFS_MinTemp', 'EC_MinTemp',
    'Forecast_MinTemp',
    'Forecast_Surf_Max', 'Forecast_Surf_Min',  # Legacy names
]


def engineer_features(df, drop_redundant=True, mode=None, verbose=True):
    """
    Transforms raw data into features for model training and inference.

    Pipeline: Unit conversion → Canonicalize names → Snow/missing →
    Time cycles → Lags → Rolling/momentum → Bias correction →
    Wind vectors → Moisture → Thermodynamics → Drop aliases → Cleanup.

    Args:
        df: DataFrame with forecast columns (Forecast_AirMass or Forecast_MaxT,
            Forecast_Wind, etc.) and optionally actuals (MaxT, MinT).
        drop_redundant: If True (default), drop duplicate alias columns after
            all derived features are computed. Reduces feature count by ~6-10
            without losing information. Set False for backward compatibility
            with models trained on alias columns.
        mode: Feature mode for train/inference alignment:
            - None (default): Keep all columns (backward compatible).
            - 'production': Drop ECMWF (EC_*) columns for GFS-only deployment.

    Returns:
        DataFrame with engineered features. Targets (MaxT, MinT) are kept;
        they are dropped at train time by model.py.

    Critical Dependencies (rows are dropped if missing):
        - Forecast_MaxT (or Forecast_AirMass): Primary forecast temperature.
        - Forecast_Wind: Primary forecast wind speed.

    Cold Start Behavior:
        - Model_Bias_*_Rolling: Filled with 0 when history < 3 days.
        - Lag features (MaxT_lag_1, etc.): NaN for first 1-3 rows (CatBoost handles NaN).
        - Rolling features (*_Rolling_Mean_7d): Use min_periods for partial windows.
    """
    df = df.copy()

    # --- 1. UNIT CONVERSION (Celsius → Fahrenheit) ---
    # Covers both alias and canonical names in case both exist.
    temp_cols = [
        'Forecast_MaxT', 'Forecast_MinT',
        'Forecast_AirMass', 'Forecast_MinTemp',
        'Forecast_Surf_Max', 'Forecast_Surf_Min',
        'Forecast_DewPoint',
        'GFS_MaxT', 'GFS_MinT', 'GFS_AirMass', 'GFS_MinTemp',
        'EC_MaxT', 'EC_MinT', 'EC_AirMass', 'EC_MinTemp',
    ]

    for col in temp_cols:
        if col in df.columns:
            # Heuristic: if mean < 50, values are likely Celsius.
            if df[col].mean() < 50:
                df[col] = (df[col] * 9/5) + 32

    # --- 2. CANONICALIZE COLUMN NAMES ---
    # Ensure canonical names exist. If the canonical column is missing but
    # the alias exists, create it. Both survive until section 10 (cleanup).
    for alias, canonical in _ALIAS_TO_CANONICAL.items():
        if canonical not in df.columns and alias in df.columns:
            df[canonical] = df[alias]

    # --- 3. MISSING VALUES & SNOW LOGIC ---
    if 'SnowDepth' in df.columns:
        df['SnowDepth'] = df['SnowDepth'].fillna(0)
        df['Has_Snow'] = (df['SnowDepth'] > 0.001).astype(int)

    if 'Pcpn' in df.columns:
        df['Pcpn'] = df['Pcpn'].fillna(0)

    # --- 4. TIME CYCLES ---
    df['day_of_year'] = df.index.dayofyear
    df['sin_time'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['cos_time'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)

    # --- 5. TARGET LAGS ---
    # NOTE: These use PAST actuals only. At inference time, lag_1 is yesterday's
    # actual (known), lag_2 is two days ago, etc. No future data leakage.
    lags = [1, 2, 3]
    for col in ['MaxT', 'MinT']:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # --- 5B. ENHANCED TEMPORAL FEATURES ---
    # Built from lag_1 (shifted), so these also use only past data.
    for col in ['MaxT', 'MinT']:
        if f'{col}_lag_1' in df.columns:
            # Rolling mean (smoothed recent history)
            df[f'{col}_Rolling_Mean_3d'] = df[f'{col}_lag_1'].rolling(window=3, min_periods=1).mean()
            df[f'{col}_Rolling_Mean_7d'] = df[f'{col}_lag_1'].rolling(window=7, min_periods=1).mean()

            # Rolling std (recent variability)
            df[f'{col}_Rolling_Std_7d'] = df[f'{col}_lag_1'].rolling(window=7, min_periods=1).std()

            # Momentum (short-term trend: positive = warming, negative = cooling)
            if f'{col}_lag_3' in df.columns:
                df[f'{col}_Momentum'] = df[f'{col}_lag_1'] - df[f'{col}_lag_3']

            # Acceleration (change in trend — second difference)
            if f'{col}_lag_2' in df.columns and f'{col}_lag_3' in df.columns:
                recent_change = df[f'{col}_lag_1'] - df[f'{col}_lag_2']
                previous_change = df[f'{col}_lag_2'] - df[f'{col}_lag_3']
                df[f'{col}_Acceleration'] = recent_change - previous_change

            # Deviation from recent mean (anomaly indicator)
            df[f'{col}_vs_30d_Mean'] = df[f'{col}_lag_1'] - df[f'{col}_lag_1'].rolling(window=30, min_periods=7).mean()

    # Day of week encoding
    df['day_of_week'] = df.index.dayofweek
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # --- 6. BIAS CORRECTION FEATURES ---
    # Computes: Model_Bias_MaxT = actual_yesterday − forecast_yesterday
    # This measures recent systematic error and helps the model self-correct.
    #
    # COLD START: Model_Bias_*_Rolling is filled with 0 (neutral bias) when
    # insufficient history exists. This is acceptable because it simply means
    # "no known bias" for the first few days of a new deployment.
    #
    # Uses alias name (Forecast_AirMass) if available, otherwise canonical
    # (Forecast_MaxT). The intermediate lag column is dropped at the end.
    _bias_maxT_col = 'Forecast_AirMass' if 'Forecast_AirMass' in df.columns else (
        'Forecast_MaxT' if 'Forecast_MaxT' in df.columns else None
    )
    if _bias_maxT_col and 'MaxT_lag_1' in df.columns:
        df['_bias_maxT_lag'] = df[_bias_maxT_col].shift(1)
        df['Model_Bias_MaxT'] = df['MaxT_lag_1'] - df['_bias_maxT_lag']
        df['Model_Bias_MaxT_Rolling'] = df['Model_Bias_MaxT'].rolling(window=3).mean()
        df['Model_Bias_MaxT_Rolling'] = df['Model_Bias_MaxT_Rolling'].fillna(0)
        df.drop(columns=['_bias_maxT_lag'], inplace=True)

    _bias_minT_col = 'Forecast_MinTemp' if 'Forecast_MinTemp' in df.columns else (
        'Forecast_MinT' if 'Forecast_MinT' in df.columns else None
    )
    if _bias_minT_col and 'MinT_lag_1' in df.columns:
        df['_bias_minT_lag'] = df[_bias_minT_col].shift(1)
        df['Model_Bias_MinT'] = df['MinT_lag_1'] - df['_bias_minT_lag']
        df['Model_Bias_MinT_Rolling'] = df['Model_Bias_MinT'].rolling(window=3).mean()
        df['Model_Bias_MinT_Rolling'] = df['Model_Bias_MinT_Rolling'].fillna(0)
        df.drop(columns=['_bias_minT_lag'], inplace=True)

    # --- 7. PHYSICS VECTORS ---
    if 'Forecast_Dir' in df.columns and 'Forecast_Wind' in df.columns:
        wind_rad = np.deg2rad(df['Forecast_Dir'])
        df['U_Wind'] = -1 * df['Forecast_Wind'] * np.sin(wind_rad)
        df['V_Wind'] = -1 * df['Forecast_Wind'] * np.cos(wind_rad)

        # Chinook Index: westerly wind × forecast temperature
        # Larger positive = warm downslope winds (Denver-specific but general signal)
        _temp = 'Forecast_AirMass' if 'Forecast_AirMass' in df.columns else 'Forecast_MaxT'
        if _temp in df.columns:
            df['Chinook_Index'] = df['U_Wind'] * df[_temp]

    # --- 8. MOISTURE ---
    _temp = 'Forecast_AirMass' if 'Forecast_AirMass' in df.columns else 'Forecast_MaxT'
    if 'Forecast_DewPoint' in df.columns and _temp in df.columns:
        df['DewPoint_Depression'] = df[_temp] - df['Forecast_DewPoint']

    # --- 9. THERMODYNAMICS ---
    if 'Forecast_Solar' in df.columns:
        df['Solar_Load'] = df['Forecast_Solar'] / 1000000.0

    if 'Forecast_Clouds' in df.columns:
        df['Clear_Sky_Index'] = (100 - df['Forecast_Clouds']) / 100.0

        # FIXED (was: Clear_Sky_Index * (100 - Forecast_AirMass) which is dimensionally
        # wrong — 100 has no meaning on the °F scale and the product could go negative).
        #
        # New formula: Clear_Sky_Index × DewPoint_Depression
        # Physical meaning: clear skies + dry air = high radiative cooling potential.
        # Both terms are dimensionally consistent and physically meaningful.
        if 'DewPoint_Depression' in df.columns:
            df['Cooling_Potential'] = df['Clear_Sky_Index'] * df['DewPoint_Depression']

    # --- 10. DROP REDUNDANT ALIAS COLUMNS ---
    # After all derived features are computed, remove alias columns that
    # duplicate canonical columns. This reduces noise and makes feature
    # importance easier to interpret.
    if drop_redundant:
        cols_to_drop = [c for c in _REDUNDANT_ALIASES if c in df.columns]

        # Drop Forecast_AirMass if the canonical Forecast_MaxT also exists
        if 'Forecast_MaxT' in df.columns and 'Forecast_AirMass' in df.columns:
            cols_to_drop.append('Forecast_AirMass')

        # Drop any lingering legacy intermediate columns
        for legacy in ['Forecast_AirMass_lag_1', 'Forecast_MinTemp_lag_1']:
            if legacy in df.columns:
                cols_to_drop.append(legacy)

        # Deduplicate the list
        cols_to_drop = sorted(set(cols_to_drop))

        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            if verbose:
                print(f"[PREPROCESSING] Dropped {len(cols_to_drop)} redundant columns: "
                      f"{', '.join(cols_to_drop)}")

    # --- 11. PRODUCTION MODE: DROP ECMWF COLUMNS ---
    # In GFS-only production, ECMWF columns are always NaN (no ECMWF data).
    # Dropping them avoids training on columns that won't exist at inference.
    if mode == 'production':
        ec_cols = [c for c in df.columns if c.startswith('EC_')]
        if ec_cols:
            df.drop(columns=ec_cols, inplace=True)
            if verbose:
                print(f"[PREPROCESSING] Production mode: dropped {len(ec_cols)} ECMWF columns")

    # --- 12. CLEANUP ---
    # Drop rows missing CRITICAL features (forecast temp + wind).
    # These are required for meaningful predictions. All other features
    # are optional (NaN is handled by CatBoost or fillna above).
    critical = []
    if 'Forecast_MaxT' in df.columns:
        critical.append('Forecast_MaxT')
    elif 'Forecast_AirMass' in df.columns:
        critical.append('Forecast_AirMass')  # Fallback if aliases weren't dropped
    if 'Forecast_Wind' in df.columns:
        critical.append('Forecast_Wind')

    if critical:
        rows_before = len(df)
        df.dropna(subset=critical, inplace=True)
        rows_after = len(df)
        if rows_before - rows_after > 0 and verbose:
            print(f"[PREPROCESSING] Dropped {rows_before - rows_after} rows "
                  f"missing critical features: {critical}")

    return df
