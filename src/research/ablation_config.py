"""
Feature group definitions for ablation studies.

Each key is an ablation condition: we DROP that group's columns and train/eval.
Columns not present in the dataframe are ignored.
"""

# Columns to drop per ablation group (must match names after engineer_features)
FEATURE_GROUPS = {
    "ECMWF": [
        "EC_MaxT", "EC_MinT", "EC_MeanT", "Forecast_Uncertainty",
    ],
    "Bias": [
        "Model_Bias_MaxT", "Model_Bias_MaxT_Rolling",
        "Model_Bias_MinT", "Model_Bias_MinT_Rolling",
    ],
    "Lags": [
        "MaxT_lag_1", "MaxT_lag_2", "MaxT_lag_3",
        "MinT_lag_1", "MinT_lag_2", "MinT_lag_3",
    ],
    "Rolling": [
        "MaxT_Rolling_Mean_3d", "MaxT_Rolling_Mean_7d", "MaxT_Rolling_Std_7d",
        "MaxT_Momentum", "MaxT_Acceleration", "MaxT_vs_30d_Mean",
        "MinT_Rolling_Mean_3d", "MinT_Rolling_Mean_7d", "MinT_Rolling_Std_7d",
        "MinT_Momentum", "MinT_Acceleration", "MinT_vs_30d_Mean",
    ],
    "NWP_atmosphere": [
        "Forecast_Wind", "Forecast_Dir", "Forecast_DewPoint",
        "Forecast_Solar", "Forecast_Clouds",
    ],
    "Physics": [
        "U_Wind", "V_Wind", "Chinook_Index", "DewPoint_Depression",
        "Solar_Load", "Clear_Sky_Index", "Cooling_Potential",
    ],
    "Time": [
        "sin_time", "cos_time", "day_of_week", "is_weekend",
    ],
    # Optional sanity ablation: core NWP temps; removing them causes very large ΔMAE.
    # Run with: python scripts/research/run_ablation.py --groups NWP_primary
    "NWP_primary": [
        "Forecast_MaxT", "Forecast_MinT", "GFS_MaxT", "GFS_MinT", "GFS_MeanT", "Forecast_MeanT",
    ],
}

# Default groups for a full ablation run (excludes NWP_primary so results stay comparable).
ABLATION_GROUP_IDS = [g for g in FEATURE_GROUPS.keys() if g != "NWP_primary"]


def columns_to_drop_for_ablation(ablation_group: str, available_columns: list) -> list:
    """
    Return the list of column names to drop for this ablation, restricted to
    columns that exist in the dataframe.
    """
    to_drop = FEATURE_GROUPS.get(ablation_group, [])
    return [c for c in to_drop if c in available_columns]
