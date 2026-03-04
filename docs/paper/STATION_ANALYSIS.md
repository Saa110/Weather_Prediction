# Comprehensive Per-Station Analysis

*Auto-generated on 2026-02-18 17:17*

## Executive Summary

This document presents a detailed, station-by-station analysis of MOS (Model Output Statistics) post-processing performance across all **28 Kalshi-tradeable U.S. temperature stations**, spanning **6 climate zones**. The analysis draws from **2,016 experiments** covering 5 ML architectures (Linear, MLP, XGBoost, LightGBM, CatBoost), 2 training data sources (Historical Forecasts vs. ERA5 Reanalysis), and a fixed test period of July 2025 -- February 2026 (~227 days).

### Key Findings

- **Overall MaxT MAE**: 0.868 F (range: 0.555 -- 1.445 F). Best architecture on average: **MLP**.
- **Overall MinT MAE**: 1.053 F (range: 0.705 -- 1.665 F). Best architecture on average: **Linear**.
- **GFS improvement**: Post-processing reduces MaxT error by **28.1%** and MinT error by **30.9%** on average vs. raw GFS.
- **Most critical feature group**: **NWP_primary** (avg +0.365 F when removed).
- **Seasonal pattern**: Winter is the hardest season (MAE 1.122 F), Summer the easiest (0.904 F).

### Document Structure

Stations are grouped by climate zone. For each station we present:

1. **Model Performance** -- all 5 architectures on both Historical Forecast and ERA5 datasets (MAE, RMSE, Bias)
2. **GFS Deviation** -- raw NWP baselines vs. best MOS model
3. **Feature Ablation** -- impact of removing each feature group
4. **Seasonal Performance** -- Summer / Fall / Winter MAE breakdown
5. **Station Insights** -- auto-generated commentary on notable patterns

---

## 1. Continental (6 stations)

Stations: KORD, KMDW, KMSP, KDTW, KDEN, KOKC

### 1.1 KORD -- Chicago O'Hare, IL

**Climate zone**: Continental | **Coordinates**: 41.96 N, 87.93 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.920 | 1.124 | 0.58 | 1.029 | 1.292 | 0.25 |
| **MLP** | **0.877** | 1.072 | 0.52 | 1.000 | 1.262 | 0.09 |
| XGBoost | 0.943 | 1.179 | 0.39 | 1.206 | 1.511 | 0.30 |
| LightGBM | 1.393 | 2.308 | 0.54 | 1.308 | 1.835 | 0.35 |
| CatBoost | 1.259 | 1.618 | 0.24 | 1.381 | 1.810 | 0.30 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.067 | 1.432 | 0.53 | 1.353 | 1.805 | 0.18 |
| **MLP** | **1.059** | 1.430 | 0.53 | 1.369 | 1.840 | 0.21 |
| XGBoost | 1.303 | 1.784 | 0.27 | 1.549 | 2.066 | 0.02 |
| LightGBM | 1.533 | 2.340 | 0.48 | 1.572 | 2.259 | 0.18 |
| CatBoost | 1.364 | 1.983 | 0.24 | 1.654 | 2.268 | -0.01 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.819 | 1.849 | 1.116 | 0.877 | -0.07 | -7.1% |
| MinT | 1.111 | 1.616 | 1.176 | 1.059 | 0.05 | 4.7% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.090 | +0.101 | +0.119 | +0.053 | +0.033 |
| ECMWF | +0.000 | -0.002 | +0.090 | +0.005 | +0.042 |
| Lags | -0.002 | -0.004 | +0.038 | -0.038 | +0.028 |
| Rolling | -0.011 | +0.011 | -0.021 | -0.033 | -0.087 |
| NWP_atmosphere | -0.021 | -0.004 | -0.011 | -0.079 | +0.005 |
| Physics | -0.041 | +0.005 | -0.011 | -0.031 | +0.047 |
| Time | -0.002 | -0.002 | +0.003 | -0.072 | +0.033 |
| NWP_primary | -0.024 | -0.044 | +0.597 | +0.405 | +0.408 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.075 | +0.089 | +0.068 | +0.051 | +0.070 |
| ECMWF | -0.001 | +0.000 | -0.023 | -0.017 | +0.099 |
| Lags | +0.004 | +0.000 | -0.023 | -0.002 | +0.056 |
| Rolling | +0.005 | -0.003 | -0.035 | +0.005 | +0.023 |
| NWP_atmosphere | -0.022 | -0.020 | -0.014 | -0.029 | +0.014 |
| Physics | -0.002 | -0.005 | -0.021 | -0.023 | +0.074 |
| Time | +0.004 | +0.001 | -0.018 | -0.010 | +0.055 |
| NWP_primary | +0.232 | +0.232 | +0.313 | +0.414 | +0.454 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 1.059 | 0.757 | 0.973 | Fall | Summer |
| MinT | 0.872 | 1.230 | 1.134 | Summer | Fall |

#### Station Insights

- MaxT shows minimal improvement (-7.1%) over raw GFS -- GFS already performs well here.
- MinT shows minimal improvement (4.7%) over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.268 F avg across architectures).
- **MaxT ablation**: Removing NWP_atmosphere, Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.329 F avg across architectures).
- **MaxT seasonality**: Summer MAE (1.059 F) is 40% worse than Fall (0.757 F).
- **MinT seasonality**: Fall MAE (1.230 F) is 41% worse than Summer (0.872 F).

---

### 1.2 KMDW -- Chicago Midway, IL

**Climate zone**: Continental | **Coordinates**: 41.78 N, 87.75 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.813** | 1.015 | 0.02 | 1.032 | 1.324 | 0.29 |
| MLP | 0.825 | 1.030 | 0.01 | 1.009 | 1.298 | 0.25 |
| XGBoost | 0.943 | 1.213 | -0.11 | 1.196 | 1.579 | 0.33 |
| LightGBM | 1.274 | 2.167 | 0.07 | 1.277 | 1.799 | 0.52 |
| CatBoost | 1.181 | 1.572 | -0.04 | 1.307 | 1.722 | 0.26 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.057 | 1.371 | 0.12 | 1.233 | 1.639 | 0.03 |
| **MLP** | **1.055** | 1.367 | 0.10 | 1.265 | 1.659 | 0.03 |
| XGBoost | 1.296 | 1.728 | -0.08 | 1.449 | 1.955 | -0.22 |
| LightGBM | 1.536 | 2.374 | 0.02 | 1.466 | 2.131 | -0.11 |
| CatBoost | 1.352 | 1.927 | 0.02 | 1.464 | 2.073 | -0.15 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.462 | 2.026 | 1.675 | 0.813 | 0.44 | 44.4% |
| MinT | 1.660 | 2.349 | 1.928 | 1.055 | 0.36 | 36.5% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.064 | +0.051 | -0.009 | +0.050 | -0.037 |
| ECMWF | -0.002 | +0.000 | +0.035 | -0.005 | -0.071 |
| Lags | +0.000 | +0.002 | +0.007 | +0.088 | -0.074 |
| Rolling | -0.002 | +0.005 | +0.012 | +0.059 | -0.101 |
| NWP_atmosphere | +0.010 | -0.014 | -0.009 | +0.027 | -0.048 |
| Physics | +0.010 | +0.003 | -0.038 | +0.098 | -0.076 |
| Time | -0.000 | -0.003 | +0.034 | -0.012 | -0.059 |
| NWP_primary | -0.006 | -0.011 | +0.412 | +0.343 | +0.283 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.010 | +0.034 | +0.118 | +0.057 | +0.144 |
| ECMWF | +0.003 | +0.000 | +0.032 | -0.036 | +0.106 |
| Lags | +0.005 | +0.003 | +0.032 | +0.019 | -0.014 |
| Rolling | +0.006 | -0.009 | +0.052 | +0.034 | -0.032 |
| NWP_atmosphere | -0.006 | -0.000 | -0.007 | +0.013 | +0.017 |
| Physics | -0.034 | -0.022 | +0.000 | +0.008 | +0.010 |
| Time | +0.006 | +0.000 | -0.018 | +0.025 | +0.095 |
| NWP_primary | +0.403 | +0.336 | +0.408 | +0.405 | +0.296 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.887 | 0.817 | 0.645 | Winter | Summer |
| MinT | 0.918 | 1.226 | 0.983 | Summer | Fall |

#### Station Insights

- **MaxT**: Best MAE (0.813 F, Linear) is 10.1% below the Continental group mean (0.904 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.204 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.369 F avg across architectures).
- **MaxT seasonality**: Summer MAE (0.887 F) is 38% worse than Winter (0.645 F).
- **MinT seasonality**: Fall MAE (1.226 F) is 34% worse than Summer (0.918 F).

---

### 1.3 KMSP -- Minneapolis, MN

**Climate zone**: Continental | **Coordinates**: 44.88 N, 93.22 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.775 | 0.974 | 0.17 | 0.862 | 1.155 | -0.13 |
| **MLP** | **0.761** | 0.952 | 0.14 | 0.857 | 1.152 | -0.14 |
| XGBoost | 1.032 | 1.425 | 0.04 | 1.121 | 1.607 | -0.04 |
| LightGBM | 1.200 | 1.931 | 0.20 | 1.152 | 1.745 | 0.02 |
| CatBoost | 1.283 | 1.820 | -0.13 | 1.332 | 1.858 | -0.06 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.969 | 1.198 | -0.03 | 1.166 | 1.573 | -0.27 |
| **MLP** | **0.969** | 1.211 | -0.00 | 1.173 | 1.574 | -0.26 |
| XGBoost | 1.225 | 1.830 | -0.01 | 1.569 | 2.146 | -0.39 |
| LightGBM | 1.341 | 2.145 | 0.08 | 1.625 | 2.344 | -0.33 |
| CatBoost | 1.512 | 2.274 | 0.04 | 1.591 | 2.208 | -0.29 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.972 | 2.109 | 1.362 | 0.761 | 0.22 | 21.8% |
| MinT | 1.275 | 1.815 | 1.392 | 0.969 | 0.24 | 24.0% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.002 | +0.010 | +0.008 | +0.001 | +0.019 |
| ECMWF | +0.001 | -0.001 | +0.020 | -0.006 | -0.019 |
| Lags | +0.000 | -0.000 | -0.031 | -0.029 | +0.041 |
| Rolling | -0.005 | -0.009 | +0.005 | -0.036 | +0.106 |
| NWP_atmosphere | -0.017 | +0.004 | -0.015 | -0.026 | +0.152 |
| Physics | -0.004 | +0.001 | -0.000 | -0.020 | +0.000 |
| Time | -0.012 | +0.006 | +0.010 | +0.017 | +0.055 |
| NWP_primary | -0.010 | +0.134 | +0.712 | +0.616 | +0.503 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.001 | -0.000 | +0.050 | -0.041 | -0.056 |
| ECMWF | +0.000 | +0.000 | +0.041 | -0.024 | -0.064 |
| Lags | +0.001 | +0.000 | +0.053 | -0.009 | -0.047 |
| Rolling | +0.003 | +0.007 | +0.052 | -0.004 | -0.111 |
| NWP_atmosphere | +0.024 | +0.034 | +0.021 | +0.008 | -0.042 |
| Physics | +0.001 | +0.011 | +0.066 | -0.010 | +0.072 |
| Time | -0.005 | -0.008 | +0.052 | -0.036 | -0.048 |
| NWP_primary | +0.349 | +0.289 | +0.619 | +0.461 | +0.408 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.900 | 0.713 | 0.636 | Winter | Summer |
| MinT | 0.917 | 1.067 | 0.869 | Winter | Fall |

#### Station Insights

- **MaxT**: Best MAE (0.761 F, MLP) is 15.8% below the Continental group mean (0.904 F).
- **MinT**: Best MAE (0.969 F, MLP) is 15.5% below the Continental group mean (1.147 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.391 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.425 F avg across architectures).
- **MaxT seasonality**: Summer MAE (0.900 F) is 42% worse than Winter (0.636 F).

---

### 1.4 KDTW -- Detroit, MI

**Climate zone**: Continental | **Coordinates**: 42.21 N, 83.35 W | **Timezone**: America/Detroit

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.837 | 1.094 | 0.45 | 1.126 | 1.417 | 0.19 |
| **MLP** | **0.814** | 1.073 | 0.36 | 1.123 | 1.414 | 0.06 |
| XGBoost | 0.934 | 1.305 | 0.33 | 1.210 | 1.633 | -0.01 |
| LightGBM | 1.385 | 2.427 | 0.59 | 1.314 | 1.851 | 0.06 |
| CatBoost | 1.169 | 1.714 | 0.16 | 1.343 | 1.835 | 0.02 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.910** | 1.205 | -0.04 | 1.286 | 1.667 | -0.27 |
| MLP | 0.913 | 1.211 | 0.04 | 1.310 | 1.686 | -0.24 |
| XGBoost | 1.110 | 1.573 | 0.00 | 1.523 | 2.082 | -0.38 |
| LightGBM | 1.623 | 2.826 | 0.24 | 1.595 | 2.328 | -0.26 |
| CatBoost | 1.465 | 2.153 | 0.20 | 1.606 | 2.213 | -0.25 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.817 | 1.532 | 1.015 | 0.814 | 0.00 | 0.4% |
| MinT | 1.034 | 1.517 | 1.077 | 0.910 | 0.12 | 12.0% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.006 | +0.025 | +0.014 | +0.067 | -0.068 |
| ECMWF | -0.000 | +0.002 | +0.006 | +0.066 | -0.046 |
| Lags | +0.000 | +0.000 | -0.014 | +0.074 | -0.023 |
| Rolling | -0.013 | -0.005 | -0.035 | +0.006 | +0.027 |
| NWP_atmosphere | -0.007 | -0.002 | -0.000 | +0.054 | -0.044 |
| Physics | +0.010 | -0.003 | -0.040 | +0.065 | -0.120 |
| Time | -0.010 | -0.001 | -0.014 | +0.018 | -0.036 |
| NWP_primary | -0.022 | +0.067 | +0.619 | +0.635 | +0.421 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.040 | +0.012 | +0.028 | +0.035 | -0.049 |
| ECMWF | -0.000 | -0.000 | +0.090 | -0.015 | +0.016 |
| Lags | +0.005 | +0.000 | +0.012 | -0.016 | -0.067 |
| Rolling | +0.016 | -0.002 | +0.000 | +0.051 | -0.061 |
| NWP_atmosphere | -0.000 | +0.010 | +0.019 | +0.020 | -0.032 |
| Physics | +0.005 | -0.005 | -0.004 | +0.029 | -0.013 |
| Time | +0.009 | -0.002 | +0.060 | +0.064 | +0.011 |
| NWP_primary | +0.458 | +0.457 | +0.622 | +0.481 | +0.321 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.866 | 0.850 | 0.746 | Winter | Summer |
| MinT | 0.797 | 0.894 | 1.191 | Summer | Winter |

#### Station Insights

- **MaxT**: Best MAE (0.814 F, MLP) is 10.0% below the Continental group mean (0.904 F).
- **MinT**: Best MAE (0.910 F, Linear) is 20.7% below the Continental group mean (1.147 F).
- MaxT shows minimal improvement (0.4%) over raw GFS -- GFS already performs well here.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.344 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.468 F avg across architectures).
- **MinT seasonality**: Winter MAE (1.191 F) is 49% worse than Summer (0.797 F).

---

### 1.5 KDEN -- Denver, CO

**Climate zone**: Continental | **Coordinates**: 39.86 N, 104.67 W | **Timezone**: America/Denver

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.542 | 2.803 | 1.37 | 1.601 | 2.023 | 0.80 |
| **MLP** | **1.445** | 2.705 | 1.23 | 1.558 | 1.985 | 0.73 |
| XGBoost | 1.614 | 2.903 | 1.25 | 1.683 | 2.132 | 0.65 |
| LightGBM | 1.739 | 3.019 | 1.42 | 1.732 | 2.215 | 0.81 |
| CatBoost | 1.780 | 3.035 | 1.35 | 1.778 | 2.285 | 0.79 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.728 | 2.261 | 0.89 | 2.220 | 2.804 | 0.66 |
| **MLP** | **1.665** | 2.160 | 0.80 | 2.213 | 2.760 | 0.62 |
| XGBoost | 1.864 | 2.409 | 0.93 | 2.348 | 2.922 | 0.55 |
| LightGBM | 1.937 | 2.462 | 0.92 | 2.336 | 2.900 | 0.61 |
| CatBoost | 1.892 | 2.423 | 0.79 | 2.433 | 3.040 | 0.50 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.281 | 1.961 | 1.513 | 1.445 | -0.13 | -12.8% |
| MinT | 2.312 | 4.790 | 3.478 | 1.665 | 0.28 | 28.0% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.154 | +0.151 | +0.089 | +0.036 | +0.140 |
| ECMWF | -0.002 | -0.009 | -0.021 | -0.038 | -0.025 |
| Lags | -0.006 | -0.000 | +0.063 | -0.030 | +0.040 |
| Rolling | -0.003 | +0.016 | +0.037 | -0.012 | +0.027 |
| NWP_atmosphere | -0.067 | -0.041 | +0.074 | -0.012 | +0.062 |
| Physics | -0.055 | -0.015 | +0.026 | -0.015 | +0.001 |
| Time | -0.010 | -0.005 | +0.010 | -0.022 | +0.029 |
| NWP_primary | +0.037 | +0.084 | +0.812 | +0.867 | +0.624 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.052 | +0.085 | +0.091 | -0.020 | +0.120 |
| ECMWF | -0.027 | +0.002 | +0.011 | -0.006 | +0.076 |
| Lags | -0.010 | +0.004 | -0.024 | -0.002 | +0.039 |
| Rolling | -0.018 | +0.000 | -0.002 | +0.029 | -0.002 |
| NWP_atmosphere | +0.012 | +0.002 | +0.020 | -0.025 | +0.078 |
| Physics | +0.025 | +0.005 | +0.060 | -0.047 | +0.097 |
| Time | +0.002 | +0.009 | +0.015 | -0.041 | +0.006 |
| NWP_primary | +1.169 | +1.271 | +1.134 | +1.071 | +1.073 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 1.501 | 1.167 | 2.471 | Fall | Winter |
| MinT | 1.374 | 1.926 | 2.032 | Summer | Winter |

#### Station Insights

- **MaxT**: Best MAE (1.445 F, MLP) is 59.8% above the Continental group mean (0.904 F).
- **MinT**: Best MAE (1.665 F, MLP) is 45.2% above the Continental group mean (1.147 F).
- MaxT shows minimal improvement (-12.8%) over raw GFS -- GFS already performs well here.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.485 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+1.144 F avg across architectures).
- **MaxT seasonality**: Winter MAE (2.471 F) is 112% worse than Fall (1.167 F).
- **MinT seasonality**: Winter MAE (2.032 F) is 48% worse than Summer (1.374 F).

---

### 1.6 KOKC -- Oklahoma City, OK

**Climate zone**: Continental | **Coordinates**: 35.39 N, 97.60 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.716** | 0.935 | -0.06 | 1.097 | 1.438 | -0.50 |
| MLP | 0.727 | 0.952 | -0.08 | 1.101 | 1.460 | -0.52 |
| XGBoost | 0.941 | 1.517 | -0.13 | 1.411 | 2.029 | -0.51 |
| LightGBM | 1.092 | 2.295 | -0.14 | 1.438 | 2.300 | -0.37 |
| CatBoost | 1.148 | 1.824 | -0.27 | 1.415 | 1.951 | -0.62 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.225** | 1.567 | 0.09 | 1.279 | 1.626 | -0.21 |
| MLP | 1.235 | 1.572 | 0.09 | 1.289 | 1.616 | -0.20 |
| XGBoost | 1.458 | 1.846 | 0.35 | 1.411 | 1.757 | -0.31 |
| LightGBM | 1.570 | 2.189 | 0.31 | 1.445 | 1.847 | -0.20 |
| CatBoost | 1.588 | 2.032 | 0.16 | 1.533 | 1.902 | -0.25 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.914 | 2.126 | 1.340 | 0.716 | 0.22 | 21.7% |
| MinT | 2.202 | 3.813 | 2.952 | 1.225 | 0.44 | 44.4% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.001 | +0.007 | +0.019 | +0.026 | +0.014 |
| ECMWF | -0.002 | +0.000 | +0.036 | +0.054 | -0.077 |
| Lags | +0.001 | +0.000 | +0.008 | +0.022 | -0.059 |
| Rolling | +0.003 | +0.002 | -0.065 | -0.030 | -0.017 |
| NWP_atmosphere | +0.004 | +0.000 | -0.022 | -0.011 | +0.020 |
| Physics | +0.004 | +0.003 | +0.007 | +0.008 | -0.028 |
| Time | -0.002 | +0.001 | +0.055 | +0.005 | -0.015 |
| NWP_primary | -0.005 | +0.213 | +0.753 | +0.700 | +0.543 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.012 | -0.001 | -0.033 | -0.003 | +0.007 |
| ECMWF | +0.003 | +0.000 | -0.044 | -0.051 | +0.002 |
| Lags | +0.007 | -0.006 | -0.068 | +0.039 | -0.020 |
| Rolling | +0.024 | +0.010 | -0.026 | -0.008 | +0.069 |
| NWP_atmosphere | +0.010 | +0.012 | -0.020 | +0.019 | +0.011 |
| Physics | +0.011 | +0.036 | +0.032 | +0.034 | +0.050 |
| Time | +0.000 | -0.001 | +0.009 | -0.022 | -0.033 |
| NWP_primary | +0.534 | +0.531 | +0.484 | +0.458 | +0.422 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.675 | 0.653 | 0.944 | Fall | Winter |
| MinT | 0.975 | 1.444 | 1.292 | Summer | Fall |

#### Station Insights

- **MaxT**: Best MAE (0.716 F, Linear) is 20.8% below the Continental group mean (0.904 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.441 F avg across architectures).
- **MaxT ablation**: Removing Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.486 F avg across architectures).
- **MaxT seasonality**: Winter MAE (0.944 F) is 45% worse than Fall (0.653 F).
- **MinT seasonality**: Fall MAE (1.444 F) is 48% worse than Summer (0.975 F).

---

## 2. NE Coastal (5 stations)

Stations: KNYC, KLGA, KBOS, KPHL, KDCA

### 2.1 KNYC -- New York Central Park, NY

**Climate zone**: NE Coastal | **Coordinates**: 40.78 N, 73.97 W | **Timezone**: America/New_York
  
*Note: Co-op station, not ASOS. Verify ACIS data.*

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.845 | 1.091 | 0.19 | 1.124 | 1.450 | 0.02 |
| **MLP** | **0.823** | 1.063 | 0.12 | 1.154 | 1.475 | -0.03 |
| XGBoost | 1.277 | 1.947 | 0.35 | 1.486 | 2.068 | 0.12 |
| LightGBM | 1.596 | 2.963 | 0.62 | 1.553 | 2.281 | 0.18 |
| CatBoost | 1.418 | 2.374 | 0.38 | 1.544 | 2.207 | 0.20 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.248** | 1.647 | 0.41 | 1.494 | 2.021 | 0.08 |
| MLP | 1.281 | 1.680 | 0.42 | 1.491 | 2.043 | 0.01 |
| XGBoost | 1.378 | 1.887 | 0.26 | 1.640 | 2.191 | 0.02 |
| LightGBM | 1.989 | 3.503 | 1.03 | 1.710 | 2.445 | 0.18 |
| CatBoost | 1.625 | 2.282 | 0.48 | 1.716 | 2.373 | 0.14 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.068 | 1.942 | 1.265 | 0.823 | 0.23 | 22.9% |
| MinT | 2.027 | 3.034 | 2.471 | 1.248 | 0.38 | 38.4% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.004 | +0.016 | -0.077 | +0.029 | -0.050 |
| ECMWF | -0.001 | -0.000 | +0.093 | +0.059 | -0.039 |
| Lags | +0.003 | +0.002 | -0.052 | +0.061 | -0.144 |
| Rolling | -0.004 | +0.007 | -0.039 | -0.001 | -0.132 |
| NWP_atmosphere | -0.004 | +0.003 | +0.001 | +0.027 | +0.015 |
| Physics | -0.011 | -0.005 | -0.053 | +0.022 | -0.089 |
| Time | +0.002 | +0.014 | +0.114 | +0.117 | -0.133 |
| NWP_primary | -0.006 | +0.066 | +0.393 | +0.603 | +0.288 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.024 | -0.004 | +0.015 | -0.067 | +0.063 |
| ECMWF | +0.000 | +0.000 | +0.025 | -0.062 | +0.043 |
| Lags | -0.001 | +0.000 | +0.094 | -0.034 | +0.118 |
| Rolling | -0.001 | +0.000 | +0.043 | -0.059 | +0.047 |
| NWP_atmosphere | +0.015 | +0.011 | +0.138 | +0.011 | +0.106 |
| Physics | +0.010 | +0.011 | -0.030 | -0.102 | +0.093 |
| Time | +0.008 | -0.000 | +0.032 | -0.058 | +0.081 |
| NWP_primary | +0.622 | +0.540 | +0.554 | +0.504 | +0.491 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.817 | 0.751 | 1.114 | Fall | Winter |
| MinT | 0.980 | 1.286 | 1.751 | Summer | Winter |

#### Station Insights

- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.269 F avg across architectures).
- **MaxT ablation**: Removing Lags, Physics, Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.542 F avg across architectures).
- **MaxT seasonality**: Winter MAE (1.114 F) is 48% worse than Fall (0.751 F).
- **MinT seasonality**: Winter MAE (1.751 F) is 79% worse than Summer (0.980 F).

---

### 2.2 KLGA -- LaGuardia, NY

**Climate zone**: NE Coastal | **Coordinates**: 40.78 N, 73.87 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.859 | 1.068 | 0.02 | 1.069 | 1.351 | -0.06 |
| **MLP** | **0.850** | 1.068 | -0.08 | 1.059 | 1.336 | -0.17 |
| XGBoost | 1.289 | 1.971 | 0.13 | 1.300 | 1.765 | 0.23 |
| LightGBM | 1.601 | 2.832 | 0.52 | 1.453 | 2.145 | 0.32 |
| CatBoost | 1.500 | 2.372 | 0.24 | 1.424 | 1.928 | 0.15 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.278** | 1.683 | 0.13 | 1.257 | 1.687 | 0.19 |
| MLP | 1.343 | 1.762 | 0.36 | 1.248 | 1.679 | 0.22 |
| XGBoost | 1.500 | 2.240 | 0.53 | 1.416 | 2.018 | 0.28 |
| LightGBM | 1.986 | 3.533 | 1.19 | 1.490 | 2.245 | 0.26 |
| CatBoost | 1.689 | 2.495 | 0.76 | 1.489 | 2.107 | 0.28 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.929 | 2.244 | 1.401 | 0.850 | 0.09 | 8.5% |
| MinT | 2.728 | 3.102 | 2.879 | 1.278 | 0.53 | 53.1% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.007 | +0.009 | -0.066 | +0.035 | -0.059 |
| ECMWF | -0.003 | +0.001 | +0.009 | +0.064 | -0.124 |
| Lags | -0.005 | +0.000 | +0.130 | +0.023 | -0.088 |
| Rolling | -0.000 | -0.001 | +0.010 | +0.008 | -0.001 |
| NWP_atmosphere | +0.001 | +0.027 | -0.071 | +0.030 | +0.032 |
| Physics | -0.008 | -0.010 | +0.012 | +0.080 | -0.044 |
| Time | +0.000 | +0.000 | +0.014 | +0.069 | -0.085 |
| NWP_primary | +0.014 | +0.102 | +0.457 | +0.456 | +0.320 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.025 | +0.061 | -0.044 | -0.141 | -0.088 |
| ECMWF | -0.001 | +0.002 | +0.016 | -0.078 | -0.025 |
| Lags | +0.002 | +0.000 | -0.097 | -0.014 | -0.048 |
| Rolling | +0.045 | +0.034 | -0.028 | -0.037 | -0.089 |
| NWP_atmosphere | +0.074 | +0.083 | +0.021 | -0.033 | +0.075 |
| Physics | +0.030 | +0.013 | -0.019 | -0.023 | -0.010 |
| Time | -0.000 | -0.020 | +0.015 | -0.092 | +0.020 |
| NWP_primary | +0.488 | +0.438 | +0.587 | +0.447 | +0.367 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.863 | 0.807 | 0.963 | Fall | Winter |
| MinT | 1.129 | 1.249 | 1.671 | Summer | Winter |

#### Station Insights

- **MinT**: Best MAE (1.278 F, Linear) is 10.5% above the NE Coastal group mean (1.157 F).
- MinT post-processing delivers exceptional 53.1% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.270 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.465 F avg across architectures).
- **MinT ablation**: Removing Bias, Lags actually *improves* performance (possible overfitting from these features).
- **MinT seasonality**: Winter MAE (1.671 F) is 48% worse than Summer (1.129 F).

---

### 2.3 KBOS -- Boston, MA

**Climate zone**: NE Coastal | **Coordinates**: 42.37 N, 71.01 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.927** | 1.211 | -0.23 | 1.227 | 1.578 | -0.36 |
| MLP | 0.933 | 1.208 | -0.27 | 1.225 | 1.547 | -0.28 |
| XGBoost | 1.242 | 1.715 | -0.32 | 1.462 | 1.951 | -0.37 |
| LightGBM | 1.506 | 2.648 | 0.25 | 1.495 | 2.143 | -0.06 |
| CatBoost | 1.329 | 1.963 | -0.18 | 1.442 | 1.933 | -0.22 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.253** | 1.633 | -0.21 | 1.354 | 1.775 | -0.26 |
| MLP | 1.254 | 1.643 | -0.04 | 1.368 | 1.808 | -0.18 |
| XGBoost | 1.545 | 2.182 | -0.58 | 1.441 | 1.956 | -0.20 |
| LightGBM | 1.770 | 2.918 | 0.49 | 1.543 | 2.113 | -0.03 |
| CatBoost | 1.552 | 2.322 | 0.21 | 1.370 | 1.852 | -0.23 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.194 | 1.906 | 1.352 | 0.927 | 0.22 | 22.4% |
| MinT | 2.802 | 1.966 | 2.268 | 1.253 | 0.55 | 55.3% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.048 | +0.047 | +0.011 | +0.015 | +0.052 |
| ECMWF | +0.009 | +0.008 | +0.053 | +0.100 | +0.083 |
| Lags | +0.003 | +0.000 | +0.072 | +0.030 | +0.094 |
| Rolling | +0.007 | +0.001 | -0.015 | -0.000 | +0.094 |
| NWP_atmosphere | -0.019 | +0.004 | -0.032 | +0.035 | +0.045 |
| Physics | -0.012 | +0.000 | +0.044 | +0.017 | +0.064 |
| Time | +0.001 | -0.000 | +0.013 | +0.033 | +0.061 |
| NWP_primary | -0.001 | +0.060 | +0.413 | +0.437 | +0.411 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.059 | +0.039 | -0.159 | +0.000 | -0.099 |
| ECMWF | -0.005 | +0.004 | -0.140 | +0.064 | +0.044 |
| Lags | -0.002 | +0.007 | -0.101 | -0.001 | +0.051 |
| Rolling | +0.025 | +0.006 | -0.142 | -0.016 | +0.061 |
| NWP_atmosphere | +0.041 | +0.064 | -0.092 | +0.007 | -0.061 |
| Physics | -0.006 | -0.006 | -0.138 | +0.024 | -0.031 |
| Time | -0.017 | -0.003 | -0.152 | +0.037 | -0.001 |
| NWP_primary | +0.092 | +0.057 | -0.054 | +0.038 | +0.178 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 1.057 | 0.804 | 0.913 | Fall | Summer |
| MinT | 1.226 | 1.198 | 1.431 | Fall | Winter |

#### Station Insights

- MinT post-processing delivers exceptional 55.3% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.264 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.062 F avg across architectures).
- **MinT ablation**: Removing Bias, Physics, Time actually *improves* performance (possible overfitting from these features).
- **MaxT seasonality**: Summer MAE (1.057 F) is 31% worse than Fall (0.804 F).

---

### 2.4 KPHL -- Philadelphia, PA

**Climate zone**: NE Coastal | **Coordinates**: 39.87 N, 75.24 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.811 | 1.053 | -0.40 | 1.048 | 1.421 | -0.36 |
| **MLP** | **0.798** | 1.048 | -0.43 | 1.047 | 1.421 | -0.38 |
| XGBoost | 1.178 | 1.843 | -0.31 | 1.211 | 1.723 | -0.19 |
| LightGBM | 1.639 | 2.933 | -0.04 | 1.405 | 2.198 | -0.14 |
| CatBoost | 1.417 | 2.115 | -0.47 | 1.341 | 1.827 | -0.30 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.026 | 1.303 | -0.24 | 1.111 | 1.455 | -0.04 |
| **MLP** | **1.001** | 1.272 | -0.24 | 1.110 | 1.463 | -0.03 |
| XGBoost | 1.268 | 1.838 | 0.02 | 1.328 | 1.706 | -0.06 |
| LightGBM | 1.582 | 2.696 | 0.45 | 1.386 | 1.900 | 0.10 |
| CatBoost | 1.278 | 1.791 | -0.02 | 1.434 | 1.852 | 0.01 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.410 | 2.831 | 2.073 | 0.798 | 0.43 | 43.4% |
| MinT | 1.403 | 2.568 | 1.838 | 1.001 | 0.29 | 28.6% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.198 | +0.172 | +0.078 | -0.034 | +0.145 |
| ECMWF | +0.004 | -0.000 | +0.086 | -0.095 | -0.028 |
| Lags | +0.002 | -0.000 | +0.005 | -0.060 | -0.013 |
| Rolling | -0.006 | -0.008 | -0.069 | -0.031 | -0.042 |
| NWP_atmosphere | -0.005 | +0.005 | +0.078 | -0.042 | +0.030 |
| Physics | +0.024 | +0.012 | -0.031 | -0.016 | -0.038 |
| Time | -0.002 | +0.010 | -0.044 | -0.021 | +0.119 |
| NWP_primary | +0.010 | +0.213 | +0.724 | +0.650 | +0.569 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.102 | +0.088 | -0.074 | +0.026 | +0.112 |
| ECMWF | +0.000 | +0.002 | +0.032 | +0.062 | +0.128 |
| Lags | -0.004 | +0.001 | -0.048 | -0.068 | +0.057 |
| Rolling | +0.012 | -0.011 | +0.077 | +0.095 | +0.041 |
| NWP_atmosphere | +0.011 | +0.008 | +0.110 | +0.059 | +0.066 |
| Physics | -0.013 | -0.002 | -0.008 | +0.010 | +0.076 |
| Time | +0.015 | +0.007 | +0.027 | +0.040 | +0.067 |
| NWP_primary | +0.224 | +0.246 | +0.322 | +0.219 | +0.337 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.859 | 0.743 | 0.853 | Fall | Summer |
| MinT | 0.928 | 0.912 | 1.493 | Fall | Winter |

#### Station Insights

- **MaxT**: Best MAE (0.798 F, MLP) is 12.1% below the NE Coastal group mean (0.908 F).
- **MinT**: Best MAE (1.001 F, MLP) is 13.5% below the NE Coastal group mean (1.157 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.433 F avg across architectures).
- **MaxT ablation**: Removing Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.270 F avg across architectures).
- **MinT seasonality**: Winter MAE (1.493 F) is 64% worse than Fall (0.912 F).

---

### 2.5 KDCA -- Washington D.C.

**Climate zone**: NE Coastal | **Coordinates**: 38.85 N, 77.04 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.157 | 1.401 | 0.51 | 1.319 | 1.709 | -0.03 |
| **MLP** | **1.139** | 1.392 | 0.56 | 1.344 | 1.732 | -0.03 |
| XGBoost | 1.408 | 2.068 | 0.67 | 1.668 | 2.535 | 0.26 |
| LightGBM | 1.872 | 3.331 | 1.26 | 1.753 | 2.706 | 0.36 |
| CatBoost | 1.524 | 2.269 | 0.93 | 1.580 | 2.354 | 0.10 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.006** | 1.323 | 0.44 | 1.190 | 1.594 | 0.09 |
| MLP | 1.054 | 1.378 | 0.58 | 1.191 | 1.599 | 0.09 |
| XGBoost | 1.515 | 2.224 | 1.04 | 1.446 | 2.062 | 0.27 |
| LightGBM | 1.748 | 2.730 | 1.23 | 1.563 | 2.368 | 0.45 |
| CatBoost | 1.639 | 2.316 | 1.07 | 1.474 | 2.158 | 0.31 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.138 | 2.022 | 1.452 | 1.139 | -0.00 | -0.1% |
| MinT | 1.657 | 2.030 | 1.771 | 1.006 | 0.39 | 39.3% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.097 | +0.080 | -0.044 | +0.034 | -0.011 |
| ECMWF | -0.034 | -0.016 | -0.022 | +0.108 | -0.029 |
| Lags | -0.007 | -0.002 | -0.021 | +0.019 | -0.053 |
| Rolling | +0.003 | -0.017 | -0.006 | -0.063 | -0.066 |
| NWP_atmosphere | -0.019 | -0.024 | +0.047 | +0.008 | -0.030 |
| Physics | -0.001 | -0.002 | -0.005 | +0.031 | -0.109 |
| Time | -0.002 | -0.014 | +0.066 | -0.085 | -0.033 |
| NWP_primary | +0.005 | +0.047 | +0.334 | +0.230 | +0.242 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.119 | +0.105 | -0.056 | +0.036 | -0.128 |
| ECMWF | -0.000 | +0.000 | +0.005 | +0.002 | -0.204 |
| Lags | -0.007 | -0.000 | +0.064 | -0.009 | -0.101 |
| Rolling | +0.028 | +0.005 | -0.142 | +0.107 | -0.060 |
| NWP_atmosphere | +0.051 | +0.017 | -0.094 | -0.011 | -0.206 |
| Physics | +0.007 | +0.007 | +0.024 | +0.020 | -0.031 |
| Time | -0.008 | -0.030 | +0.050 | +0.005 | -0.229 |
| NWP_primary | +0.390 | +0.314 | +0.504 | +0.280 | +0.199 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 1.119 | 1.154 | 1.247 | Summer | Winter |
| MinT | 0.777 | 1.146 | 1.205 | Summer | Winter |

#### Station Insights

- **MaxT**: Best MAE (1.139 F, MLP) is 25.5% above the NE Coastal group mean (0.908 F).
- **MinT**: Best MAE (1.006 F, Linear) is 13.1% below the NE Coastal group mean (1.157 F).
- MaxT shows minimal improvement (-0.1%) over raw GFS -- GFS already performs well here.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.172 F avg across architectures).
- **MaxT ablation**: Removing Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.338 F avg across architectures).
- **MinT ablation**: Removing ECMWF, NWP_atmosphere, Time actually *improves* performance (possible overfitting from these features).
- **MinT seasonality**: Winter MAE (1.205 F) is 55% worse than Summer (0.777 F).

---

## 3. SE Subtropical (6 stations)

Stations: KATL, KCLT, KBNA, KJAX, KTPA, KMIA

### 3.1 KATL -- Atlanta, GA

**Climate zone**: SE Subtropical | **Coordinates**: 33.63 N, 84.42 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.796** | 1.040 | -0.20 | 1.219 | 1.892 | -0.12 |
| MLP | 0.815 | 1.059 | -0.26 | 1.205 | 1.885 | -0.13 |
| XGBoost | 1.074 | 1.399 | -0.28 | 1.354 | 2.025 | 0.02 |
| LightGBM | 1.186 | 1.804 | -0.17 | 1.465 | 2.270 | 0.15 |
| CatBoost | 1.205 | 1.601 | -0.46 | 1.377 | 2.030 | -0.21 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.838 | 1.059 | 0.15 | 1.075 | 1.365 | 0.13 |
| **MLP** | **0.834** | 1.056 | 0.17 | 1.087 | 1.389 | 0.07 |
| XGBoost | 0.996 | 1.300 | -0.05 | 1.230 | 1.638 | -0.13 |
| LightGBM | 1.131 | 1.925 | 0.20 | 1.348 | 2.039 | 0.07 |
| CatBoost | 1.083 | 1.441 | -0.09 | 1.246 | 1.616 | -0.31 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 2.107 | 3.405 | 2.721 | 0.796 | 0.62 | 62.2% |
| MinT | 1.301 | 2.416 | 1.789 | 0.834 | 0.36 | 35.9% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.015 | +0.007 | -0.048 | +0.002 | -0.058 |
| ECMWF | +0.011 | +0.002 | -0.011 | -0.009 | -0.038 |
| Lags | +0.001 | -0.002 | +0.022 | -0.004 | -0.071 |
| Rolling | +0.003 | +0.005 | -0.046 | -0.033 | -0.056 |
| NWP_atmosphere | +0.023 | -0.000 | -0.031 | -0.054 | -0.075 |
| Physics | +0.017 | +0.002 | -0.009 | -0.030 | -0.086 |
| Time | +0.001 | -0.002 | +0.030 | +0.023 | -0.048 |
| NWP_primary | +0.015 | +0.131 | +0.388 | +0.367 | +0.300 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.002 | -0.000 | -0.017 | +0.031 | -0.041 |
| ECMWF | -0.000 | -0.000 | +0.018 | +0.017 | +0.047 |
| Lags | -0.000 | +0.000 | +0.022 | +0.013 | -0.052 |
| Rolling | +0.001 | -0.001 | -0.037 | -0.019 | -0.067 |
| NWP_atmosphere | +0.004 | +0.000 | +0.012 | -0.018 | -0.068 |
| Physics | +0.010 | +0.008 | +0.016 | -0.008 | -0.007 |
| Time | -0.002 | +0.000 | -0.012 | +0.009 | +0.076 |
| NWP_primary | +0.221 | +0.164 | +0.280 | +0.271 | +0.271 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.899 | 0.746 | 0.682 | Winter | Summer |
| MinT | 0.665 | 0.828 | 1.237 | Summer | Winter |

#### Station Insights

- **MinT**: Best MAE (0.834 F, MLP) is 15.1% below the SE Subtropical group mean (0.982 F).
- MaxT post-processing delivers exceptional 62.2% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.240 F avg across architectures).
- **MaxT ablation**: Removing NWP_atmosphere, Physics, Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.241 F avg across architectures).
- **MinT ablation**: Removing Rolling actually *improves* performance (possible overfitting from these features).
- **MaxT seasonality**: Summer MAE (0.899 F) is 32% worse than Winter (0.682 F).
- **MinT seasonality**: Winter MAE (1.237 F) is 86% worse than Summer (0.665 F).

---

### 3.2 KCLT -- Charlotte, NC

**Climate zone**: SE Subtropical | **Coordinates**: 35.21 N, 80.95 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.878** | 1.098 | -0.25 | 1.220 | 1.602 | -0.00 |
| MLP | 0.899 | 1.112 | -0.26 | 1.238 | 1.619 | 0.03 |
| XGBoost | 1.134 | 1.593 | -0.24 | 1.482 | 2.064 | 0.24 |
| LightGBM | 1.373 | 2.233 | 0.02 | 1.564 | 2.287 | 0.23 |
| CatBoost | 1.178 | 1.747 | -0.22 | 1.421 | 1.876 | 0.05 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.971 | 1.288 | -0.15 | 1.404 | 1.935 | -0.34 |
| **MLP** | **0.957** | 1.268 | -0.15 | 1.467 | 2.001 | -0.44 |
| XGBoost | 1.202 | 1.740 | -0.14 | 1.548 | 2.064 | -0.41 |
| LightGBM | 1.375 | 2.154 | -0.05 | 1.646 | 2.287 | -0.26 |
| CatBoost | 1.380 | 1.907 | -0.20 | 1.677 | 2.237 | -0.41 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.954 | 2.349 | 2.090 | 0.878 | 0.55 | 55.1% |
| MinT | 1.057 | 1.233 | 0.989 | 0.957 | 0.10 | 9.5% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.053 | +0.031 | +0.004 | +0.056 | -0.015 |
| ECMWF | +0.017 | +0.009 | +0.047 | +0.073 | -0.013 |
| Lags | +0.001 | +0.000 | +0.029 | +0.023 | +0.033 |
| Rolling | -0.003 | +0.001 | -0.029 | +0.043 | -0.072 |
| NWP_atmosphere | +0.019 | -0.001 | +0.048 | +0.017 | +0.024 |
| Physics | +0.007 | +0.010 | +0.005 | +0.025 | -0.009 |
| Time | -0.003 | -0.003 | +0.031 | +0.008 | -0.027 |
| NWP_primary | +0.006 | +0.092 | +0.450 | +0.336 | +0.281 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.029 | +0.010 | +0.050 | -0.006 | -0.025 |
| ECMWF | -0.002 | +0.000 | -0.021 | -0.000 | -0.038 |
| Lags | +0.002 | +0.000 | -0.064 | -0.019 | +0.019 |
| Rolling | +0.010 | +0.004 | -0.046 | +0.006 | -0.074 |
| NWP_atmosphere | -0.013 | +0.004 | -0.032 | -0.016 | -0.029 |
| Physics | -0.025 | -0.002 | +0.008 | -0.028 | -0.033 |
| Time | +0.001 | -0.007 | +0.019 | -0.025 | -0.017 |
| NWP_primary | +0.390 | +0.399 | +0.474 | +0.415 | +0.336 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 1.013 | 0.720 | 0.926 | Fall | Summer |
| MinT | 0.696 | 1.101 | 1.293 | Summer | Winter |

#### Station Insights

- MaxT post-processing delivers exceptional 55.1% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.233 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.403 F avg across architectures).
- **MaxT seasonality**: Summer MAE (1.013 F) is 41% worse than Fall (0.720 F).
- **MinT seasonality**: Winter MAE (1.293 F) is 86% worse than Summer (0.696 F).

---

### 3.3 KBNA -- Nashville, TN

**Climate zone**: SE Subtropical | **Coordinates**: 36.12 N, 86.68 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.816 | 1.074 | -0.23 | 1.208 | 1.609 | -0.23 |
| **MLP** | **0.811** | 1.064 | -0.21 | 1.259 | 1.648 | -0.24 |
| XGBoost | 1.003 | 1.370 | -0.23 | 1.330 | 1.718 | -0.32 |
| LightGBM | 1.172 | 1.997 | -0.07 | 1.420 | 2.127 | -0.13 |
| CatBoost | 1.063 | 1.420 | -0.28 | 1.344 | 1.796 | -0.30 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.901** | 1.207 | 0.36 | 1.197 | 1.552 | 0.13 |
| MLP | 0.927 | 1.235 | 0.43 | 1.185 | 1.549 | 0.13 |
| XGBoost | 1.112 | 1.504 | 0.14 | 1.298 | 1.698 | -0.08 |
| LightGBM | 1.147 | 1.642 | 0.27 | 1.365 | 1.811 | 0.10 |
| CatBoost | 1.144 | 1.578 | 0.14 | 1.369 | 1.807 | -0.27 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 2.617 | 2.723 | 2.612 | 0.811 | 0.69 | 69.0% |
| MinT | 1.057 | 1.308 | 1.000 | 0.901 | 0.15 | 14.8% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.000 | +0.008 | -0.010 | +0.003 | -0.040 |
| ECMWF | +0.004 | +0.002 | -0.087 | -0.052 | +0.035 |
| Lags | +0.001 | -0.000 | -0.037 | +0.001 | -0.046 |
| Rolling | -0.016 | -0.000 | -0.068 | -0.051 | +0.007 |
| NWP_atmosphere | -0.001 | +0.004 | -0.042 | -0.039 | +0.174 |
| Physics | -0.006 | -0.005 | -0.059 | +0.006 | +0.119 |
| Time | -0.002 | -0.000 | -0.036 | +0.006 | -0.002 |
| NWP_primary | +0.010 | +0.178 | +0.380 | +0.377 | +0.390 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.006 | +0.003 | +0.051 | +0.012 | +0.021 |
| ECMWF | -0.000 | -0.004 | +0.023 | +0.006 | +0.042 |
| Lags | +0.003 | +0.001 | +0.015 | +0.020 | +0.072 |
| Rolling | +0.006 | +0.005 | +0.012 | -0.006 | +0.050 |
| NWP_atmosphere | +0.021 | +0.008 | +0.000 | +0.007 | +0.115 |
| Physics | +0.023 | +0.014 | -0.006 | +0.032 | -0.025 |
| Time | -0.010 | -0.010 | +0.064 | -0.021 | +0.037 |
| NWP_primary | +0.518 | +0.465 | +0.437 | +0.443 | +0.514 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.907 | 0.756 | 0.748 | Winter | Summer |
| MinT | 0.665 | 0.953 | 1.303 | Summer | Winter |

#### Station Insights

- MaxT post-processing delivers exceptional 69.0% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.267 F avg across architectures).
- **MaxT ablation**: Removing Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.475 F avg across architectures).
- **MinT seasonality**: Winter MAE (1.303 F) is 96% worse than Summer (0.665 F).

---

### 3.4 KJAX -- Jacksonville, FL

**Climate zone**: SE Subtropical | **Coordinates**: 30.49 N, 81.69 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.844 | 1.107 | -0.27 | 1.255 | 1.598 | -0.78 |
| **MLP** | **0.817** | 1.083 | -0.24 | 1.307 | 1.639 | -0.88 |
| XGBoost | 0.897 | 1.213 | -0.28 | 1.463 | 1.829 | -1.02 |
| LightGBM | 1.059 | 1.736 | -0.09 | 1.543 | 1.998 | -1.01 |
| CatBoost | 0.999 | 1.363 | -0.54 | 1.486 | 1.865 | -1.06 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.028 | 1.356 | -0.41 | 1.201 | 1.624 | 0.10 |
| **MLP** | **1.012** | 1.361 | -0.34 | 1.217 | 1.632 | 0.12 |
| XGBoost | 1.231 | 1.700 | -0.16 | 1.399 | 1.911 | 0.14 |
| LightGBM | 1.240 | 1.884 | -0.09 | 1.434 | 2.058 | 0.22 |
| CatBoost | 1.371 | 1.951 | -0.22 | 1.379 | 1.847 | 0.15 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 2.190 | 3.020 | 2.581 | 0.817 | 0.63 | 62.7% |
| MinT | 1.895 | 1.690 | 1.588 | 1.012 | 0.47 | 46.6% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.074 | +0.095 | +0.073 | +0.033 | +0.122 |
| ECMWF | +0.008 | +0.007 | -0.003 | -0.047 | +0.047 |
| Lags | -0.004 | +0.000 | +0.019 | +0.014 | +0.051 |
| Rolling | -0.005 | +0.007 | -0.033 | +0.018 | +0.025 |
| NWP_atmosphere | -0.024 | -0.004 | +0.048 | +0.035 | +0.063 |
| Physics | -0.018 | +0.000 | -0.010 | -0.012 | +0.055 |
| Time | -0.020 | +0.006 | -0.023 | -0.001 | +0.081 |
| NWP_primary | -0.024 | +0.060 | +0.338 | +0.281 | +0.206 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.078 | +0.073 | +0.016 | +0.053 | +0.107 |
| ECMWF | -0.034 | -0.011 | -0.045 | +0.063 | -0.140 |
| Lags | -0.007 | -0.006 | -0.124 | +0.016 | -0.088 |
| Rolling | +0.005 | -0.012 | -0.049 | -0.001 | +0.005 |
| NWP_atmosphere | +0.009 | +0.003 | -0.102 | +0.004 | -0.059 |
| Physics | -0.020 | -0.042 | -0.016 | +0.009 | -0.018 |
| Time | -0.008 | +0.000 | -0.106 | -0.032 | -0.028 |
| NWP_primary | +0.437 | +0.419 | +0.364 | +0.482 | +0.283 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.978 | 0.782 | 0.688 | Winter | Summer |
| MinT | 0.774 | 0.983 | 1.683 | Summer | Winter |

#### Station Insights

- MaxT post-processing delivers exceptional 62.7% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.172 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.397 F avg across architectures).
- **MinT ablation**: Removing ECMWF, Lags, NWP_atmosphere, Time actually *improves* performance (possible overfitting from these features).
- **MaxT seasonality**: Summer MAE (0.978 F) is 42% worse than Winter (0.688 F).
- **MinT seasonality**: Winter MAE (1.683 F) is 117% worse than Summer (0.774 F).

---

### 3.5 KTPA -- Tampa, FL

**Climate zone**: SE Subtropical | **Coordinates**: 27.98 N, 82.53 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.045 | 1.330 | -0.34 | 1.211 | 1.514 | -0.15 |
| **MLP** | **1.033** | 1.311 | -0.33 | 1.232 | 1.531 | -0.12 |
| XGBoost | 1.160 | 1.522 | -0.08 | 1.379 | 1.828 | 0.12 |
| LightGBM | 1.252 | 1.888 | -0.03 | 1.323 | 1.762 | -0.07 |
| CatBoost | 1.240 | 1.678 | -0.30 | 1.283 | 1.662 | -0.07 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.044** | 1.345 | -0.01 | 1.326 | 1.738 | 0.04 |
| MLP | 1.045 | 1.380 | 0.05 | 1.325 | 1.789 | 0.11 |
| XGBoost | 1.180 | 1.589 | 0.01 | 1.415 | 1.908 | 0.15 |
| LightGBM | 1.496 | 2.454 | 0.36 | 1.512 | 2.137 | 0.21 |
| CatBoost | 1.299 | 1.816 | 0.14 | 1.430 | 1.930 | 0.13 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.713 | 2.799 | 2.201 | 1.033 | 0.40 | 39.7% |
| MinT | 2.329 | 2.062 | 2.170 | 1.044 | 0.55 | 55.2% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.069 | +0.070 | +0.061 | +0.048 | +0.027 |
| ECMWF | +0.003 | +0.006 | -0.006 | -0.002 | -0.106 |
| Lags | +0.002 | +0.002 | -0.001 | +0.005 | -0.085 |
| Rolling | +0.011 | +0.001 | -0.003 | +0.003 | -0.049 |
| NWP_atmosphere | -0.006 | +0.001 | -0.007 | -0.009 | -0.039 |
| Physics | +0.001 | +0.014 | -0.018 | +0.001 | -0.048 |
| Time | -0.001 | +0.011 | -0.021 | +0.005 | -0.073 |
| NWP_primary | +0.001 | +0.086 | +0.272 | +0.262 | +0.146 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.010 | +0.029 | +0.065 | +0.003 | +0.089 |
| ECMWF | -0.001 | -0.000 | +0.059 | -0.037 | +0.059 |
| Lags | -0.002 | -0.000 | +0.008 | -0.005 | -0.023 |
| Rolling | -0.023 | -0.000 | -0.003 | -0.017 | +0.007 |
| NWP_atmosphere | -0.018 | -0.006 | +0.065 | +0.016 | +0.091 |
| Physics | -0.007 | +0.004 | -0.001 | -0.026 | -0.028 |
| Time | -0.000 | +0.002 | +0.052 | -0.004 | +0.019 |
| NWP_primary | +0.152 | +0.157 | +0.240 | +0.146 | +0.192 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.994 | 1.107 | 1.019 | Summer | Fall |
| MinT | 1.057 | 0.996 | 1.120 | Fall | Winter |

#### Station Insights

- **MaxT**: Best MAE (1.033 F, MLP) is 20.0% above the SE Subtropical group mean (0.861 F).
- MinT post-processing delivers exceptional 55.2% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.153 F avg across architectures).
- **MaxT ablation**: Removing ECMWF actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.177 F avg across architectures).

---

### 3.6 KMIA -- Miami, FL

**Climate zone**: SE Subtropical | **Coordinates**: 25.79 N, 80.32 W | **Timezone**: America/New_York

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.831** | 1.076 | -0.07 | 1.133 | 1.440 | 0.45 |
| MLP | 0.845 | 1.090 | -0.10 | 1.106 | 1.424 | 0.38 |
| XGBoost | 0.895 | 1.344 | 0.02 | 1.202 | 1.737 | 0.39 |
| LightGBM | 1.089 | 1.974 | 0.28 | 1.298 | 2.055 | 0.52 |
| CatBoost | 0.959 | 1.425 | 0.01 | 1.207 | 1.733 | 0.34 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.146** | 1.544 | 0.81 | 1.170 | 1.584 | 0.36 |
| MLP | 1.153 | 1.551 | 0.80 | 1.202 | 1.620 | 0.44 |
| XGBoost | 1.444 | 2.127 | 1.08 | 1.364 | 1.977 | 0.48 |
| LightGBM | 1.790 | 3.000 | 1.44 | 1.466 | 2.373 | 0.70 |
| CatBoost | 1.529 | 2.319 | 1.10 | 1.400 | 2.038 | 0.54 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.400 | 3.136 | 2.192 | 0.831 | 0.41 | 40.7% |
| MinT | 1.182 | 1.418 | 1.089 | 1.146 | 0.03 | 3.1% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.010 | +0.059 | -0.017 | +0.032 | -0.035 |
| ECMWF | -0.012 | -0.003 | -0.021 | +0.026 | -0.030 |
| Lags | -0.001 | +0.000 | -0.014 | +0.038 | -0.032 |
| Rolling | -0.016 | -0.014 | -0.024 | +0.012 | +0.008 |
| NWP_atmosphere | -0.015 | +0.009 | +0.014 | +0.007 | +0.041 |
| Physics | +0.010 | -0.003 | -0.009 | +0.025 | +0.040 |
| Time | +0.002 | +0.000 | -0.008 | +0.009 | +0.049 |
| NWP_primary | +0.018 | +0.040 | +0.150 | +0.155 | +0.173 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.118 | +0.142 | -0.006 | +0.004 | -0.012 |
| ECMWF | -0.000 | +0.000 | -0.031 | -0.049 | -0.100 |
| Lags | +0.003 | +0.004 | -0.036 | -0.028 | -0.042 |
| Rolling | +0.014 | -0.008 | -0.011 | +0.018 | -0.015 |
| NWP_atmosphere | +0.061 | +0.097 | -0.001 | -0.041 | -0.030 |
| Physics | -0.006 | -0.001 | -0.039 | -0.050 | +0.013 |
| Time | -0.007 | +0.013 | -0.033 | -0.046 | -0.106 |
| NWP_primary | +0.837 | +0.976 | +0.601 | +0.485 | +0.462 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.939 | 0.756 | 0.755 | Winter | Summer |
| MinT | 1.123 | 0.990 | 1.538 | Fall | Winter |

#### Station Insights

- **MinT**: Best MAE (1.146 F, Linear) is 16.7% above the SE Subtropical group mean (0.982 F).
- MinT shows minimal improvement (3.1%) over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.107 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.672 F avg across architectures).
- **MinT ablation**: Removing ECMWF, Time actually *improves* performance (possible overfitting from these features).
- **MinT seasonality**: Winter MAE (1.538 F) is 55% worse than Fall (0.990 F).

---

## 4. Gulf/SC (6 stations)

Stations: KHOU, KMSY, KDFW, KDAL, KAUS, KSAT

### 4.1 KHOU -- Houston, TX

**Climate zone**: Gulf/SC | **Coordinates**: 29.64 N, 95.28 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.924** | 1.138 | 0.35 | 1.060 | 1.371 | -0.22 |
| MLP | 0.925 | 1.141 | 0.36 | 1.077 | 1.393 | -0.24 |
| XGBoost | 1.023 | 1.297 | 0.24 | 1.154 | 1.476 | -0.15 |
| LightGBM | 1.179 | 1.826 | 0.40 | 1.238 | 1.710 | -0.16 |
| CatBoost | 1.118 | 1.429 | 0.13 | 1.259 | 1.581 | -0.13 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.887 | 1.117 | 0.43 | 1.091 | 1.423 | -0.01 |
| **MLP** | **0.879** | 1.117 | 0.44 | 1.130 | 1.463 | -0.00 |
| XGBoost | 1.016 | 1.391 | 0.42 | 1.230 | 1.585 | -0.17 |
| LightGBM | 1.163 | 1.884 | 0.62 | 1.302 | 1.760 | -0.07 |
| CatBoost | 1.078 | 1.476 | 0.41 | 1.174 | 1.529 | 0.08 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.107 | 3.227 | 1.894 | 0.924 | 0.17 | 16.5% |
| MinT | 0.934 | 1.470 | 1.053 | 0.879 | 0.06 | 5.9% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.025 | +0.040 | -0.002 | -0.043 | -0.050 |
| ECMWF | -0.000 | -0.011 | +0.051 | -0.033 | +0.030 |
| Lags | -0.000 | -0.005 | +0.040 | -0.007 | -0.005 |
| Rolling | +0.012 | +0.007 | +0.029 | -0.023 | -0.020 |
| NWP_atmosphere | +0.002 | -0.004 | +0.021 | -0.047 | +0.001 |
| Physics | -0.004 | +0.011 | +0.050 | -0.009 | +0.003 |
| Time | +0.003 | -0.011 | -0.013 | -0.060 | +0.034 |
| NWP_primary | -0.005 | -0.021 | +0.384 | +0.317 | +0.341 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.018 | +0.016 | -0.005 | -0.020 | -0.013 |
| ECMWF | +0.001 | +0.000 | -0.008 | +0.003 | -0.028 |
| Lags | -0.002 | -0.000 | -0.031 | +0.008 | -0.017 |
| Rolling | -0.006 | -0.001 | -0.022 | -0.026 | -0.000 |
| NWP_atmosphere | +0.009 | +0.007 | +0.000 | -0.018 | -0.013 |
| Physics | -0.001 | +0.007 | -0.011 | -0.025 | +0.070 |
| Time | -0.000 | +0.009 | -0.016 | -0.010 | +0.032 |
| NWP_primary | +0.350 | +0.365 | +0.383 | +0.319 | +0.263 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.923 | 0.975 | 0.814 | Winter | Fall |
| MinT | 0.797 | 0.934 | 0.985 | Summer | Winter |

#### Station Insights

- **MaxT**: Best MAE (0.924 F, Linear) is 14.2% above the Gulf/SC group mean (0.809 F).
- **MinT**: Best MAE (0.879 F, MLP) is 19.9% below the Gulf/SC group mean (1.097 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.203 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.336 F avg across architectures).

---

### 4.2 KMSY -- New Orleans, LA

**Climate zone**: Gulf/SC | **Coordinates**: 29.99 N, 90.26 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.924** | 1.106 | -0.26 | 1.400 | 1.725 | -0.85 |
| MLP | 0.945 | 1.139 | -0.36 | 1.414 | 1.742 | -0.79 |
| XGBoost | 1.025 | 1.229 | -0.31 | 1.353 | 1.684 | -0.69 |
| LightGBM | 1.204 | 1.965 | -0.13 | 1.579 | 2.095 | -0.60 |
| CatBoost | 1.205 | 1.503 | -0.47 | 1.496 | 1.852 | -0.65 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.196 | 1.532 | 0.20 | 1.389 | 1.816 | 0.03 |
| **MLP** | **1.183** | 1.542 | 0.22 | 1.415 | 1.869 | 0.13 |
| XGBoost | 1.408 | 1.879 | 0.32 | 1.620 | 2.031 | 0.00 |
| LightGBM | 1.501 | 2.198 | 0.45 | 1.625 | 2.148 | 0.08 |
| CatBoost | 1.388 | 1.804 | 0.20 | 1.538 | 1.984 | 0.01 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.613 | 4.022 | 2.745 | 0.924 | 0.43 | 42.7% |
| MinT | 1.303 | 2.080 | 1.539 | 1.183 | 0.09 | 9.2% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.068 | +0.053 | +0.065 | +0.050 | -0.035 |
| ECMWF | -0.012 | -0.005 | +0.016 | -0.033 | -0.063 |
| Lags | +0.000 | -0.000 | +0.019 | -0.009 | -0.034 |
| Rolling | -0.035 | +0.002 | -0.002 | -0.037 | -0.104 |
| NWP_atmosphere | +0.005 | -0.002 | +0.038 | -0.039 | -0.062 |
| Physics | -0.000 | -0.000 | -0.006 | -0.006 | -0.123 |
| Time | -0.005 | +0.005 | +0.011 | -0.035 | +0.004 |
| NWP_primary | +0.007 | +0.076 | +0.288 | +0.297 | +0.236 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.028 | +0.069 | -0.057 | +0.054 | -0.033 |
| ECMWF | +0.004 | +0.006 | -0.088 | -0.008 | -0.063 |
| Lags | -0.003 | +0.000 | -0.089 | +0.035 | -0.076 |
| Rolling | +0.019 | +0.013 | -0.079 | -0.014 | -0.083 |
| NWP_atmosphere | -0.008 | +0.015 | +0.003 | +0.032 | +0.031 |
| Physics | -0.031 | -0.001 | -0.030 | +0.009 | -0.031 |
| Time | +0.008 | +0.001 | -0.069 | +0.001 | -0.017 |
| NWP_primary | +0.352 | +0.418 | +0.290 | +0.282 | +0.194 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.924 | 0.987 | 0.784 | Winter | Fall |
| MinT | 1.059 | 1.208 | 1.470 | Summer | Winter |

#### Station Insights

- **MaxT**: Best MAE (0.924 F, Linear) is 14.1% above the Gulf/SC group mean (0.809 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.181 F avg across architectures).
- **MaxT ablation**: Removing Physics, Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.307 F avg across architectures).
- **MinT ablation**: Removing ECMWF, Lags, Rolling actually *improves* performance (possible overfitting from these features).
- **MinT seasonality**: Winter MAE (1.470 F) is 39% worse than Summer (1.059 F).

---

### 4.3 KDFW -- Dallas DFW, TX

**Climate zone**: Gulf/SC | **Coordinates**: 32.90 N, 97.04 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.758** | 0.962 | 0.08 | 0.914 | 1.191 | 0.17 |
| MLP | 0.761 | 0.961 | 0.02 | 0.912 | 1.189 | 0.11 |
| XGBoost | 0.858 | 1.222 | -0.22 | 1.083 | 1.449 | 0.22 |
| LightGBM | 1.081 | 2.081 | -0.08 | 1.175 | 1.828 | 0.27 |
| CatBoost | 1.042 | 1.418 | -0.22 | 1.187 | 1.572 | 0.25 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.909 | 1.223 | 0.43 | 1.111 | 1.516 | 0.42 |
| **MLP** | **0.908** | 1.215 | 0.41 | 1.144 | 1.534 | 0.35 |
| XGBoost | 1.083 | 1.434 | 0.57 | 1.116 | 1.527 | 0.32 |
| LightGBM | 1.143 | 1.778 | 0.60 | 1.176 | 1.672 | 0.43 |
| CatBoost | 1.061 | 1.399 | 0.40 | 1.156 | 1.571 | 0.29 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.135 | 2.682 | 1.852 | 0.758 | 0.33 | 33.2% |
| MinT | 1.335 | 1.503 | 1.313 | 0.908 | 0.32 | 32.0% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.001 | +0.021 | +0.114 | +0.058 | +0.081 |
| ECMWF | +0.000 | +0.006 | +0.003 | -0.045 | +0.006 |
| Lags | -0.000 | -0.000 | +0.076 | -0.023 | -0.038 |
| Rolling | -0.013 | +0.003 | -0.018 | -0.076 | -0.086 |
| NWP_atmosphere | +0.015 | +0.001 | +0.057 | -0.043 | +0.002 |
| Physics | +0.028 | +0.001 | +0.043 | -0.048 | -0.085 |
| Time | -0.000 | +0.006 | +0.059 | -0.027 | +0.053 |
| NWP_primary | +0.005 | +0.188 | +0.634 | +0.523 | +0.421 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.055 | +0.054 | +0.082 | +0.093 | +0.042 |
| ECMWF | +0.002 | -0.002 | -0.057 | +0.026 | -0.020 |
| Lags | -0.000 | +0.003 | -0.015 | -0.019 | +0.084 |
| Rolling | +0.000 | +0.006 | -0.004 | +0.005 | -0.027 |
| NWP_atmosphere | +0.027 | +0.023 | +0.006 | +0.028 | -0.050 |
| Physics | +0.003 | +0.007 | -0.007 | +0.025 | -0.017 |
| Time | +0.004 | -0.004 | -0.043 | +0.009 | +0.045 |
| NWP_primary | +0.358 | +0.344 | +0.392 | +0.381 | +0.418 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.761 | 0.637 | 1.018 | Fall | Winter |
| MinT | 0.799 | 0.931 | 1.104 | Summer | Winter |

#### Station Insights

- **MinT**: Best MAE (0.908 F, MLP) is 17.2% below the Gulf/SC group mean (1.097 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.354 F avg across architectures).
- **MaxT ablation**: Removing Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.379 F avg across architectures).
- **MaxT seasonality**: Winter MAE (1.018 F) is 60% worse than Fall (0.637 F).
- **MinT seasonality**: Winter MAE (1.104 F) is 38% worse than Summer (0.799 F).

---

### 4.4 KDAL -- Dallas Love, TX

**Climate zone**: Gulf/SC | **Coordinates**: 32.85 N, 96.85 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.805** | 1.046 | -0.25 | 0.903 | 1.176 | -0.16 |
| MLP | 0.813 | 1.067 | -0.31 | 0.912 | 1.190 | -0.22 |
| XGBoost | 0.919 | 1.245 | -0.30 | 1.049 | 1.376 | -0.24 |
| LightGBM | 1.131 | 2.117 | -0.19 | 1.122 | 1.721 | -0.19 |
| CatBoost | 0.994 | 1.319 | -0.27 | 1.178 | 1.611 | -0.11 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.944** | 1.291 | 0.31 | 1.198 | 1.620 | 0.04 |
| MLP | 0.944 | 1.271 | 0.28 | 1.267 | 1.705 | 0.04 |
| XGBoost | 1.135 | 1.574 | 0.39 | 1.321 | 1.711 | -0.16 |
| LightGBM | 1.225 | 1.962 | 0.53 | 1.340 | 1.867 | 0.05 |
| CatBoost | 1.188 | 1.617 | 0.37 | 1.430 | 1.887 | -0.19 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.813 | 1.866 | 1.110 | 0.805 | 0.01 | 1.0% |
| MinT | 1.033 | 1.661 | 1.184 | 0.944 | 0.09 | 8.7% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.028 | +0.027 | +0.019 | +0.014 | +0.073 |
| ECMWF | +0.011 | +0.005 | +0.030 | -0.022 | +0.113 |
| Lags | +0.001 | -0.007 | +0.018 | +0.013 | +0.025 |
| Rolling | -0.005 | -0.001 | +0.007 | -0.006 | +0.012 |
| NWP_atmosphere | +0.015 | +0.008 | +0.001 | -0.009 | +0.123 |
| Physics | +0.014 | +0.012 | +0.007 | -0.003 | +0.069 |
| Time | -0.000 | +0.000 | +0.063 | -0.022 | +0.079 |
| NWP_primary | -0.003 | +0.015 | +0.418 | +0.369 | +0.295 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.030 | +0.005 | -0.014 | +0.026 | -0.044 |
| ECMWF | -0.000 | -0.001 | -0.031 | +0.000 | -0.004 |
| Lags | +0.001 | -0.000 | -0.015 | +0.037 | +0.005 |
| Rolling | +0.006 | +0.002 | -0.011 | -0.013 | -0.039 |
| NWP_atmosphere | -0.011 | -0.002 | +0.013 | +0.036 | +0.002 |
| Physics | -0.003 | -0.019 | +0.030 | +0.027 | +0.005 |
| Time | +0.009 | +0.007 | +0.039 | +0.034 | +0.018 |
| NWP_primary | +0.436 | +0.419 | +0.359 | +0.415 | +0.346 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.835 | 0.727 | 0.907 | Fall | Winter |
| MinT | 0.800 | 0.940 | 1.268 | Summer | Winter |

#### Station Insights

- **MinT**: Best MAE (0.944 F, Linear) is 13.9% below the Gulf/SC group mean (1.097 F).
- MaxT shows minimal improvement (1.0%) over raw GFS -- GFS already performs well here.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.219 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.395 F avg across architectures).
- **MinT seasonality**: Winter MAE (1.268 F) is 59% worse than Summer (0.800 F).

---

### 4.5 KAUS -- Austin, TX

**Climate zone**: Gulf/SC | **Coordinates**: 30.21 N, 97.68 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.717** | 0.952 | -0.06 | 1.270 | 1.783 | -0.08 |
| MLP | 0.749 | 0.986 | -0.11 | 1.246 | 1.750 | -0.13 |
| XGBoost | 0.852 | 1.198 | -0.16 | 1.315 | 1.887 | -0.13 |
| LightGBM | 0.922 | 1.603 | -0.05 | 1.334 | 1.983 | -0.07 |
| CatBoost | 1.053 | 1.507 | -0.25 | 1.329 | 1.815 | -0.16 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.531 | 2.036 | -0.07 | 1.821 | 2.405 | -0.18 |
| **MLP** | **1.507** | 2.012 | 0.06 | 1.830 | 2.401 | -0.13 |
| XGBoost | 1.719 | 2.311 | -0.18 | 1.906 | 2.506 | -0.43 |
| LightGBM | 1.690 | 2.356 | -0.28 | 1.883 | 2.447 | -0.38 |
| CatBoost | 1.783 | 2.374 | -0.39 | 2.016 | 2.628 | -0.53 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.701 | 2.681 | 2.172 | 0.717 | 0.58 | 57.8% |
| MinT | 3.747 | 4.654 | 4.149 | 1.507 | 0.60 | 59.8% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.024 | -0.020 | +0.025 | -0.009 | -0.055 |
| ECMWF | +0.000 | +0.003 | +0.030 | +0.027 | -0.069 |
| Lags | +0.000 | -0.008 | +0.012 | +0.001 | -0.063 |
| Rolling | +0.022 | -0.001 | +0.001 | -0.054 | -0.079 |
| NWP_atmosphere | +0.035 | +0.001 | +0.038 | +0.008 | -0.051 |
| Physics | +0.010 | -0.001 | +0.017 | -0.011 | -0.079 |
| Time | -0.000 | -0.013 | +0.024 | -0.009 | -0.043 |
| NWP_primary | +0.011 | +0.193 | +0.556 | +0.573 | +0.477 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.036 | +0.036 | +0.024 | +0.024 | -0.014 |
| ECMWF | +0.002 | -0.006 | -0.009 | +0.017 | -0.024 |
| Lags | +0.002 | +0.000 | +0.004 | +0.010 | -0.015 |
| Rolling | +0.042 | +0.004 | -0.000 | +0.069 | +0.002 |
| NWP_atmosphere | +0.039 | +0.023 | -0.054 | +0.015 | +0.137 |
| Physics | -0.020 | +0.018 | -0.045 | -0.033 | +0.043 |
| Time | -0.000 | +0.026 | -0.063 | -0.003 | -0.048 |
| NWP_primary | +0.578 | +0.586 | +0.643 | +0.578 | +0.450 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.817 | 0.580 | 0.812 | Fall | Summer |
| MinT | 0.946 | 1.732 | 2.303 | Summer | Winter |

#### Station Insights

- **MaxT**: Best MAE (0.717 F, Linear) is 11.4% below the Gulf/SC group mean (0.809 F).
- **MinT**: Best MAE (1.507 F, MLP) is 37.5% above the Gulf/SC group mean (1.097 F).
- MaxT post-processing delivers exceptional 57.8% improvement over raw GFS.
- MinT post-processing delivers exceptional 59.8% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.362 F avg across architectures).
- **MaxT ablation**: Removing Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.567 F avg across architectures).
- **MaxT seasonality**: Summer MAE (0.817 F) is 41% worse than Fall (0.580 F).
- **MinT seasonality**: Winter MAE (2.303 F) is 143% worse than Summer (0.946 F).

---

### 4.6 KSAT -- San Antonio, TX

**Climate zone**: Gulf/SC | **Coordinates**: 29.53 N, 98.47 W | **Timezone**: America/Chicago

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.743 | 0.962 | 0.12 | 1.263 | 1.690 | 0.22 |
| **MLP** | **0.728** | 0.951 | 0.08 | 1.251 | 1.690 | 0.19 |
| XGBoost | 0.889 | 1.319 | 0.12 | 1.344 | 1.910 | 0.32 |
| LightGBM | 0.923 | 1.560 | 0.15 | 1.344 | 1.899 | 0.28 |
| CatBoost | 1.118 | 1.509 | 0.08 | 1.308 | 1.798 | 0.20 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.159** | 1.690 | 0.37 | 1.446 | 2.004 | -0.02 |
| MLP | 1.176 | 1.695 | 0.40 | 1.445 | 1.998 | -0.09 |
| XGBoost | 1.215 | 1.743 | 0.36 | 1.550 | 2.078 | -0.25 |
| LightGBM | 1.340 | 2.011 | 0.50 | 1.563 | 2.142 | -0.16 |
| CatBoost | 1.371 | 1.924 | 0.35 | 1.514 | 2.065 | -0.17 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.829 | 2.414 | 1.483 | 0.728 | 0.12 | 12.3% |
| MinT | 1.352 | 2.159 | 1.378 | 1.159 | 0.14 | 14.3% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.000 | -0.013 | +0.001 | -0.031 | -0.114 |
| ECMWF | -0.001 | -0.008 | +0.011 | +0.016 | -0.100 |
| Lags | +0.000 | -0.008 | -0.009 | -0.022 | -0.154 |
| Rolling | +0.002 | -0.008 | +0.021 | -0.024 | -0.100 |
| NWP_atmosphere | +0.000 | +0.003 | +0.039 | -0.035 | -0.094 |
| Physics | -0.002 | +0.000 | -0.023 | -0.025 | -0.113 |
| Time | -0.005 | +0.003 | +0.034 | -0.055 | -0.081 |
| NWP_primary | +0.000 | +0.066 | +0.402 | +0.427 | +0.282 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.061 | +0.044 | +0.017 | +0.006 | +0.012 |
| ECMWF | -0.000 | +0.000 | +0.043 | -0.008 | +0.033 |
| Lags | -0.002 | +0.001 | -0.009 | -0.041 | -0.006 |
| Rolling | -0.009 | +0.008 | +0.019 | -0.016 | +0.010 |
| NWP_atmosphere | -0.022 | -0.039 | -0.035 | +0.005 | -0.102 |
| Physics | -0.022 | -0.026 | -0.031 | -0.001 | -0.047 |
| Time | -0.007 | -0.043 | +0.013 | -0.011 | -0.080 |
| NWP_primary | +0.543 | +0.520 | +0.609 | +0.555 | +0.420 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.876 | 0.582 | 0.806 | Fall | Summer |
| MinT | 0.845 | 1.389 | 1.341 | Summer | Fall |

#### Station Insights

- **MaxT**: Best MAE (0.728 F, MLP) is 10.1% below the Gulf/SC group mean (0.809 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.235 F avg across architectures).
- **MaxT ablation**: Removing Bias, Lags, Physics, Rolling, Time actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.530 F avg across architectures).
- **MinT ablation**: Removing NWP_atmosphere, Physics, Time actually *improves* performance (possible overfitting from these features).
- **MaxT seasonality**: Summer MAE (0.876 F) is 51% worse than Fall (0.582 F).
- **MinT seasonality**: Fall MAE (1.389 F) is 64% worse than Summer (0.845 F).

---

## 5. Pacific (3 stations)

Stations: KLAX, KSFO, KSEA

### 5.1 KLAX -- Los Angeles, CA

**Climate zone**: Pacific | **Coordinates**: 33.94 N, 118.39 W | **Timezone**: America/Los_Angeles

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.876 | 1.746 | 0.20 | 1.729 | 2.400 | -0.54 |
| **MLP** | **0.874** | 1.726 | 0.19 | 1.797 | 2.514 | -0.60 |
| XGBoost | 1.000 | 1.773 | 0.36 | 1.709 | 2.300 | -0.29 |
| LightGBM | 1.003 | 1.795 | 0.29 | 1.740 | 2.373 | -0.30 |
| CatBoost | 0.969 | 1.736 | 0.25 | 1.720 | 2.337 | -0.34 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 1.056 | 1.355 | -0.23 | 1.169 | 1.583 | -0.30 |
| MLP | 1.076 | 1.376 | -0.27 | 1.232 | 1.634 | -0.20 |
| XGBoost | 1.061 | 1.393 | -0.30 | 1.196 | 1.609 | -0.27 |
| LightGBM | 1.112 | 1.429 | -0.45 | 1.213 | 1.621 | -0.27 |
| **CatBoost** | **1.047** | 1.353 | -0.18 | 1.163 | 1.582 | -0.25 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.024 | 6.218 | 3.056 | 0.874 | 0.15 | 14.6% |
| MinT | 2.081 | 2.383 | 1.968 | 1.047 | 0.50 | 49.7% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.004 | -0.004 | +0.003 | -0.008 | +0.053 |
| ECMWF | +0.006 | +0.017 | -0.020 | -0.036 | +0.019 |
| Lags | -0.005 | +0.001 | -0.016 | -0.041 | +0.033 |
| Rolling | +0.015 | +0.007 | -0.044 | -0.002 | -0.002 |
| NWP_atmosphere | +0.005 | -0.004 | -0.016 | +0.016 | -0.007 |
| Physics | +0.008 | +0.007 | -0.032 | -0.039 | -0.013 |
| Time | +0.004 | +0.010 | -0.015 | -0.013 | +0.031 |
| NWP_primary | +0.308 | +0.323 | +0.815 | +0.900 | +0.742 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.035 | -0.047 | +0.003 | -0.024 | -0.040 |
| ECMWF | +0.009 | -0.001 | +0.035 | +0.030 | +0.003 |
| Lags | +0.020 | -0.003 | +0.045 | +0.042 | +0.042 |
| Rolling | -0.021 | +0.021 | +0.014 | +0.017 | +0.013 |
| NWP_atmosphere | +0.034 | +0.000 | +0.021 | -0.002 | -0.002 |
| Physics | +0.048 | -0.008 | +0.030 | -0.021 | +0.011 |
| Time | +0.000 | -0.005 | +0.043 | +0.004 | +0.027 |
| NWP_primary | +0.265 | +0.304 | +0.259 | +0.266 | +0.375 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.623 | 0.835 | 1.538 | Summer | Winter |
| MinT | 0.869 | 1.192 | 1.170 | Summer | Fall |

#### Station Insights

- **MaxT**: Best MAE (0.874 F, MLP) is 12.9% below the Pacific group mean (1.003 F).
- **MinT**: Best MAE (1.047 F, CatBoost) is 12.8% above the Pacific group mean (0.928 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.618 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.294 F avg across architectures).
- **MinT ablation**: Removing Bias actually *improves* performance (possible overfitting from these features).
- **MaxT seasonality**: Winter MAE (1.538 F) is 147% worse than Summer (0.623 F).
- **MinT seasonality**: Fall MAE (1.192 F) is 37% worse than Summer (0.869 F).

---

### 5.2 KSFO -- San Francisco, CA

**Climate zone**: Pacific | **Coordinates**: 37.62 N, 122.38 W | **Timezone**: America/Los_Angeles

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **1.305** | 1.628 | 0.04 | 1.438 | 1.897 | -0.08 |
| MLP | 1.330 | 1.653 | 0.12 | 1.439 | 1.893 | -0.05 |
| XGBoost | 1.442 | 1.790 | 0.26 | 1.594 | 1.991 | 0.25 |
| LightGBM | 1.447 | 1.747 | 0.25 | 1.607 | 2.004 | 0.24 |
| CatBoost | 1.441 | 1.779 | 0.35 | 1.523 | 1.972 | 0.36 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.950 | 1.308 | 0.10 | 1.014 | 1.300 | -0.18 |
| MLP | 0.931 | 1.263 | 0.21 | 1.024 | 1.321 | -0.25 |
| XGBoost | 0.951 | 1.295 | 0.35 | 1.094 | 1.420 | -0.38 |
| **LightGBM** | **0.929** | 1.268 | 0.31 | 1.119 | 1.448 | -0.40 |
| CatBoost | 0.978 | 1.294 | 0.23 | 1.062 | 1.388 | -0.41 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.597 | 2.788 | 1.782 | 1.305 | 0.18 | 18.3% |
| MinT | 2.132 | 1.475 | 1.330 | 0.929 | 0.56 | 56.4% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.118 | +0.111 | +0.060 | +0.120 | +0.022 |
| ECMWF | -0.017 | -0.007 | -0.009 | +0.012 | -0.042 |
| Lags | +0.006 | -0.003 | -0.033 | +0.026 | -0.080 |
| Rolling | -0.006 | -0.002 | -0.012 | +0.011 | -0.037 |
| NWP_atmosphere | +0.005 | +0.003 | -0.028 | +0.001 | -0.066 |
| Physics | +0.075 | +0.066 | +0.034 | +0.026 | -0.040 |
| Time | +0.022 | +0.013 | +0.017 | +0.023 | -0.130 |
| NWP_primary | +0.024 | +0.063 | +0.201 | +0.203 | +0.104 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.126 | +0.114 | +0.078 | +0.065 | +0.013 |
| ECMWF | -0.000 | -0.000 | -0.038 | -0.007 | -0.036 |
| Lags | -0.004 | -0.006 | -0.019 | -0.004 | -0.006 |
| Rolling | +0.008 | +0.005 | -0.030 | +0.039 | -0.020 |
| NWP_atmosphere | -0.033 | -0.012 | -0.004 | +0.005 | +0.001 |
| Physics | -0.016 | +0.003 | +0.008 | +0.005 | +0.007 |
| Time | -0.000 | +0.005 | -0.016 | +0.012 | -0.032 |
| NWP_primary | +0.391 | +0.347 | +0.316 | +0.286 | +0.192 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 1.343 | 1.268 | 1.300 | Fall | Summer |
| MinT | 0.733 | 1.137 | 1.014 | Summer | Fall |

#### Station Insights

- **MaxT**: Best MAE (1.305 F, Linear) is 30.0% above the Pacific group mean (1.003 F).
- MinT post-processing delivers exceptional 56.4% improvement over raw GFS.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.119 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.306 F avg across architectures).
- **MinT seasonality**: Fall MAE (1.137 F) is 55% worse than Summer (0.733 F).

---

### 5.3 KSEA -- Seattle, WA

**Climate zone**: Pacific | **Coordinates**: 47.45 N, 122.31 W | **Timezone**: America/Los_Angeles

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.857 | 1.051 | 0.29 | 1.275 | 1.563 | -0.16 |
| **MLP** | **0.832** | 1.044 | 0.22 | 1.261 | 1.569 | -0.09 |
| XGBoost | 0.881 | 1.101 | 0.16 | 1.314 | 1.632 | -0.15 |
| LightGBM | 0.920 | 1.142 | 0.11 | 1.294 | 1.636 | -0.13 |
| CatBoost | 1.042 | 1.324 | 0.03 | 1.454 | 1.832 | -0.24 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.809** | 1.048 | 0.14 | 1.144 | 1.464 | -0.05 |
| MLP | 0.811 | 1.047 | 0.08 | 1.159 | 1.487 | -0.01 |
| XGBoost | 0.905 | 1.173 | 0.11 | 1.224 | 1.560 | -0.14 |
| LightGBM | 0.911 | 1.229 | 0.15 | 1.216 | 1.576 | -0.12 |
| CatBoost | 0.927 | 1.183 | 0.21 | 1.255 | 1.581 | -0.11 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.557 | 2.292 | 1.869 | 0.832 | 0.47 | 46.6% |
| MinT | 0.990 | 1.174 | 0.945 | 0.809 | 0.18 | 18.3% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.030 | -0.028 | -0.014 | -0.039 | +0.034 |
| ECMWF | -0.002 | -0.001 | +0.005 | -0.034 | +0.044 |
| Lags | -0.002 | -0.000 | +0.002 | +0.006 | +0.021 |
| Rolling | +0.012 | +0.004 | +0.012 | -0.016 | +0.035 |
| NWP_atmosphere | -0.013 | -0.001 | +0.011 | -0.032 | +0.024 |
| Physics | +0.000 | +0.007 | +0.017 | +0.005 | +0.046 |
| Time | +0.002 | +0.006 | -0.001 | -0.039 | +0.064 |
| NWP_primary | +0.003 | +0.128 | +0.469 | +0.433 | +0.415 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.014 | +0.013 | -0.014 | -0.008 | +0.012 |
| ECMWF | +0.000 | -0.000 | +0.009 | +0.007 | +0.033 |
| Lags | +0.002 | -0.000 | +0.005 | +0.003 | -0.011 |
| Rolling | +0.013 | +0.021 | +0.004 | +0.002 | +0.035 |
| NWP_atmosphere | -0.002 | +0.008 | -0.008 | -0.000 | +0.009 |
| Physics | +0.001 | +0.012 | +0.007 | +0.002 | +0.004 |
| Time | +0.002 | +0.003 | +0.003 | +0.006 | +0.002 |
| NWP_primary | +0.408 | +0.328 | +0.415 | +0.417 | +0.295 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.908 | 0.821 | 0.826 | Fall | Summer |
| MinT | 0.833 | 0.757 | 0.871 | Fall | Winter |

#### Station Insights

- **MaxT**: Best MAE (0.832 F, MLP) is 17.1% below the Pacific group mean (1.003 F).
- **MinT**: Best MAE (0.809 F, Linear) is 12.8% below the Pacific group mean (0.928 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.289 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.373 F avg across architectures).

---

## 6. Arid (2 stations)

Stations: KPHX, KLAS

### 6.1 KPHX -- Phoenix, AZ

**Climate zone**: Arid | **Coordinates**: 33.44 N, 112.01 W | **Timezone**: America/Phoenix

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| Linear | 0.750 | 0.976 | 0.00 | 0.999 | 1.262 | 0.15 |
| **MLP** | **0.744** | 0.966 | -0.04 | 0.996 | 1.265 | 0.11 |
| XGBoost | 0.832 | 1.118 | 0.26 | 1.157 | 1.452 | 0.11 |
| LightGBM | 0.803 | 1.050 | 0.15 | 1.178 | 1.497 | 0.15 |
| CatBoost | 1.158 | 1.495 | -0.15 | 1.253 | 1.603 | 0.05 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.856** | 1.154 | 0.10 | 1.505 | 2.135 | -0.35 |
| MLP | 0.864 | 1.162 | 0.05 | 1.541 | 2.165 | -0.46 |
| XGBoost | 0.915 | 1.221 | 0.25 | 1.614 | 2.245 | -0.74 |
| LightGBM | 0.945 | 1.252 | 0.28 | 1.725 | 2.316 | -0.80 |
| CatBoost | 1.088 | 1.479 | 0.16 | 1.767 | 2.431 | -0.74 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 1.463 | 2.607 | 2.010 | 0.744 | 0.49 | 49.2% |
| MinT | 1.373 | 1.884 | 1.456 | 0.856 | 0.38 | 37.6% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.046 | +0.093 | +0.057 | +0.054 | -0.068 |
| ECMWF | -0.001 | -0.001 | +0.030 | +0.038 | +0.020 |
| Lags | +0.000 | -0.000 | +0.008 | +0.010 | -0.186 |
| Rolling | +0.012 | -0.012 | -0.012 | -0.020 | -0.148 |
| NWP_atmosphere | +0.001 | -0.005 | -0.004 | -0.013 | -0.117 |
| Physics | +0.017 | +0.016 | +0.003 | +0.003 | -0.081 |
| Time | -0.003 | -0.002 | +0.023 | -0.019 | -0.049 |
| NWP_primary | +0.017 | +0.104 | +0.265 | +0.312 | +0.247 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | +0.028 | +0.024 | +0.062 | +0.034 | -0.033 |
| ECMWF | -0.002 | +0.001 | -0.002 | +0.026 | -0.001 |
| Lags | -0.000 | +0.000 | +0.014 | +0.033 | +0.002 |
| Rolling | -0.002 | -0.012 | +0.039 | -0.000 | +0.024 |
| NWP_atmosphere | +0.026 | +0.006 | +0.037 | -0.010 | +0.004 |
| Physics | +0.014 | -0.006 | +0.027 | -0.016 | +0.001 |
| Time | -0.000 | +0.000 | +0.036 | +0.002 | -0.026 |
| NWP_primary | +1.027 | +1.089 | +1.178 | +1.214 | +0.533 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.674 | 0.756 | 0.901 | Summer | Winter |
| MinT | 0.839 | 0.851 | 0.905 | Summer | Winter |

#### Station Insights

- **MaxT**: Best MAE (0.744 F, MLP) is 14.5% above the Arid group mean (0.649 F).
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.189 F avg across architectures).
- **MaxT ablation**: Removing Lags, NWP_atmosphere, Rolling actually *improves* performance (possible overfitting from these features).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+1.008 F avg across architectures).
- **MaxT seasonality**: Winter MAE (0.901 F) is 34% worse than Summer (0.674 F).

---

### 6.2 KLAS -- Las Vegas, NV

**Climate zone**: Arid | **Coordinates**: 36.08 N, 115.15 W | **Timezone**: America/Los_Angeles

#### Model Performance (All Architectures, Both Datasets)

**MaxT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.555** | 0.701 | 0.01 | 1.069 | 1.503 | 0.17 |
| MLP | 0.566 | 0.708 | 0.05 | 1.058 | 1.486 | 0.16 |
| XGBoost | 0.611 | 0.808 | -0.10 | 1.045 | 1.506 | 0.18 |
| LightGBM | 0.626 | 0.818 | -0.12 | 1.120 | 1.630 | 0.22 |
| CatBoost | 0.819 | 1.037 | -0.01 | 1.232 | 1.653 | -0.08 |

**MinT**

| Architecture | HF MAE | HF RMSE | HF Bias | ERA5 MAE | ERA5 RMSE | ERA5 Bias |
| --- | --- | --- | --- | --- | --- | --- |
| **Linear** | **0.705** | 0.921 | 0.02 | 1.160 | 1.461 | -0.07 |
| MLP | 0.706 | 0.921 | 0.01 | 1.234 | 1.536 | -0.20 |
| XGBoost | 0.890 | 1.181 | 0.18 | 1.294 | 1.661 | -0.18 |
| LightGBM | 0.855 | 1.093 | 0.25 | 1.339 | 1.688 | -0.21 |
| CatBoost | 0.991 | 1.273 | 0.05 | 1.282 | 1.612 | -0.37 |

#### GFS Deviation Analysis

| Target | GFS MAE | ECMWF MAE | Blend MAE | Model MAE | Skill vs GFS | Improvement % |
| --- | --- | --- | --- | --- | --- | --- |
| MaxT | 0.551 | 2.038 | 1.191 | 0.555 | -0.01 | -0.7% |
| MinT | 1.280 | 2.752 | 1.354 | 0.705 | 0.45 | 44.9% |

#### Feature Ablation

**MaxT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.067 | -0.070 | +0.102 | +0.030 | +0.052 |
| ECMWF | +0.002 | -0.010 | +0.050 | +0.029 | +0.006 |
| Lags | +0.001 | +0.000 | +0.040 | +0.017 | +0.005 |
| Rolling | -0.020 | -0.012 | +0.015 | +0.030 | -0.060 |
| NWP_atmosphere | -0.018 | -0.015 | +0.041 | -0.017 | +0.055 |
| Physics | +0.011 | -0.003 | +0.018 | +0.063 | +0.007 |
| Time | +0.001 | -0.004 | +0.044 | +0.018 | +0.122 |
| NWP_primary | +0.011 | +0.170 | +0.610 | +0.529 | +0.306 |

**MinT** -- Mean delta-MAE when feature group is removed (positive = group is helpful)

| Feature Group | Linear | MLP | XGBoost | LightGBM | CatBoost |
| --- | --- | --- | --- | --- | --- |
| Bias | -0.023 | -0.023 | +0.114 | +0.003 | +0.057 |
| ECMWF | +0.005 | +0.006 | -0.031 | +0.023 | -0.074 |
| Lags | +0.003 | -0.000 | +0.009 | -0.032 | +0.023 |
| Rolling | -0.008 | -0.008 | +0.042 | -0.013 | +0.014 |
| NWP_atmosphere | +0.013 | +0.014 | -0.013 | -0.002 | +0.027 |
| Physics | +0.011 | -0.002 | +0.020 | -0.032 | +0.023 |
| Time | +0.017 | +0.013 | -0.048 | +0.006 | +0.043 |
| NWP_primary | +0.828 | +0.898 | +0.993 | +0.954 | +0.559 |

#### Seasonal Performance

| Target | Summer MAE | Fall MAE | Winter MAE | Best Season | Worst Season |
| --- | --- | --- | --- | --- | --- |
| MaxT | 0.526 | 0.609 | 0.501 | Winter | Fall |
| MinT | 0.746 | 0.657 | 0.722 | Fall | Summer |

#### Station Insights

- **MaxT**: Best MAE (0.555 F, Linear) is 14.5% below the Arid group mean (0.649 F).
- MaxT shows minimal improvement (-0.7%) over raw GFS -- GFS already performs well here.
- **MaxT ablation**: Removing **NWP_primary** causes the largest degradation (+0.325 F avg across architectures).
- **MinT ablation**: Removing **NWP_primary** causes the largest degradation (+0.847 F avg across architectures).

---

## Cross-Cutting Insights

### Architecture Recommendations by Climate Zone

| Climate Zone | Best MaxT Arch | Best MaxT MAE | Best MinT Arch | Best MinT MAE |
| --- | --- | --- | --- | --- |
| Continental | MLP | 0.908 | MLP | 1.149 |
| NE Coastal | MLP | 0.909 | Linear | 1.162 |
| SE Subtropical | Linear | 0.868 | MLP | 0.988 |
| Gulf/SC | Linear | 0.812 | MLP | 1.100 |
| Pacific | MLP | 1.012 | Linear | 0.938 |
| Arid | Linear | 0.652 | Linear | 0.781 |

### Historical Forecast vs. ERA5 Advantage by Climate Zone

| Climate Zone | MaxT HF MAE | MaxT ERA5 MAE | MaxT Advantage | MinT HF MAE | MinT ERA5 MAE | MinT Advantage |
| --- | --- | --- | --- | --- | --- | --- |
| Continental | 0.904 | 1.107 | 18.4% | 1.147 | 1.422 | 19.3% |
| NE Coastal | 0.908 | 1.155 | 21.4% | 1.157 | 1.279 | 9.5% |
| SE Subtropical | 0.861 | 1.201 | 28.3% | 0.982 | 1.227 | 19.9% |
| Gulf/SC | 0.809 | 1.121 | 27.8% | 1.097 | 1.343 | 18.3% |
| Pacific | 1.003 | 1.469 | 31.7% | 0.928 | 1.107 | 16.1% |
| Arid | 0.649 | 1.021 | 36.4% | 0.781 | 1.333 | 41.4% |

### GFS Post-Processing Improvement by Climate Zone

| Climate Zone | MaxT Avg Improvement | MaxT Range | MinT Avg Improvement | MinT Range |
| --- | --- | --- | --- | --- |
| Continental | 11.4% | -12.8% -- 44.4% | 24.9% | 4.7% -- 44.4% |
| NE Coastal | 19.4% | -0.1% -- 43.4% | 42.9% | 28.6% -- 55.3% |
| SE Subtropical | 54.9% | 39.7% -- 69.0% | 27.5% | 3.1% -- 55.2% |
| Gulf/SC | 27.2% | 1.0% -- 57.8% | 21.7% | 5.9% -- 59.8% |
| Pacific | 26.5% | 14.6% -- 46.6% | 41.5% | 18.3% -- 56.4% |
| Arid | 24.2% | -0.7% -- 49.2% | 41.2% | 37.6% -- 44.9% |

### Feature Group Importance by Climate Zone

Mean delta-MAE when each group is removed, averaged across architectures and stations within each climate zone.

**MaxT**

| Climate Zone | Bias | ECMWF | Lags | Rolling | NWP_atmosphere | Physics | Time | NWP_primary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continental | +0.041 | +0.001 | +0.003 | -0.006 | -0.001 | -0.007 | -0.000 | +0.355 |
| NE Coastal | +0.029 | +0.011 | +0.000 | -0.015 | +0.006 | -0.005 | +0.008 | +0.282 |
| SE Subtropical | +0.024 | -0.005 | -0.002 | -0.013 | +0.006 | +0.001 | -0.000 | +0.195 |
| Gulf/SC | +0.012 | -0.001 | -0.006 | -0.019 | +0.001 | -0.009 | -0.000 | +0.259 |
| Pacific | +0.027 | -0.004 | -0.006 | -0.002 | -0.007 | +0.011 | -0.000 | +0.342 |
| Arid | +0.023 | +0.016 | -0.010 | -0.023 | -0.009 | +0.005 | +0.013 | +0.257 |

**MinT**

| Climate Zone | Bias | ECMWF | Lags | Rolling | NWP_atmosphere | Physics | Time | NWP_primary |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continental | +0.035 | +0.006 | -0.001 | +0.002 | +0.004 | +0.015 | +0.006 | +0.537 |
| NE Coastal | +0.001 | -0.003 | -0.006 | +0.002 | +0.019 | -0.004 | -0.008 | +0.335 |
| SE Subtropical | +0.030 | -0.006 | -0.010 | -0.008 | +0.003 | -0.008 | -0.006 | +0.394 |
| Gulf/SC | +0.021 | -0.007 | -0.005 | -0.005 | +0.003 | -0.005 | -0.005 | +0.419 |
| Pacific | +0.018 | +0.003 | +0.007 | +0.008 | +0.001 | +0.006 | +0.004 | +0.324 |
| Arid | +0.024 | -0.005 | +0.005 | +0.007 | +0.010 | +0.004 | +0.004 | +0.927 |

### Seasonal Vulnerability by Climate Zone

Mean MAE by season for each climate zone (Linear model, 3yr HF).

| Climate Zone | Target | Summer | Fall | Winter | Worst/Best Ratio |
| --- | --- | --- | --- | --- | --- |
| Continental | MaxT | 0.977 | 0.826 | 1.063 | 1.29 |
| Continental | MinT | 0.972 | 1.298 | 1.247 | 1.34 |
| NE Coastal | MaxT | 0.943 | 0.852 | 1.017 | 1.19 |
| NE Coastal | MinT | 1.008 | 1.158 | 1.511 | 1.50 |
| SE Subtropical | MaxT | 0.955 | 0.811 | 0.803 | 1.19 |
| SE Subtropical | MinT | 0.830 | 0.975 | 1.363 | 1.64 |
| Gulf/SC | MaxT | 0.856 | 0.748 | 0.857 | 1.15 |
| Gulf/SC | MinT | 0.874 | 1.189 | 1.412 | 1.62 |
| Pacific | MaxT | 0.958 | 0.974 | 1.219 | 1.27 |
| Pacific | MinT | 0.812 | 1.029 | 1.017 | 1.27 |
| Arid | MaxT | 0.600 | 0.683 | 0.701 | 1.17 |
| Arid | MinT | 0.792 | 0.754 | 0.813 | 1.08 |

### Hardest and Easiest Stations to Predict

**MaxT -- Easiest (lowest MAE)**

1. KLAS (Las Vegas, NV): 0.555 F
1. KOKC (Oklahoma City, OK): 0.716 F
1. KAUS (Austin, TX): 0.717 F
1. KSAT (San Antonio, TX): 0.728 F
1. KPHX (Phoenix, AZ): 0.744 F

**MaxT -- Hardest (highest MAE)**

1. KDEN (Denver, CO): 1.445 F
1. KSFO (San Francisco, CA): 1.305 F
1. KDCA (Washington D.C.): 1.139 F
1. KTPA (Tampa, FL): 1.033 F
1. KBOS (Boston, MA): 0.927 F

**MinT -- Easiest (lowest MAE)**

1. KLAS (Las Vegas, NV): 0.705 F
1. KSEA (Seattle, WA): 0.809 F
1. KATL (Atlanta, GA): 0.834 F
1. KPHX (Phoenix, AZ): 0.856 F
1. KHOU (Houston, TX): 0.879 F

**MinT -- Hardest (highest MAE)**

1. KDEN (Denver, CO): 1.665 F
1. KAUS (Austin, TX): 1.507 F
1. KLGA (LaGuardia, NY): 1.278 F
1. KBOS (Boston, MA): 1.253 F
1. KNYC (New York Central Park, NY): 1.248 F
