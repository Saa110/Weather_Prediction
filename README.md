# Historical Forecasts vs. Reanalysis as Training Data for ML Temperature Post-Processing

> **A Multi-Station, Multi-Architecture Comparison**

This repository contains the code, experiment configurations, and pre-computed
results needed to reproduce all tables and figures from the paper:

**Asad, S. A., Agarwal, S., Qasim, S. M., Khan, A. T., & Shah, K. (2026).**
*Historical Forecasts vs. Reanalysis as Training Data for Machine Learning
Temperature Post-Processing: A Multi-Station, Multi-Architecture Comparison.*

---

## Abstract

Model Output Statistics (MOS) has been the standard approach for post-processing
numerical weather prediction (NWP) output since the 1970s, traditionally trained
on actual past forecasts. Modern machine learning (ML) practitioners, however,
commonly default to ERA5 reanalysis as training data because of its global
coverage and ease of access. This study systematically evaluates whether training
data source matters for ML-based temperature post-processing by comparing
historical forecast data against ERA5 reanalysis across 28 U.S. stations
spanning six climate types, five ML architectures (linear regression, multilayer
perceptron, XGBoost, LightGBM, CatBoost), and six training windows (1-8 years).
Over 2,200 experiments with block-bootstrap confidence intervals and Wilcoxon
signed-rank tests, we find that: (1) historical forecasts reduce mean absolute
error (MAE) by 6-26% compared to reanalysis for every architecture tested
(p < 0.01 in 9 of 10 cases); (2) simple models (linear regression, MLP) match
or outperform gradient-boosted trees (p < 0.001); (3) three years of recent data
suffices for near-optimal performance; and (4) the best ML post-processing
reduces MAE by 35% over raw GFS and 52% over raw NWP blends. A feature ablation
study across 2,240 runs confirms that core NWP predictors and bias-correction
features are essential, while lagged observations and rolling statistics can harm
tree-based models.

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── configs/
│   └── experiment_manifest.json   # Full experiment matrix (2,200+ runs)
├── src/
│   ├── research/
│   │   ├── experiment_runner.py   # Core experiment engine
│   │   ├── models.py             # 5 ML architectures (unified interface)
│   │   ├── stations.py           # 28 station definitions
│   │   ├── database.py           # SQLite results storage
│   │   ├── statistics.py         # Metrics, block bootstrap, significance tests
│   │   └── ablation_config.py    # Feature group definitions for ablation
│   ├── models/
│   │   └── preprocessing.py      # Feature engineering pipeline
│   └── calibration/
│       └── calibration_conformal.py  # Conformal prediction calibrator
├── scripts/
│   └── research/
│       ├── download_all_stations.py  # Download all data (GFS, ECMWF, actuals)
│       ├── download_reanalysis.py    # Download ERA5/Open-Meteo reanalysis
│       ├── run_experiment.py         # Run a single experiment
│       ├── run_all_experiments.py    # Batch runner (resumes from DB)
│       ├── run_ablation.py           # Feature ablation study
│       ├── generate_paper_assets.py  # Generate all tables and figures
│       └── ...                       # Additional utility scripts
├── docs/
│   ├── paper/
│   │   ├── main.tex              # LaTeX manuscript
│   │   ├── references.bib        # Bibliography
│   │   ├── tables/               # Generated CSV/LaTeX tables
│   │   └── figures/              # Generated PDF/PNG figures
│   └── analysis/
│       ├── feature_importance_raw.csv
│       └── ablation_results.csv
└── data/
    └── research/
        └── results.db            # Pre-computed experiment results (SQLite)
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-username>/paper-weather-mos.git
cd paper-weather-mos
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Regenerate paper tables and figures (from bundled results)

The repository includes the pre-computed experiment database (`data/research/results.db`,
~1.2 MB, containing all 2,200+ experiment results). You can regenerate every table
and figure in the paper without re-running experiments:

```bash
python scripts/research/generate_paper_assets.py
```

Output appears in `docs/paper/tables/` and `docs/paper/figures/`.

### 3. Compile the paper

```bash
cd docs/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Full Reproduction (from scratch)

If you want to re-run every experiment from raw data:

### Step 1: Download data

Downloads GFS historical forecasts, ECMWF historical forecasts, ERA5/Open-Meteo
reanalysis, and ACIS actuals for all 28 stations:

```bash
python scripts/research/download_all_stations.py --what all
```

This fetches data from public APIs (ACIS, Open-Meteo) and writes CSVs to
`data/research/`. No API keys are required.

### Step 2: Run the full experiment matrix

```bash
python scripts/research/run_all_experiments.py --phase all --verbose
```

This runs all 2,200+ experiments defined in `configs/experiment_manifest.json`.
Results are saved to `data/research/results.db`. The runner is resumable -- if
interrupted, it skips experiments already in the database.

**Estimated time:** 8-24 hours on a modern machine (depends on CPU cores).

### Step 3: Run ablation study

```bash
python scripts/research/run_ablation.py
```

Results are saved to `docs/analysis/ablation_results.csv`.

### Step 4: Generate paper assets

```bash
python scripts/research/generate_paper_assets.py
```

---

## Experiment Design

| Dimension | Values |
|-----------|--------|
| **Stations** | 28 U.S. stations across 6 climate types |
| **Data sources** | Historical forecasts (GFS + ECMWF) vs. ERA5 reanalysis |
| **Architectures** | Linear regression, MLP, XGBoost, LightGBM, CatBoost |
| **Training windows** | 1, 2, 3, 4, 6, 8 years |
| **Targets** | Daily maximum temperature (MaxT), minimum temperature (MinT) |
| **Test period** | July 2025 - February 2026 (fixed across all experiments) |
| **Statistical tests** | Block bootstrap CIs, Wilcoxon signed-rank tests |

---

## Data Sources

| Source | Description | Access |
|--------|-------------|--------|
| **ACIS** | Official NWS daily climate data (MaxT, MinT) | [rcc-acis.org](https://www.rcc-acis.org/) |
| **Open-Meteo Forecast API** | GFS and ECMWF historical forecasts | [open-meteo.com](https://open-meteo.com/) |
| **Open-Meteo Archive API** | ERA5 reanalysis (used as baseline comparison) | [open-meteo.com](https://open-meteo.com/) |

All data is freely available and downloaded programmatically by the scripts in
this repository. No API keys or paid subscriptions are needed.

---

## Citation

```bibtex
@article{asad2026historicalforecasts,
  title   = {Historical Forecasts vs.\ Reanalysis as Training Data for Machine
             Learning Temperature Post-Processing: A Multi-Station,
             Multi-Architecture Comparison},
  author  = {Asad, Syed Ali and Agarwal, Shalini and Qasim, Syed Mohammad
             and Khan, Akram Tariq and Shah, Karnav},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
