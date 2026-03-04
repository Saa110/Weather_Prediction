"""
Statistical utilities for the research experiments.

- Bootstrap confidence intervals for MAE, RMSE, Bias
- Paired significance tests (Wilcoxon, t-test)
- Skill score computation
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from scipy import stats


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_metrics(predictions: pd.DataFrame, actuals: pd.Series) -> Dict[str, float]:
    """
    Compute all evaluation metrics from calibrated predictions.

    Args:
        predictions: DataFrame with columns q25, q50, q75
        actuals: Series of actual values

    Returns:
        Dictionary of metric name -> value
    """
    q50 = predictions["q50"].values
    y = actuals.values

    errors = q50 - y
    abs_errors = np.abs(errors)

    coverage = np.mean(
        (y >= predictions["q25"].values) & (y <= predictions["q75"].values)
    )
    interval_width = np.mean(predictions["q75"].values - predictions["q25"].values)

    return {
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "bias": float(np.mean(errors)),
        "coverage_50": float(coverage),
        "interval_width": float(interval_width),
        "n_test": int(len(y)),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(
    predictions: pd.DataFrame,
    actuals: pd.Series,
    metric: str = "mae",
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    block_size: int = 10,
) -> Tuple[float, float, float]:
    """
    Block bootstrap confidence interval for a single metric.

    Uses a moving-block bootstrap to account for temporal autocorrelation
    in forecast errors. Blocks of `block_size` consecutive days are
    resampled with replacement, producing wider (more honest) CIs than
    i.i.d. bootstrap for serially correlated data.

    Args:
        predictions: DataFrame with q25, q50, q75
        actuals: Series of actuals
        metric: One of 'mae', 'rmse', 'bias'
        n_boot: Number of bootstrap resamples
        confidence: Confidence level (e.g. 0.95)
        seed: Random seed
        block_size: Number of consecutive days per block (default 10)

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    q50 = predictions["q50"].values
    y = actuals.values
    n = len(y)

    def _metric(pred, true):
        err = pred - true
        if metric == "mae":
            return np.mean(np.abs(err))
        elif metric == "rmse":
            return np.sqrt(np.mean(err ** 2))
        elif metric == "bias":
            return np.mean(err)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    point = _metric(q50, y)
    boot_values = np.empty(n_boot)

    # Moving-block bootstrap: resample contiguous blocks
    n_blocks_needed = int(np.ceil(n / block_size))
    max_start = n - block_size  # last valid block start index

    if max_start < 1:
        # Fall back to i.i.d. bootstrap if series is shorter than block_size
        for i in range(n_boot):
            idx = rng.randint(0, n, size=n)
            boot_values[i] = _metric(q50[idx], y[idx])
    else:
        for i in range(n_boot):
            # Draw random block starting positions
            starts = rng.randint(0, max_start + 1, size=n_blocks_needed)
            # Concatenate blocks and trim to length n
            idx = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n]
            boot_values[i] = _metric(q50[idx], y[idx])

    alpha = (1 - confidence) / 2
    ci_lo = float(np.percentile(boot_values, alpha * 100))
    ci_hi = float(np.percentile(boot_values, (1 - alpha) * 100))

    return point, ci_lo, ci_hi


def bootstrap_all_metrics(
    predictions: pd.DataFrame,
    actuals: pd.Series,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap CIs for MAE, RMSE, and Bias.

    Returns:
        Dict of metric -> (point, ci_lower, ci_upper)
    """
    result = {}
    for m in ["mae", "rmse", "bias"]:
        result[m] = bootstrap_ci(predictions, actuals, m, n_boot, confidence, seed)
    return result


# ---------------------------------------------------------------------------
# Significance tests
# ---------------------------------------------------------------------------
def paired_test(
    errors_a: np.ndarray,
    errors_b: np.ndarray,
    test: str = "wilcoxon",
) -> Dict[str, float]:
    """
    Test whether two sets of absolute errors differ significantly.

    Args:
        errors_a: Absolute errors from model A (e.g. reanalysis)
        errors_b: Absolute errors from model B (e.g. historical forecast)
        test: 'wilcoxon' (non-parametric) or 'ttest' (parametric)

    Returns:
        Dict with 'statistic', 'p_value', 'mean_diff'
    """
    diff = errors_a - errors_b  # positive = B is better

    if test == "wilcoxon":
        stat, p = stats.wilcoxon(diff, alternative="greater")
    elif test == "ttest":
        stat, p = stats.ttest_rel(errors_a, errors_b)
        p = p / 2  # one-sided
    else:
        raise ValueError(f"Unknown test: {test}")

    return {
        "statistic": float(stat),
        "p_value": float(p),
        "mean_diff": float(np.mean(diff)),
        "median_diff": float(np.median(diff)),
    }


# ---------------------------------------------------------------------------
# Skill score
# ---------------------------------------------------------------------------
def skill_score(model_mae: float, baseline_mae: float) -> float:
    """
    Compute skill score relative to a baseline.

    SS = 1 - (model_MAE / baseline_MAE)
    SS > 0 means the model is better than baseline.
    SS = 1 means the model is perfect.
    """
    if baseline_mae == 0:
        return 0.0
    return 1.0 - (model_mae / baseline_mae)
