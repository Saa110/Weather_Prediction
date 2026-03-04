"""
Improved Calibration using Conformal Prediction

Instead of isotonic regression (which distorts predictions), use conformal 
prediction to adjust interval widths while preserving median predictions.

This approach:
1. Keeps q50 unchanged (preserves MAE)
2. Adjusts q25/q75 based on empirical residuals
3. Guarantees proper coverage without distorting point predictions
"""

import numpy as np
import pandas as pd
from typing import Tuple
import pickle


class ConformalCalibrator:
    """
    Conformal prediction-based calibrator that preserves median predictions
    while adjusting interval widths for proper coverage.
    """
    
    def __init__(self, target_coverage=0.50, verbose=True):
        """
        Initialize calibrator.
        
        Args:
            target_coverage: Desired coverage (e.g., 0.50 for 50% intervals)
            verbose: If True, print calibration details
        """
        self.target_coverage = target_coverage
        self.lower_adjustment = None
        self.upper_adjustment = None
        self.is_fitted = False
        self.verbose = verbose
    
    def fit(self, predictions: pd.DataFrame, actuals: pd.Series):
        """
        Learn interval adjustments from validation set.
        
        Strategy:
        1. Keep q50 unchanged
        2. Compute residuals (actual - q50)
        3. Find quantiles of residuals that give proper coverage
        4. Adjust q25/q75 based on residual quantiles
        
        Args:
            predictions: DataFrame with columns ['q25', 'q50', 'q75']
            actuals: Series of actual values
        """
        # Compute residuals
        residuals = actuals.values - predictions['q50'].values
        
        # Find the quantiles of residuals for target coverage
        # For 50% coverage, we want 25th and 75th percentile intervals
        lower_percentile = (1 - self.target_coverage) / 2  # 0.25 for 50% coverage
        upper_percentile = 1 - lower_percentile  # 0.75 for 50% coverage
        
        # Compute residual quantiles
        lower_residual_quantile = np.percentile(residuals, lower_percentile * 100)
        upper_residual_quantile = np.percentile(residuals, upper_percentile * 100)
        
        # Store adjustments (these will be added to q50 to get new q25/q75)
        self.lower_adjustment = lower_residual_quantile
        self.upper_adjustment = upper_residual_quantile
        
        self.is_fitted = True
        
        if self.verbose:
            print(f"[CALIBRATION] Learned adjustments:")
            print(f"  Lower (Q25): {self.lower_adjustment:+.2f}°F from median")
            print(f"  Upper (Q75): {self.upper_adjustment:+.2f}°F from median")
        
        return self
    
    def transform(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply calibration to predictions.
        
        Args:
            predictions: DataFrame with columns ['q25', 'q50', 'q75']
        
        Returns:
            DataFrame with calibrated predictions
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before transform")
        
        calibrated = predictions.copy()
        
        # Keep q50 unchanged (preserves MAE)
        # Adjust q25/q75 based on learned residual quantiles
        calibrated['q25'] = predictions['q50'] + self.lower_adjustment
        calibrated['q75'] = predictions['q50'] + self.upper_adjustment
        
        return calibrated
    
    def fit_transform(self, predictions: pd.DataFrame, actuals: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(predictions, actuals)
        return self.transform(predictions)
    
    def save(self, filepath: str):
        """Save calibrator to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"✅ Calibrator saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load calibrator from disk."""
        with open(filepath, 'rb') as f:
            calibrator = pickle.load(f)
        print(f"✅ Calibrator loaded from {filepath}")
        return calibrator


def compute_coverage_metrics(predictions: pd.DataFrame, actuals: pd.Series) -> dict:
    """Compute comprehensive coverage metrics."""
    
    coverage_50 = ((actuals >= predictions['q25']) & (actuals <= predictions['q75'])).mean()
    below_q25 = (actuals < predictions['q25']).mean()
    above_q75 = (actuals > predictions['q75']).mean()
    
    # MAE for median
    mae = np.abs(predictions['q50'] - actuals).mean()
    
    return {
        'mae': mae,
        'coverage_50': coverage_50,
        'below_q25': below_q25,
        'above_q75': above_q75,
        'q25_error': abs(below_q25 - 0.25),
        'q75_error': abs(above_q75 - 0.25),
        'avg_interval_width': (predictions['q75'] - predictions['q25']).mean()
    }


def print_calibration_comparison(uncalibrated_metrics: dict, calibrated_metrics: dict):
    """Print side-by-side comparison of calibration results."""
    
    print("\n" + "="*80)
    print("CALIBRATION RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Uncalibrated':>15} {'Calibrated':>15} {'Change':>15}")
    print("-" * 80)
    
    # MAE (should not change)
    mae_change = calibrated_metrics['mae'] - uncalibrated_metrics['mae']
    print(f"{'MAE':<30} {uncalibrated_metrics['mae']:>14.2f}°F {calibrated_metrics['mae']:>14.2f}°F {mae_change:>+14.2f}°F")
    
    # Coverage
    cov_change = (calibrated_metrics['coverage_50'] - uncalibrated_metrics['coverage_50']) * 100
    print(f"{'50% Interval Coverage':<30} {uncalibrated_metrics['coverage_50']:>14.1%} {calibrated_metrics['coverage_50']:>14.1%} {cov_change:>+14.1f}%")
    
    # Below Q25
    q25_change = (calibrated_metrics['below_q25'] - uncalibrated_metrics['below_q25']) * 100
    print(f"{'Below Q25 (target: 25%)':<30} {uncalibrated_metrics['below_q25']:>14.1%} {calibrated_metrics['below_q25']:>14.1%} {q25_change:>+14.1f}%")
    
    # Above Q75
    q75_change = (calibrated_metrics['above_q75'] - uncalibrated_metrics['above_q75']) * 100
    print(f"{'Above Q75 (target: 25%)':<30} {uncalibrated_metrics['above_q75']:>14.1%} {calibrated_metrics['above_q75']:>14.1%} {q75_change:>+14.1f}%")
    
    # Calibration errors  
    q25_err_change = (calibrated_metrics['q25_error'] - uncalibrated_metrics['q25_error']) * 100
    q75_err_change = (calibrated_metrics['q75_error'] - uncalibrated_metrics['q75_error']) * 100
    print(f"{'Q25 Calibration Error':<30} {uncalibrated_metrics['q25_error']:>14.1%} {calibrated_metrics['q25_error']:>14.1%} {q25_err_change:>+14.1f}%")
    print(f"{'Q75 Calibration Error':<30} {uncalibrated_metrics['q75_error']:>14.1%} {calibrated_metrics['q75_error']:>14.1%} {q75_err_change:>+14.1f}%")
    
    # Interval width
    width_change = calibrated_metrics['avg_interval_width'] - uncalibrated_metrics['avg_interval_width']
    print(f"{'Avg Interval Width':<30} {uncalibrated_metrics['avg_interval_width']:>14.2f}°F {calibrated_metrics['avg_interval_width']:>14.2f}°F {width_change:>+14.2f}°F")
    
    print("="*80)
    
    # Assessment
    print("\n📊 Assessment:")
    if abs(mae_change) < 0.05:
        print(f"  ✅ MAE preserved ({mae_change:+.2f}°F change)")
    else:
        print(f"  ⚠️  MAE changed by {mae_change:+.2f}°F")
    
    if abs(calibrated_metrics['coverage_50'] - 0.50) < abs(uncalibrated_metrics['coverage_50'] - 0.50):
        print(f"  ✅ Coverage improved (closer to 50%)")
    else:
        print(f"  ❌ Coverage did not improve")
    
    if calibrated_metrics['q25_error'] < uncalibrated_metrics['q25_error']:
        print(f"  ✅ Q25 calibration improved")
    else:
        print(f"  ⚠️  Q25 calibration worse")
    
    if calibrated_metrics['q75_error'] < uncalibrated_metrics['q75_error']:
        print(f"  ✅ Q75 calibration improved")
    else:
        print(f"  ⚠️  Q75 calibration worse")
