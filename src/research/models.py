"""
Multi-architecture model wrapper for the research experiment.

Provides a unified interface across 5 ML architectures so the experiment
runner can swap architectures with a single parameter.

All models train quantile regression (q25, q50, q75) with the same
60/20/20 chronological split and early stopping where applicable.
"""

import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from typing import Tuple, Dict, Optional
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseResearchModel(ABC):
    """Common interface for all research models."""

    def __init__(self, station_id: str, target_col: str = "MaxT"):
        self.station_id = station_id
        self.target_col = target_col
        self.feature_names: list = []
        self.models: dict = {}  # keyed by "q25", "q50", "q75"

    # Columns never used as features
    _DROP_COLS = ["MaxT", "MinT", "Pcpn", "SnowDepth", "day_of_year"]

    def _split(self, df: pd.DataFrame):
        """
        80/20 chronological split for train/validation.

        When used inside the experiment runner, the actual test set is
        separate (fixed test period).  So we only need train + validation
        for early stopping and calibration.

        Returns 6 values for backward compatibility but val == internal_test.
        """
        y = df[self.target_col]
        X = df.drop(columns=[c for c in self._DROP_COLS if c in df.columns])
        self.feature_names = X.columns.tolist()

        n = len(df)
        t1 = int(n * 0.80)

        return (
            X.iloc[:t1], y.iloc[:t1],      # train (80%)
            X.iloc[t1:], y.iloc[t1:],      # val   (20%) - for early stopping
            X.iloc[t1:], y.iloc[t1:],      # same as val - returned for calibration
        )

    @abstractmethod
    def train(self, df: pd.DataFrame, verbose: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """Train models.  Return (X_test, y_test)."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame with columns q25, q50, q75."""

    def save(self, directory: Path, metadata_extra: dict = None):
        """Save model weights and metadata to directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {
            "station_id": self.station_id,
            "target_col": self.target_col,
            "architecture": self.__class__.__name__,
            "feature_names": self.feature_names,
        }
        if metadata_extra:
            meta.update(metadata_extra)
        with open(directory / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    @classmethod
    def load(cls, directory: Path, station_id: str = None, target_col: str = None):
        """Load a saved model from directory."""
        directory = Path(directory)
        with open(directory / "metadata.json") as f:
            meta = json.load(f)
        sid = station_id or meta["station_id"]
        tc = target_col or meta["target_col"]
        instance = cls(sid, tc)
        instance.feature_names = meta["feature_names"]
        instance._load_weights(directory)
        return instance

    @abstractmethod
    def _load_weights(self, directory: Path):
        """Subclass-specific weight loading."""


# ---------------------------------------------------------------------------
# 1. CatBoost  (existing workhorse)
# ---------------------------------------------------------------------------
class CatBoostResearchModel(BaseResearchModel):

    def train(self, df, verbose=False):
        from catboost import CatBoostRegressor

        X_tr, y_tr, X_val, y_val, X_te, y_te = self._split(df)

        for q in [0.25, 0.50, 0.75]:
            qname = f"q{int(q*100)}"
            model = CatBoostRegressor(
                iterations=5000,
                learning_rate=0.03,
                depth=4,
                l2_leaf_reg=10,
                early_stopping_rounds=100,
                bagging_temperature=0.5,
                random_strength=1.0,
                subsample=0.8,
                colsample_bylevel=0.8,
                loss_function=f"Quantile:alpha={q}",
                eval_metric=f"Quantile:alpha={q}",
                use_best_model=True,
                verbose=0,
                allow_writing_files=False,
                random_seed=42,
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=0)
            self.models[qname] = model

        return X_te, y_te

    def predict(self, X):
        out = pd.DataFrame(index=X.index)
        for qname, model in self.models.items():
            out[qname] = model.predict(X)
        return out


# ---------------------------------------------------------------------------
# 2. XGBoost
# ---------------------------------------------------------------------------
class XGBoostResearchModel(BaseResearchModel):

    def train(self, df, verbose=False):
        import xgboost as xgb

        X_tr, y_tr, X_val, y_val, X_te, y_te = self._split(df)

        for q in [0.25, 0.50, 0.75]:
            qname = f"q{int(q*100)}"
            dtrain = xgb.DMatrix(X_tr, label=y_tr)
            dval = xgb.DMatrix(X_val, label=y_val)

            params = {
                "objective": "reg:quantileerror",
                "quantile_alpha": q,
                "max_depth": 4,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 10,
                "seed": 42,
                "verbosity": 0,
            }

            model = xgb.train(
                params, dtrain,
                num_boost_round=5000,
                evals=[(dval, "val")],
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            self.models[qname] = model

        return X_te, y_te

    def predict(self, X):
        import xgboost as xgb

        dX = xgb.DMatrix(X)
        out = pd.DataFrame(index=X.index)
        for qname, model in self.models.items():
            out[qname] = model.predict(dX)
        return out


# ---------------------------------------------------------------------------
# 3. LightGBM
# ---------------------------------------------------------------------------
class LightGBMResearchModel(BaseResearchModel):

    def train(self, df, verbose=False):
        import lightgbm as lgb

        X_tr, y_tr, X_val, y_val, X_te, y_te = self._split(df)

        for q in [0.25, 0.50, 0.75]:
            qname = f"q{int(q*100)}"
            train_ds = lgb.Dataset(X_tr, label=y_tr)
            val_ds = lgb.Dataset(X_val, label=y_val, reference=train_ds)

            params = {
                "objective": "quantile",
                "alpha": q,
                "num_leaves": 31,
                "learning_rate": 0.03,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 10,
                "seed": 42,
                "verbose": -1,
            }

            callbacks = [
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(0),
            ]

            model = lgb.train(
                params, train_ds,
                num_boost_round=5000,
                valid_sets=[val_ds],
                callbacks=callbacks,
            )
            self.models[qname] = model

        return X_te, y_te

    def predict(self, X):
        out = pd.DataFrame(index=X.index)
        for qname, model in self.models.items():
            out[qname] = model.predict(X)
        return out


# ---------------------------------------------------------------------------
# 4. Linear Quantile Regression  (scikit-learn)
# ---------------------------------------------------------------------------
class LinearResearchModel(BaseResearchModel):

    def train(self, df, verbose=False):
        from sklearn.linear_model import QuantileRegressor

        X_tr, y_tr, X_val, y_val, X_te, y_te = self._split(df)

        # Fill NaN with training-set median (not 0, which is a real temp value)
        self._fill_values = X_tr.median()
        X_tr_filled = X_tr.fillna(self._fill_values)

        for q in [0.25, 0.50, 0.75]:
            qname = f"q{int(q*100)}"
            model = QuantileRegressor(
                quantile=q,
                alpha=0.01,  # light L1 regularisation
                solver="highs",
            )
            model.fit(X_tr_filled, y_tr)
            self.models[qname] = model

        return X_te, y_te

    def predict(self, X):
        X_filled = X.fillna(self._fill_values)
        out = pd.DataFrame(index=X.index)
        for qname, model in self.models.items():
            out[qname] = model.predict(X_filled)
        return out

    def save(self, directory: Path, metadata_extra: dict = None):
        directory = Path(directory)
        super().save(directory, metadata_extra)
        for qname, model in self.models.items():
            with open(directory / f"{qname}.pkl", "wb") as f:
                pickle.dump(model, f)
        with open(directory / "fill_values.pkl", "wb") as f:
            pickle.dump(self._fill_values, f)

    def _load_weights(self, directory: Path):
        for qname in ["q25", "q50", "q75"]:
            with open(directory / f"{qname}.pkl", "rb") as f:
                self.models[qname] = pickle.load(f)
        with open(directory / "fill_values.pkl", "rb") as f:
            self._fill_values = pickle.load(f)


# ---------------------------------------------------------------------------
# 5. MLP Neural Network  (PyTorch)
# ---------------------------------------------------------------------------
class MLPResearchModel(BaseResearchModel):
    """
    Simple 2-hidden-layer MLP trained with pinball (quantile) loss.
    Uses PyTorch if available, otherwise falls back to sklearn MLPRegressor
    with a median-only approach.
    """

    def train(self, df, verbose=False):
        X_tr, y_tr, X_val, y_val, X_te, y_te = self._split(df)

        try:
            self._train_torch(X_tr, y_tr, X_val, y_val)
        except ImportError:
            self._train_sklearn(X_tr, y_tr, X_val, y_val)

        return X_te, y_te

    # -- torch path --
    def _train_torch(self, X_tr, y_tr, X_val, y_val):
        import torch
        import torch.nn as nn

        # Explicit seed for reproducibility across runs
        torch.manual_seed(42)

        device = torch.device("cpu")

        # Fill NaN with training-set median (not 0, which is a real temp value)
        self._fill_values = X_tr.median()
        X_tr_filled = X_tr.fillna(self._fill_values)

        # Standardise using filled training data
        self._mean = X_tr_filled.mean()
        self._std = X_tr_filled.std().replace(0, 1)

        def to_tensor(X, y=None):
            Xt = torch.tensor(
                ((X.fillna(self._fill_values) - self._mean) / self._std).values,
                dtype=torch.float32, device=device,
            )
            if y is not None:
                yt = torch.tensor(y.values, dtype=torch.float32, device=device)
                return Xt, yt
            return Xt

        Xtr_t, ytr_t = to_tensor(X_tr, y_tr)
        Xval_t, yval_t = to_tensor(X_val, y_val)

        n_features = Xtr_t.shape[1]

        for q in [0.25, 0.50, 0.75]:
            qname = f"q{int(q*100)}"

            net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            ).to(device)

            optimiser = torch.optim.Adam(net.parameters(), lr=1e-3)

            def pinball(pred, target, alpha):
                err = target - pred.squeeze()
                return torch.mean(torch.max(alpha * err, (alpha - 1) * err))

            best_val = float("inf")
            patience, wait = 50, 0
            best_state = None

            for epoch in range(2000):
                net.train()
                optimiser.zero_grad()
                loss = pinball(net(Xtr_t), ytr_t, q)
                loss.backward()
                optimiser.step()

                net.eval()
                with torch.no_grad():
                    val_loss = pinball(net(Xval_t), yval_t, q).item()
                if val_loss < best_val - 1e-5:
                    best_val = val_loss
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        break

            if best_state is not None:
                net.load_state_dict(best_state)
            net.eval()
            self.models[qname] = net

    def _train_sklearn(self, X_tr, y_tr, X_val, y_val):
        """Fallback: sklearn QuantileRegressor if torch unavailable."""
        from sklearn.linear_model import QuantileRegressor

        self._fill_values = X_tr.median()
        X_tr_filled = X_tr.fillna(self._fill_values)
        self._mean = X_tr_filled.mean()
        self._std = X_tr_filled.std().replace(0, 1)
        X_tr_s = (X_tr_filled - self._mean) / self._std

        for q in [0.25, 0.50, 0.75]:
            qname = f"q{int(q*100)}"
            model = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
            model.fit(X_tr_s, y_tr)
            self.models[qname] = model

    def predict(self, X):
        X_s = (X.fillna(self._fill_values) - self._mean) / self._std
        out = pd.DataFrame(index=X.index)

        try:
            import torch
            Xt = torch.tensor(X_s.values, dtype=torch.float32)
            for qname, net in self.models.items():
                net.eval()
                with torch.no_grad():
                    out[qname] = net(Xt).squeeze().numpy()
        except (ImportError, AttributeError):
            # sklearn fallback
            for qname, model in self.models.items():
                out[qname] = model.predict(X_s)

        return out

    def save(self, directory: Path, metadata_extra: dict = None):
        import torch
        directory = Path(directory)
        super().save(directory, metadata_extra)
        for qname, net in self.models.items():
            torch.save(net.state_dict(), directory / f"{qname}.pt")
        with open(directory / "normalization.pkl", "wb") as f:
            pickle.dump({
                "fill_values": self._fill_values,
                "mean": self._mean,
                "std": self._std,
                "n_features": len(self.feature_names),
            }, f)

    def _load_weights(self, directory: Path):
        import torch
        import torch.nn as nn
        with open(directory / "normalization.pkl", "rb") as f:
            norm = pickle.load(f)
        self._fill_values = norm["fill_values"]
        self._mean = norm["mean"]
        self._std = norm["std"]
        n_features = norm["n_features"]
        for qname in ["q25", "q50", "q75"]:
            net = nn.Sequential(
                nn.Linear(n_features, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
            net.load_state_dict(torch.load(directory / f"{qname}.pt", weights_only=True))
            net.eval()
            self.models[qname] = net


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
ARCHITECTURES = {
    "catboost": CatBoostResearchModel,
    "xgboost": XGBoostResearchModel,
    "lightgbm": LightGBMResearchModel,
    "linear": LinearResearchModel,
    "mlp": MLPResearchModel,
}


def build_model(architecture: str, station_id: str, target_col: str = "MaxT") -> BaseResearchModel:
    """
    Factory function to create a model by name.

    Args:
        architecture: One of 'catboost', 'xgboost', 'lightgbm', 'linear', 'mlp'
        station_id: Station ICAO code
        target_col: Target column name

    Returns:
        Instance of BaseResearchModel subclass
    """
    cls = ARCHITECTURES.get(architecture.lower())
    if cls is None:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            f"Choose from: {list(ARCHITECTURES.keys())}"
        )
    return cls(station_id, target_col)
