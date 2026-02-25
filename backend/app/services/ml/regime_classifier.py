"""
Volatility Regime Classifier using XGBoost.

Classifies market conditions into:
- Low Vol Trending (regime 0)
- High Vol Mean Reverting (regime 1)
- Crisis (regime 2)

Uses walk-forward validation for robust evaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except Exception:
    HAS_XGBOOST = False
    XGBClassifier = None

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

from ...models.options import VolatilityRegime, VolatilityPrediction
from .features import FeatureEngineer, FeatureSet
from ...core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ModelMetrics:
    accuracy: float
    per_class_f1: dict[str, float]
    feature_importance: dict[str, float]


class VolatilityRegimeClassifier:
    """XGBoost-based volatility regime classification."""

    REGIME_MAP = {
        0: VolatilityRegime.LOW_VOL_TRENDING,
        1: VolatilityRegime.HIGH_VOL_MEAN_REVERTING,
        2: VolatilityRegime.CRISIS,
    }

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self.model: XGBClassifier | None = None
        self.scaler = StandardScaler()
        self.feature_engineer = FeatureEngineer()
        self.feature_names: list[str] = []
        self._is_trained = False

    def train(
        self,
        prices: pd.DataFrame,
        iv_data: pd.Series | None = None,
        n_splits: int = 5,
    ) -> ModelMetrics:
        """
        Train regime classifier with walk-forward cross-validation.

        Args:
            prices: OHLCV DataFrame
            iv_data: Optional IV time series
            n_splits: Number of time-series CV splits
        """
        log.info("training_regime_classifier", rows=len(prices))

        # Feature engineering
        feature_set = self.feature_engineer.compute_all_features(prices, iv_data)
        labels = self.feature_engineer.compute_regime_labels(prices)

        # Align features and labels
        common_idx = feature_set.features.index.intersection(labels.index)
        X = feature_set.features.loc[common_idx]
        y = labels.loc[common_idx]

        # Remove any remaining NaN/inf
        mask = X.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        X = X[mask]
        y = y[mask]

        self.feature_names = list(X.columns)

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns,
        )

        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []

        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42,
                verbosity=0,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )

            score = accuracy_score(y_test, model.predict(X_test))
            cv_scores.append(score)

        # Final model on all data
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        )
        self.model.fit(X_scaled, y, verbose=False)
        self._is_trained = True

        # Feature importance
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist(),
        ))
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])

        # Per-class metrics
        y_pred = self.model.predict(X_scaled)
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)

        metrics = ModelMetrics(
            accuracy=round(np.mean(cv_scores), 4),
            per_class_f1={
                self.REGIME_MAP[int(k)].value: round(v["f1-score"], 4)
                for k, v in report.items()
                if k.isdigit()
            },
            feature_importance=top_features,
        )

        log.info(
            "regime_classifier_trained",
            cv_accuracy=metrics.accuracy,
            n_features=len(self.feature_names),
        )

        return metrics

    def predict(
        self,
        prices: pd.DataFrame,
        iv_data: pd.Series | None = None,
        symbol: str = "SPY",
    ) -> VolatilityPrediction:
        """Predict current volatility regime."""
        if not self._is_trained:
            # Return default prediction if not trained
            return self._default_prediction(symbol)

        feature_set = self.feature_engineer.compute_all_features(prices, iv_data)
        X = feature_set.features.iloc[[-1]]  # Latest row

        # Handle missing features
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_names]

        X_scaled = self.scaler.transform(X)
        proba = self.model.predict_proba(X_scaled)[0]
        regime_idx = int(np.argmax(proba))

        # Feature importance for this prediction
        importance = dict(zip(
            self.feature_names,
            self.model.feature_importances_.tolist(),
        ))
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])

        return VolatilityPrediction(
            symbol=symbol,
            regime=self.REGIME_MAP[regime_idx],
            regime_probabilities={
                self.REGIME_MAP[i].value: round(float(p), 4)
                for i, p in enumerate(proba)
            },
            predicted_iv_30d=round(float(feature_set.features.iloc[-1].get("realized_vol_20", 20.0)), 2),
            confidence=round(float(np.max(proba)), 4),
            features_importance=top_features,
            timestamp=datetime.utcnow(),
        )

    def _default_prediction(self, symbol: str) -> VolatilityPrediction:
        """Return default prediction when model isn't trained."""
        return VolatilityPrediction(
            symbol=symbol,
            regime=VolatilityRegime.LOW_VOL_TRENDING,
            regime_probabilities={
                "low_vol_trending": 0.60,
                "high_vol_mean_reverting": 0.30,
                "crisis": 0.10,
            },
            predicted_iv_30d=18.5,
            confidence=0.60,
            features_importance={},
            timestamp=datetime.utcnow(),
        )

    def save(self, path: str | None = None):
        """Save model to disk."""
        if not self._is_trained:
            return
        path = path or os.path.join(self.model_dir, "regime_classifier.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
        }, path)
        log.info("model_saved", path=path)

    def load(self, path: str | None = None):
        """Load model from disk."""
        path = path or os.path.join(self.model_dir, "regime_classifier.joblib")
        if not os.path.exists(path):
            log.warning("model_file_not_found", path=path)
            return
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data["feature_names"]
        self._is_trained = True
        log.info("model_loaded", path=path)
