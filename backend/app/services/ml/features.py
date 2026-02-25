"""
Feature engineering for ML models.

Computes technical, volatility, and market microstructure features
from raw market data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class FeatureSet:
    features: pd.DataFrame
    feature_names: list[str]
    target: pd.Series | None = None


class FeatureEngineer:
    """Compute quant features for volatility regime classification."""

    def compute_all_features(
        self,
        prices: pd.DataFrame,
        iv_data: pd.Series | None = None,
    ) -> FeatureSet:
        """
        Compute full feature set from OHLCV data.

        Args:
            prices: DataFrame with columns [open, high, low, close, volume]
            iv_data: Optional implied volatility series (same index as prices)
        """
        df = prices.copy()

        # Price features
        df["returns"] = df["close"].pct_change()
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # ─── Momentum / Trend ─────────────────────────────
        df["rsi_14"] = self._rsi(df["close"], 14)
        df["rsi_7"] = self._rsi(df["close"], 7)

        # Bollinger Bands
        bb_mid = df["close"].rolling(20).mean()
        bb_std = df["close"].rolling(20).std()
        df["bb_upper"] = bb_mid + 2 * bb_std
        df["bb_lower"] = bb_mid - 2 * bb_std
        df["bb_pctb"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / bb_mid

        # Moving averages
        for window in [5, 10, 20, 50]:
            df[f"sma_{window}"] = df["close"].rolling(window).mean()
            df[f"close_vs_sma_{window}"] = df["close"] / df[f"sma_{window}"] - 1

        # MACD
        ema12 = df["close"].ewm(span=12).mean()
        ema26 = df["close"].ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]

        # ─── Volatility ───────────────────────────────────
        for window in [5, 10, 20, 60]:
            df[f"realized_vol_{window}"] = df["log_returns"].rolling(window).std() * np.sqrt(252)

        # ATR (Average True Range)
        df["atr_14"] = self._atr(df, 14)
        df["atr_pct"] = df["atr_14"] / df["close"]

        # Parkinson volatility (using high-low)
        df["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (np.log(df["high"] / df["low"]) ** 2)
        ).rolling(20).mean() * np.sqrt(252)

        # Garman-Klass volatility
        df["gk_vol"] = np.sqrt(
            0.5 * np.log(df["high"] / df["low"]) ** 2
            - (2 * np.log(2) - 1) * np.log(df["close"] / df["open"]) ** 2
        ).rolling(20).mean() * np.sqrt(252)

        # Vol of vol
        df["vol_of_vol"] = df["realized_vol_20"].rolling(20).std()

        # ─── Volatility Surface Features ──────────────────
        if iv_data is not None:
            df["iv"] = iv_data
            df["iv_rv_spread"] = df["iv"] - df["realized_vol_20"]  # Variance Risk Premium
            df["iv_rank_20"] = df["iv"].rolling(252).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
                raw=False,
            )

        # ─── Volume / Microstructure ──────────────────────
        df["volume_sma_20"] = df["volume"].rolling(20).mean()
        df["relative_volume"] = df["volume"] / df["volume_sma_20"]
        df["volume_change"] = df["volume"].pct_change()

        # ─── Mean reversion ───────────────────────────────
        df["z_score_20"] = (df["close"] - df["close"].rolling(20).mean()) / df["close"].rolling(20).std()
        df["z_score_50"] = (df["close"] - df["close"].rolling(50).mean()) / df["close"].rolling(50).std()

        # ─── Trend strength ───────────────────────────────
        df["adx_14"] = self._adx(df, 14)

        # Drop NaN rows and select feature columns
        feature_cols = [c for c in df.columns if c not in ["open", "high", "low", "close", "volume", "returns", "log_returns"]]
        df = df.dropna()

        return FeatureSet(
            features=df[feature_cols],
            feature_names=feature_cols,
        )

    def compute_regime_labels(
        self,
        prices: pd.DataFrame,
        lookforward: int = 20,
    ) -> pd.Series:
        """
        Generate regime labels based on forward-looking characteristics.

        Labels:
        0 = Low Vol Trending
        1 = High Vol Mean Reverting
        2 = Crisis
        """
        returns = prices["close"].pct_change()
        fwd_vol = returns.rolling(lookforward).std().shift(-lookforward) * np.sqrt(252)
        fwd_return = prices["close"].pct_change(lookforward).shift(-lookforward)

        labels = pd.Series(0, index=prices.index)

        # Crisis: forward vol > 30% and drawdown > 5%
        labels[fwd_vol > 0.30] = 2

        # High vol mean reverting: vol > 20% but recovery
        labels[(fwd_vol > 0.20) & (fwd_vol <= 0.30)] = 1

        # Low vol trending: vol < 20%
        labels[fwd_vol <= 0.20] = 0

        return labels.dropna()

    # ─── Technical Indicator Helpers ──────────────────────

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift()).abs()
        low_close = (df["low"] - df["close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(period).mean()

    @staticmethod
    def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Simplified ADX calculation."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        atr = FeatureEngineer._atr(df, period)
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1)
        return dx.rolling(period).mean()
