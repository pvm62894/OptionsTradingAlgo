"""
Options Trading Algorithm - Technical Indicators Module

Advanced technical indicators for momentum analysis and signal generation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')


class MomentumIndicators:
    """
    Advanced momentum indicators for trend analysis and signal generation.
    """
    
    @staticmethod
    def calculate_momentum_score(data: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
        """
        Calculate composite momentum score combining multiple indicators.
        
        Args:
            data: DataFrame with OHLCV data
            lookback: Lookback period for calculations
        
        Returns:
            DataFrame with momentum scores and individual components
        """
        df = data.copy()
        
        # Price momentum
        df['price_momentum'] = (df['Close'] / df['Close'].shift(lookback) - 1) * 100
        
        # Volume momentum
        df['volume_momentum'] = (df['Volume'] / df['Volume'].rolling(lookback).mean() - 1) * 100
        
        # RSI momentum (deviation from 50)
        df['rsi_momentum'] = (df['RSI'] - 50) * 2
        
        # MACD momentum
        df['macd_momentum'] = np.where(df['MACD'] > df['MACD_Signal'], 
                                     np.abs(df['MACD_Histogram']) * 100,
                                     -np.abs(df['MACD_Histogram']) * 100)
        
        # Bollinger Band position momentum
        df['bb_momentum'] = (df['BB_Position'] - 0.5) * 200
        
        # Composite momentum score (weighted average)
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Price, Volume, RSI, MACD, BB
        df['momentum_score'] = (
            df['price_momentum'] * weights[0] +
            df['volume_momentum'] * weights[1] +
            df['rsi_momentum'] * weights[2] +
            df['macd_momentum'] * weights[3] +
            df['bb_momentum'] * weights[4]
        )
        
        # Normalize momentum score
        df['momentum_score_normalized'] = (
            (df['momentum_score'] - df['momentum_score'].rolling(50).mean()) /
            df['momentum_score'].rolling(50).std()
        )
        
        return df
    
    @staticmethod
    def detect_trend_strength(data: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate trend strength using multiple methods.
        
        Args:
            data: DataFrame with price data
            period: Period for trend calculations
        
        Returns:
            DataFrame with trend strength metrics
        """
        df = data.copy()
        
        # ADX (Average Directional Index)
        df = MomentumIndicators._calculate_adx(df, period)
        
        # Trend consistency (percentage of higher closes)
        df['trend_consistency'] = df['Close'].rolling(period).apply(
            lambda x: (x.diff() > 0).sum() / len(x) * 100
        )
        
        # Linear regression slope
        df['lr_slope'] = df['Close'].rolling(period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
        )
        
        # R-squared for trend quality
        df['lr_r_squared'] = df['Close'].rolling(period).apply(
            lambda x: np.corrcoef(range(len(x)), x)[0, 1]**2 if len(x) == period else np.nan
        )
        
        # Composite trend strength
        df['trend_strength'] = (
            df['ADX'] * 0.4 +
            np.abs(df['trend_consistency'] - 50) * 2 * 0.3 +
            np.abs(df['lr_slope']) * 1000 * 0.3
        )
        
        return df
    
    @staticmethod
    def _calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        df = data.copy()
        
        # True Range
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                np.abs(df['High'] - df['Close'].shift(1)),
                np.abs(df['Low'] - df['Close'].shift(1))
            )
        )
        
        # Directional Movement
        df['DM_plus'] = np.where(
            (df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']),
            np.maximum(df['High'] - df['High'].shift(1), 0),
            0
        )
        
        df['DM_minus'] = np.where(
            (df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)),
            np.maximum(df['Low'].shift(1) - df['Low'], 0),
            0
        )
        
        # Smoothed values
        df['ATR'] = df['TR'].rolling(period).mean()
        df['DI_plus'] = 100 * (df['DM_plus'].rolling(period).mean() / df['ATR'])
        df['DI_minus'] = 100 * (df['DM_minus'].rolling(period).mean() / df['ATR'])
        
        # ADX calculation
        df['DX'] = 100 * np.abs(df['DI_plus'] - df['DI_minus']) / (df['DI_plus'] + df['DI_minus'])
        df['ADX'] = df['DX'].rolling(period).mean()
        
        return df
    
    @staticmethod
    def identify_support_resistance(data: pd.DataFrame, window: int = 5) -> Dict:
        """
        Identify support and resistance levels using pivot points.
        
        Args:
            data: DataFrame with OHLC data
            window: Window size for pivot detection
        
        Returns:
            Dictionary with support and resistance levels
        """
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find pivot highs (resistance)
        resistance_idx = argrelextrema(highs, np.greater, order=window)[0]
        resistance_levels = [(data.index[i], highs[i]) for i in resistance_idx]
        
        # Find pivot lows (support)
        support_idx = argrelextrema(lows, np.less, order=window)[0]
        support_levels = [(data.index[i], lows[i]) for i in support_idx]
        
        return {
            'resistance': resistance_levels[-10:],  # Last 10 resistance levels
            'support': support_levels[-10:],        # Last 10 support levels
            'current_price': data['Close'].iloc[-1]
        }
    
    @staticmethod
    def calculate_volatility_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various volatility indicators.
        
        Args:
            data: DataFrame with OHLCV data
        
        Returns:
            DataFrame with volatility indicators
        """
        df = data.copy()
        
        # Historical volatility (multiple periods)
        for period in [10, 20, 30]:
            df[f'hist_vol_{period}'] = (
                df['Returns'].rolling(period).std() * np.sqrt(252) * 100
            )
        
        # Volatility percentile
        df['vol_percentile'] = df['hist_vol_20'].rolling(252).apply(
            lambda x: (x.iloc[-1] > x).sum() / len(x) * 100
        )
        
        # Volatility regime
        df['vol_regime'] = np.where(
            df['vol_percentile'] > 80, 'high',
            np.where(df['vol_percentile'] < 20, 'low', 'medium')
        )
        
        # VIX-like calculation (simplified)
        df['volatility_index'] = df['hist_vol_20'] * (1 + df['ATR'] / df['Close'])
        
        return df


class SignalGenerator:
    """
    Generate trading signals based on momentum and technical indicators.
    """
    
    def __init__(self, momentum_threshold: float = 1.5, trend_threshold: float = 25):
        """
        Initialize signal generator with thresholds.
        
        Args:
            momentum_threshold: Minimum momentum score for signal
            trend_threshold: Minimum trend strength for signal
        """
        self.momentum_threshold = momentum_threshold
        self.trend_threshold = trend_threshold
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive trading signals.
        
        Args:
            data: DataFrame with technical indicators
        
        Returns:
            DataFrame with signal columns
        """
        df = data.copy()
        
        # Momentum signals
        df['momentum_signal'] = np.where(
            df['momentum_score_normalized'] > self.momentum_threshold, 1,
            np.where(df['momentum_score_normalized'] < -self.momentum_threshold, -1, 0)
        )
        
        # Trend signals
        df['trend_signal'] = np.where(
            (df['trend_strength'] > self.trend_threshold) & (df['lr_slope'] > 0), 1,
            np.where(
                (df['trend_strength'] > self.trend_threshold) & (df['lr_slope'] < 0), -1, 0
            )
        )
        
        # Volatility signals
        df['vol_signal'] = np.where(
            df['vol_regime'] == 'high', 1,  # High volatility = options opportunities
            np.where(df['vol_regime'] == 'low', -1, 0)
        )
        
        # RSI signals
        df['rsi_signal'] = np.where(
            df['RSI'] < 30, 1,  # Oversold
            np.where(df['RSI'] > 70, -1, 0)  # Overbought
        )
        
        # MACD signals
        df['macd_signal'] = np.where(
            (df['MACD'] > df['MACD_Signal']) & 
            (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1)), 1,
            np.where(
                (df['MACD'] < df['MACD_Signal']) & 
                (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1)), -1, 0
            )
        )
        
        # Composite signal
        signal_weights = {
            'momentum_signal': 0.25,
            'trend_signal': 0.25,
            'vol_signal': 0.2,
            'rsi_signal': 0.15,
            'macd_signal': 0.15
        }
        
        df['composite_signal'] = sum(
            df[signal] * weight for signal, weight in signal_weights.items()
        )
        
        # Final signal (with threshold)
        df['final_signal'] = np.where(
            df['composite_signal'] > 0.3, 1,
            np.where(df['composite_signal'] < -0.3, -1, 0)
        )
        
        # Signal strength
        df['signal_strength'] = np.abs(df['composite_signal'])
        
        return df
    
    def identify_options_opportunities(self, data: pd.DataFrame) -> List[Dict]:
        """
        Identify specific options trading opportunities.
        
        Args:
            data: DataFrame with signals and indicators
        
        Returns:
            List of trading opportunities
        """
        opportunities = []
        latest = data.iloc[-1]
        
        # High momentum + high volatility = Long straddle/strangle
        if (latest['momentum_score_normalized'] > 2.0 and 
            latest['vol_regime'] == 'high'):
            opportunities.append({
                'strategy': 'long_straddle',
                'reason': 'High momentum with high volatility',
                'confidence': min(latest['signal_strength'] * 100, 95)
            })
        
        # Strong trend + low volatility = Directional play
        if (latest['trend_strength'] > 30 and 
            latest['vol_regime'] == 'low'):
            direction = 'bullish' if latest['trend_signal'] > 0 else 'bearish'
            opportunities.append({
                'strategy': f'{direction}_call' if direction == 'bullish' else f'{direction}_put',
                'reason': f'Strong {direction} trend with low volatility',
                'confidence': min(latest['signal_strength'] * 100, 95)
            })
        
        # Mean reversion setup
        if (latest['RSI'] > 70 or latest['RSI'] < 30) and latest['vol_regime'] != 'low':
            direction = 'bearish' if latest['RSI'] > 70 else 'bullish'
            opportunities.append({
                'strategy': 'iron_condor' if latest['vol_regime'] == 'high' else f'{direction}_reversion',
                'reason': f'Mean reversion setup - RSI {latest["RSI"]:.1f}',
                'confidence': min((abs(latest['RSI'] - 50) / 50) * 100, 90)
            })
        
        return opportunities


if __name__ == "__main__":
    # Test the indicators
    from data_provider import MarketDataProvider
    
    provider = MarketDataProvider()
    data = provider.get_stock_data("AAPL", "6mo")
    
    if not data.empty:
        print("Testing momentum indicators...")
        
        # Calculate momentum
        momentum_calc = MomentumIndicators()
        data = momentum_calc.calculate_momentum_score(data)
        data = momentum_calc.detect_trend_strength(data)
        data = momentum_calc.calculate_volatility_indicators(data)
        
        # Generate signals
        signal_gen = SignalGenerator()
        data = signal_gen.generate_signals(data)
        
        print(f"Latest momentum score: {data['momentum_score_normalized'].iloc[-1]:.2f}")
        print(f"Latest trend strength: {data['trend_strength'].iloc[-1]:.2f}")
        print(f"Latest signal: {data['final_signal'].iloc[-1]}")
        
        # Identify opportunities
        opportunities = signal_gen.identify_options_opportunities(data)
        print(f"\nFound {len(opportunities)} trading opportunities:")
        for opp in opportunities:
            print(f"- {opp['strategy']}: {opp['reason']} (Confidence: {opp['confidence']:.1f}%)")
    else:
        print("No data available for testing")