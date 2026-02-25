"""
Options Trading Algorithm - Data Provider Module

This module handles market data acquisition, options chains, and historical data
for momentum analysis and machine learning model training.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MarketDataProvider:
    """
    Handles all market data operations including stock prices, options chains,
    and historical volatility calculations.
    """
    
    def __init__(self, cache_dir: str = "data"):
        """
        Initialize the market data provider.
        
        Args:
            cache_dir: Directory to cache data locally
        """
        self.cache_dir = cache_dir
        self.data_cache = {}
        
    def get_stock_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        """
        Retrieve historical stock data.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data and additional technical columns
        """
        cache_key = f"{symbol}_{period}_{interval}"
        
        if cache_key in self.data_cache:
            return self.data_cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            
            # Cache the data
            self.data_cache[cache_key] = data
            
            return data
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_options_chain(self, symbol: str, expiration_date: Optional[str] = None) -> Dict:
        """
        Retrieve options chain data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol
            expiration_date: Specific expiration date (YYYY-MM-DD)
        
        Returns:
            Dictionary containing calls and puts options data
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            expirations = ticker.options
            
            if not expirations:
                return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expirations": []}
            
            # Use the nearest expiration if none specified
            if expiration_date is None:
                expiration_date = expirations[0]
            
            # Get options chain
            options_chain = ticker.option_chain(expiration_date)
            
            # Add Greeks calculations
            calls = self._add_greeks(options_chain.calls, "call")
            puts = self._add_greeks(options_chain.puts, "put")
            
            return {
                "calls": calls,
                "puts": puts,
                "expirations": list(expirations),
                "current_price": ticker.info.get('currentPrice', 0)
            }
            
        except Exception as e:
            print(f"Error fetching options chain for {symbol}: {e}")
            return {"calls": pd.DataFrame(), "puts": pd.DataFrame(), "expirations": []}
    
    def get_implied_volatility_surface(self, symbol: str) -> pd.DataFrame:
        """
        Calculate implied volatility surface across strikes and expirations.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            DataFrame with IV surface data
        """
        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            
            iv_surface = []
            current_price = ticker.info.get('currentPrice', 0)
            
            for exp_date in expirations[:4]:  # First 4 expirations
                try:
                    chain = ticker.option_chain(exp_date)
                    calls = chain.calls
                    
                    # Calculate moneyness and time to expiry
                    exp_datetime = datetime.strptime(exp_date, '%Y-%m-%d')
                    dte = (exp_datetime - datetime.now()).days
                    
                    for _, option in calls.iterrows():
                        if option['impliedVolatility'] > 0:
                            iv_surface.append({
                                'expiration': exp_date,
                                'strike': option['strike'],
                                'dte': dte,
                                'moneyness': option['strike'] / current_price,
                                'iv': option['impliedVolatility'],
                                'volume': option['volume'] or 0,
                                'openInterest': option['openInterest'] or 0
                            })
                            
                except Exception:
                    continue
            
            return pd.DataFrame(iv_surface)
            
        except Exception as e:
            print(f"Error calculating IV surface for {symbol}: {e}")
            return pd.DataFrame()
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators for momentum analysis."""
        df = data.copy()
        
        # Price-based indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Volatility (ATR)
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        true_range = pd.DataFrame({'HL': high_low, 'HC': high_close, 'LC': low_close}).max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()
        
        # Historical Volatility (20-day)
        df['Returns'] = df['Close'].pct_change()
        df['Historical_Vol'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _add_greeks(self, options_df: pd.DataFrame, option_type: str) -> pd.DataFrame:
        """
        Add simplified Greeks calculations to options data.
        
        Note: These are simplified calculations. For production use,
        consider implementing full Black-Scholes Greeks.
        """
        df = options_df.copy()
        
        # Simplified Delta approximation
        if option_type == "call":
            df['delta_approx'] = np.where(df['inTheMoney'], 0.7, 0.3)
        else:
            df['delta_approx'] = np.where(df['inTheMoney'], -0.7, -0.3)
        
        # Simplified Gamma (higher for ATM options)
        df['gamma_approx'] = np.where(
            np.abs(df['strike'] - df['lastPrice']) < df['lastPrice'] * 0.05,
            0.1, 0.05
        )
        
        # Theta approximation (time decay)
        df['theta_approx'] = -df['lastPrice'] * 0.02  # Simplified
        
        return df


class DataValidator:
    """Validates and cleans market data."""
    
    @staticmethod
    def validate_stock_data(data: pd.DataFrame) -> bool:
        """Validate stock data quality."""
        if data.empty:
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            return False
        
        # Check for reasonable price ranges
        if (data['Close'] <= 0).any():
            return False
        
        # Check for missing data
        if data[required_columns].isnull().sum().sum() > len(data) * 0.1:
            return False
        
        return True
    
    @staticmethod
    def clean_options_data(options_df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter options data."""
        df = options_df.copy()
        
        # Remove options with zero volume and open interest
        df = df[(df['volume'].fillna(0) > 0) | (df['openInterest'].fillna(0) > 0)]
        
        # Remove options with unrealistic bid-ask spreads
        df['bid_ask_spread'] = df['ask'] - df['bid']
        df = df[df['bid_ask_spread'] <= df['lastPrice'] * 0.5]  # Max 50% spread
        
        # Remove options with zero or negative prices
        df = df[df['lastPrice'] > 0]
        
        return df


if __name__ == "__main__":
    # Test the data provider
    provider = MarketDataProvider()
    
    # Test stock data
    print("Testing stock data retrieval...")
    spy_data = provider.get_stock_data("SPY", "3mo")
    print(f"Retrieved {len(spy_data)} days of SPY data")
    print(spy_data[['Close', 'RSI', 'MACD', 'Historical_Vol']].tail())
    
    # Test options data
    print("\nTesting options data retrieval...")
    spy_options = provider.get_options_chain("SPY")
    print(f"Found {len(spy_options['calls'])} call options")
    print(f"Current SPY price: ${spy_options['current_price']:.2f}")