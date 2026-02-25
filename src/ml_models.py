"""
Options Trading Algorithm - Machine Learning Models Module

Advanced ML models for price prediction, volatility forecasting, and signal enhancement.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
import joblib
import warnings
from typing import Dict, Tuple, List, Optional
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Advanced feature engineering for financial time series.
    """
    
    @staticmethod
    def create_ml_features(data: pd.DataFrame, lookback_periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Create comprehensive features for ML models.
        
        Args:
            data: DataFrame with market data and technical indicators
            lookback_periods: Periods for rolling features
        
        Returns:
            DataFrame with engineered features
        """
        df = data.copy()
        
        # Price-based features
        for period in lookback_periods:
            # Returns
            df[f'return_{period}d'] = df['Close'].pct_change(period)
            df[f'return_std_{period}d'] = df['Close'].pct_change().rolling(period).std()
            
            # Price ratios
            df[f'price_ratio_{period}d'] = df['Close'] / df['Close'].shift(period)
            df[f'high_low_ratio_{period}d'] = df['High'] / df['Low']
            
            # Volume features
            df[f'volume_ratio_{period}d'] = df['Volume'] / df['Volume'].rolling(period).mean()
            df[f'volume_price_trend_{period}d'] = (
                df['Close'].pct_change().rolling(period).sum() * 
                df['Volume'].rolling(period).sum()
            )
            
            # Volatility features
            df[f'volatility_ratio_{period}d'] = (
                df['Historical_Vol'] / df['Historical_Vol'].rolling(period).mean()
            )
        
        # Technical indicator features
        # RSI-based features
        df['rsi_velocity'] = df['RSI'].diff()
        df['rsi_acceleration'] = df['rsi_velocity'].diff()
        df['rsi_divergence'] = df['RSI'].rolling(10).corr(df['Close'])
        
        # MACD features
        df['macd_velocity'] = df['MACD'].diff()
        df['macd_signal_cross'] = ((df['MACD'] > df['MACD_Signal']) & 
                                   (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
        
        # Bollinger Bands features
        df['bb_squeeze'] = (df['BB_Width'] < df['BB_Width'].rolling(20).quantile(0.1)).astype(int)
        df['bb_expansion'] = (df['BB_Width'] > df['BB_Width'].rolling(20).quantile(0.9)).astype(int)
        
        # Support/Resistance proximity
        df['price_level_proximity'] = FeatureEngineer._calculate_level_proximity(df)
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Cyclical features
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['Volume'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['RSI'].shift(lag)
        
        return df
    
    @staticmethod
    def _calculate_level_proximity(data: pd.DataFrame) -> pd.Series:
        """Calculate proximity to significant price levels."""
        # Simplified calculation - in practice, you'd use more sophisticated S/R detection
        rolling_high = data['High'].rolling(20).max()
        rolling_low = data['Low'].rolling(20).min()
        
        proximity_high = (rolling_high - data['Close']) / data['Close']
        proximity_low = (data['Close'] - rolling_low) / data['Close']
        
        return np.minimum(proximity_high, proximity_low)
    
    @staticmethod
    def select_features(data: pd.DataFrame, target_col: str, max_features: int = 50) -> List[str]:
        """
        Select most important features using correlation and variance thresholds.
        
        Args:
            data: DataFrame with features
            target_col: Target column name
            max_features: Maximum number of features to select
        
        Returns:
            List of selected feature names
        """
        # Remove non-numeric columns and target
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col != target_col and not col.endswith('_target')]
        
        # Remove features with low variance
        feature_data = data[feature_cols].dropna()
        low_variance_features = feature_data.columns[feature_data.var() < 1e-10].tolist()
        feature_cols = [col for col in feature_cols if col not in low_variance_features]
        
        # Calculate correlation with target
        correlations = data[feature_cols + [target_col]].corr()[target_col].abs().sort_values(ascending=False)
        
        # Select top features
        selected_features = correlations.head(max_features).index.tolist()
        selected_features = [col for col in selected_features if col != target_col]
        
        return selected_features


class PricePredictionModel:
    """
    Ensemble model for price prediction using multiple algorithms.
    """
    
    def __init__(self, model_config: Dict = None):
        """
        Initialize the prediction model.
        
        Args:
            model_config: Configuration for model parameters
        """
        self.config = model_config or {
            'random_forest': {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            'gradient_boost': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42},
            'neural_network': {'hidden_layer_sizes': (100, 50), 'max_iter': 500, 'random_state': 42}
        }
        
        self.models = {}
        self.scalers = {}
        self.selected_features = []
        self.is_trained = False
    
    def prepare_data(self, data: pd.DataFrame, target_periods: List[int] = [1, 5, 10]) -> pd.DataFrame:
        """
        Prepare data for training with multiple prediction horizons.
        
        Args:
            data: DataFrame with features
            target_periods: Prediction horizons in days
        
        Returns:
            DataFrame with target variables
        """
        df = data.copy()
        
        # Create target variables for different horizons
        for period in target_periods:
            df[f'price_target_{period}d'] = df['Close'].shift(-period)
            df[f'return_target_{period}d'] = df['Close'].pct_change(-period)
            df[f'direction_target_{period}d'] = (df[f'return_target_{period}d'] > 0).astype(int)
        
        return df
    
    def train(self, data: pd.DataFrame, target_col: str = 'return_target_5d') -> Dict:
        """
        Train the ensemble model.
        
        Args:
            data: DataFrame with features and targets
            target_col: Target column name
        
        Returns:
            Dictionary with training metrics
        """
        print(f"Training model to predict {target_col}...")
        
        # Feature selection
        feature_engineer = FeatureEngineer()
        self.selected_features = feature_engineer.select_features(data, target_col)
        
        # Prepare training data
        X = data[self.selected_features].dropna()
        y = data.loc[X.index, target_col]
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:
            raise ValueError("Insufficient data for training (need at least 100 samples)")
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        split_results = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train models
            fold_models = {}
            fold_predictions = {}
            
            # Random Forest
            rf_model = RandomForestRegressor(**self.config['random_forest'])
            rf_model.fit(X_train_scaled, y_train)
            fold_models['rf'] = rf_model
            fold_predictions['rf'] = rf_model.predict(X_val_scaled)
            
            # Gradient Boosting
            gb_model = GradientBoostingRegressor(**self.config['gradient_boost'])
            gb_model.fit(X_train_scaled, y_train)
            fold_models['gb'] = gb_model
            fold_predictions['gb'] = gb_model.predict(X_val_scaled)
            
            # Neural Network
            nn_model = MLPRegressor(**self.config['neural_network'])
            nn_model.fit(X_train_scaled, y_train)
            fold_models['nn'] = nn_model
            fold_predictions['nn'] = nn_model.predict(X_val_scaled)
            
            # Ensemble prediction (simple average)
            ensemble_pred = np.mean(list(fold_predictions.values()), axis=0)
            
            # Calculate metrics
            fold_metrics = {
                'mae': mean_absolute_error(y_val, ensemble_pred),
                'mse': mean_squared_error(y_val, ensemble_pred),
                'r2': r2_score(y_val, ensemble_pred)
            }
            
            split_results.append(fold_metrics)
        
        # Train final models on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.models['rf'] = RandomForestRegressor(**self.config['random_forest'])
        self.models['rf'].fit(X_scaled, y)
        
        self.models['gb'] = GradientBoostingRegressor(**self.config['gradient_boost'])
        self.models['gb'].fit(X_scaled, y)
        
        self.models['nn'] = MLPRegressor(**self.config['neural_network'])
        self.models['nn'].fit(X_scaled, y)
        
        self.scalers['features'] = scaler
        self.is_trained = True
        
        # Calculate average metrics
        avg_metrics = {
            'mae': np.mean([split['mae'] for split in split_results]),
            'mse': np.mean([split['mse'] for split in split_results]),
            'r2': np.mean([split['r2'] for split in split_results]),
            'feature_count': len(self.selected_features)
        }
        
        print(f"Training complete. MAE: {avg_metrics['mae']:.6f}, R²: {avg_metrics['r2']:.4f}")
        return avg_metrics
    
    def predict(self, data: pd.DataFrame) -> Dict:
        """
        Make predictions using the trained ensemble.
        
        Args:
            data: DataFrame with features
        
        Returns:
            Dictionary with predictions and confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare features
        X = data[self.selected_features].iloc[-1:].fillna(method='ffill').fillna(0)
        X_scaled = self.scalers['features'].transform(X)
        
        # Get predictions from each model
        predictions = {}
        for model_name, model in self.models.items():
            predictions[model_name] = model.predict(X_scaled)[0]
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()))
        prediction_std = np.std(list(predictions.values()))
        
        # Calculate confidence (inverse of prediction variance)
        confidence = max(0, min(100, 100 * (1 - prediction_std / (abs(ensemble_pred) + 1e-6))))
        
        return {
            'prediction': ensemble_pred,
            'confidence': confidence,
            'individual_predictions': predictions,
            'prediction_std': prediction_std
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models."""
        if not self.is_trained:
            return pd.DataFrame()
        
        # Combine importance from RF and GB models
        rf_importance = self.models['rf'].feature_importances_
        gb_importance = self.models['gb'].feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.selected_features,
            'rf_importance': rf_importance,
            'gb_importance': gb_importance,
            'avg_importance': (rf_importance + gb_importance) / 2
        }).sort_values('avg_importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'selected_features': self.selected_features,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.models = model_data['models']
        self.scalers = model_data['scalers']
        self.selected_features = model_data['selected_features']
        self.config = model_data['config']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")


class VolatilityForecaster:
    """
    Specialized model for volatility forecasting using GARCH-like approaches.
    """
    
    def __init__(self):
        """Initialize volatility forecaster."""
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features specifically for volatility forecasting.
        
        Args:
            data: DataFrame with market data
        
        Returns:
            DataFrame with volatility features
        """
        df = data.copy()
        
        # Historical volatility features
        for window in [5, 10, 20, 30]:
            df[f'realized_vol_{window}'] = df['Returns'].rolling(window).std() * np.sqrt(252)
            df[f'vol_ratio_{window}'] = df[f'realized_vol_{window}'] / df[f'realized_vol_{window}'].rolling(60).mean()
        
        # Range-based volatility (Parkinson estimator)
        df['parkinson_vol'] = np.sqrt(
            (1/(4*np.log(2))) * np.log(df['High']/df['Low'])**2
        ) * np.sqrt(252)
        
        # Volume-based features
        df['volume_volatility'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
        
        # Price gap features
        df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
        df['gap_volatility'] = df['gap'].rolling(20).std()
        
        # Target: next day's volatility
        df['vol_target'] = df['realized_vol_20'].shift(-1)
        
        return df
    
    def train(self, data: pd.DataFrame) -> Dict:
        """
        Train volatility forecasting model.
        
        Args:
            data: DataFrame with volatility features
        
        Returns:
            Training metrics
        """
        vol_data = self.prepare_volatility_features(data)
        
        feature_cols = [col for col in vol_data.columns 
                       if 'vol' in col and 'target' not in col]
        feature_cols.extend(['volume_volatility', 'gap_volatility'])
        
        X = vol_data[feature_cols].dropna()
        y = vol_data.loc[X.index, 'vol_target']
        
        # Remove NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            raise ValueError("Insufficient data for volatility model training")
        
        # Train-test split
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale and train
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.feature_cols = feature_cols
        self.is_trained = True
        
        return metrics
    
    def predict_volatility(self, data: pd.DataFrame) -> float:
        """Predict next period volatility."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        vol_data = self.prepare_volatility_features(data)
        X = vol_data[self.feature_cols].iloc[-1:].fillna(method='ffill').fillna(0)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)[0]


if __name__ == "__main__":
    # Test the ML models
    from data_provider import MarketDataProvider
    from technical_indicators import MomentumIndicators
    
    print("Testing ML models...")
    
    provider = MarketDataProvider()
    data = provider.get_stock_data("AAPL", "1y")
    
    if not data.empty:
        # Add technical indicators
        momentum = MomentumIndicators()
        data = momentum.calculate_momentum_score(data)
        data = momentum.detect_trend_strength(data)
        
        # Create features
        engineer = FeatureEngineer()
        data = engineer.create_ml_features(data)
        
        # Train price prediction model
        price_model = PricePredictionModel()
        data = price_model.prepare_data(data)
        
        try:
            metrics = price_model.train(data, 'return_target_5d')
            print(f"Price model trained - R²: {metrics['r2']:.4f}")
            
            # Make prediction
            prediction = price_model.predict(data)
            print(f"Latest prediction: {prediction['prediction']:.4f} (confidence: {prediction['confidence']:.1f}%)")
            
            # Feature importance
            importance = price_model.get_feature_importance()
            print(f"Top 5 features: {importance.head()['feature'].tolist()}")
            
        except Exception as e:
            print(f"Error training price model: {e}")
        
        # Train volatility model
        vol_model = VolatilityForecaster()
        try:
            vol_metrics = vol_model.train(data)
            print(f"Volatility model trained - R²: {vol_metrics['r2']:.4f}")
            
            vol_pred = vol_model.predict_volatility(data)
            print(f"Predicted volatility: {vol_pred:.2f}%")
            
        except Exception as e:
            print(f"Error training volatility model: {e}")
    
    else:
        print("No data available for testing")