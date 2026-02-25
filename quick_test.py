"""
Quick test with more data to demonstrate ML capabilities.
"""

import sys
sys.path.append('src')

from data_provider import MarketDataProvider
from technical_indicators import MomentumIndicators, SignalGenerator
from ml_models import PricePredictionModel, VolatilityForecaster, FeatureEngineer


def quick_test():
    """Run a quick test with more data."""
    print("🚀 Quick Options Trading Test - Extended Data")
    print("=" * 50)
    
    # Get more data
    provider = MarketDataProvider()
    data = provider.get_stock_data("SPY", "2y", "1d")  # 2 years of data
    
    if data.empty:
        print("❌ Failed to get data")
        return
    
    print(f"✅ Retrieved {len(data)} days of SPY data")
    
    # Add technical indicators
    momentum = MomentumIndicators()
    data = momentum.calculate_momentum_score(data)
    data = momentum.detect_trend_strength(data)
    data = momentum.calculate_volatility_indicators(data)
    
    # Generate signals
    signal_gen = SignalGenerator()
    data = signal_gen.generate_signals(data)
    
    latest = data.iloc[-1]
    print(f"\n📊 Current Analysis:")
    print(f"   Price: ${latest['Close']:.2f}")
    print(f"   Momentum: {latest['momentum_score_normalized']:.2f}")
    print(f"   Signal: {latest['final_signal']} (Strength: {latest['signal_strength']:.2f})")
    print(f"   Vol Regime: {latest['vol_regime']}")
    
    # Test ML models
    feature_eng = FeatureEngineer()
    data = feature_eng.create_ml_features(data)
    
    # Price prediction
    price_model = PricePredictionModel()
    data = price_model.prepare_data(data)
    
    try:
        metrics = price_model.train(data)
        print(f"\n🤖 ML Model Results:")
        print(f"   Training R²: {metrics['r2']:.4f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        
        prediction = price_model.predict(data)
        print(f"   5-day prediction: {prediction['prediction']:.4f}")
        print(f"   Confidence: {prediction['confidence']:.1f}%")
        
    except Exception as e:
        print(f"❌ ML training failed: {e}")
    
    # Volatility forecast
    vol_model = VolatilityForecaster()
    try:
        vol_metrics = vol_model.train(data)
        vol_pred = vol_model.predict_volatility(data)
        print(f"   Volatility forecast: {vol_pred:.2%}")
        print(f"   Vol model R²: {vol_metrics['r2']:.4f}")
    except Exception as e:
        print(f"❌ Vol model failed: {e}")
    
    # Find opportunities
    opportunities = signal_gen.identify_options_opportunities(data)
    if opportunities:
        print(f"\n💡 Trading Opportunities:")
        for opp in opportunities:
            print(f"   • {opp['strategy']}: {opp['confidence']:.1f}% confidence")
    
    print(f"\n✅ Test Complete!")


if __name__ == "__main__":
    quick_test()