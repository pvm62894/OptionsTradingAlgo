"""
Options Trading Algorithm - System Test

Comprehensive test of the momentum + ML options trading system.
"""

import sys
import os
sys.path.append('src')

from data_provider import MarketDataProvider, DataValidator
from technical_indicators import MomentumIndicators, SignalGenerator
from ml_models import PricePredictionModel, VolatilityForecaster, FeatureEngineer
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class SystemTester:
    """Test the complete options trading system."""
    
    def __init__(self, symbol: str = "SPY"):
        """Initialize system tester."""
        self.symbol = symbol
        self.data_provider = MarketDataProvider()
        self.momentum_indicators = MomentumIndicators()
        self.signal_generator = SignalGenerator()
        self.price_model = PricePredictionModel()
        self.vol_forecaster = VolatilityForecaster()
        self.feature_engineer = FeatureEngineer()
        
    def run_comprehensive_test(self):
        """Run complete system test."""
        print("🚀 Starting Options Trading Algorithm System Test")
        print("=" * 60)
        
        # Step 1: Data Acquisition
        print("📊 Step 1: Data Acquisition")
        print("-" * 30)
        
        stock_data = self.data_provider.get_stock_data(self.symbol, "6mo")
        if stock_data.empty:
            print("❌ Failed to retrieve stock data")
            return
        
        print(f"✅ Retrieved {len(stock_data)} days of {self.symbol} data")
        print(f"   Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")
        print(f"   Price range: ${stock_data['Close'].min():.2f} - ${stock_data['Close'].max():.2f}")
        
        # Get options data
        options_data = self.data_provider.get_options_chain(self.symbol)
        print(f"✅ Retrieved options chain: {len(options_data['calls'])} calls, {len(options_data['puts'])} puts")
        
        # Step 2: Technical Analysis
        print(f"\n📈 Step 2: Technical Analysis")
        print("-" * 30)
        
        # Calculate momentum indicators
        stock_data = self.momentum_indicators.calculate_momentum_score(stock_data)
        stock_data = self.momentum_indicators.detect_trend_strength(stock_data)
        stock_data = self.momentum_indicators.calculate_volatility_indicators(stock_data)
        
        latest_data = stock_data.iloc[-1]
        print(f"✅ Current Price: ${latest_data['Close']:.2f}")
        print(f"   Momentum Score: {latest_data['momentum_score_normalized']:.2f}")
        print(f"   Trend Strength: {latest_data['trend_strength']:.2f}")
        print(f"   RSI: {latest_data['RSI']:.1f}")
        print(f"   Historical Volatility: {latest_data['Historical_Vol']*100:.1f}%")
        print(f"   Volatility Regime: {latest_data['vol_regime']}")
        
        # Generate signals
        stock_data = self.signal_generator.generate_signals(stock_data)
        latest_data = stock_data.iloc[-1]  # Update latest_data after signal generation
        print(f"   Final Signal: {latest_data['final_signal']} (Strength: {latest_data['signal_strength']:.2f})")
        
        # Identify opportunities
        opportunities = self.signal_generator.identify_options_opportunities(stock_data)
        print(f"✅ Identified {len(opportunities)} trading opportunities:")
        for opp in opportunities:
            print(f"   • {opp['strategy']}: {opp['reason']} (Confidence: {opp['confidence']:.1f}%)")
        
        # Step 3: Machine Learning Analysis
        print(f"\n🤖 Step 3: Machine Learning Analysis")
        print("-" * 30)
        
        # Feature engineering
        stock_data = self.feature_engineer.create_ml_features(stock_data)
        print(f"✅ Created {len([col for col in stock_data.columns if any(x in col for x in ['return_', 'volume_', 'rsi_', 'macd_'])])} ML features")
        
        # Train price prediction model
        try:
            stock_data = self.price_model.prepare_data(stock_data, [1, 5, 10])
            metrics = self.price_model.train(stock_data, 'return_target_5d')
            
            print(f"✅ Price Prediction Model Trained:")
            print(f"   MAE: {metrics['mae']:.6f}")
            print(f"   R²: {metrics['r2']:.4f}")
            print(f"   Features: {metrics['feature_count']}")
            
            # Make prediction
            prediction = self.price_model.predict(stock_data)
            print(f"   5-day Return Prediction: {prediction['prediction']:.2%}")
            print(f"   Confidence: {prediction['confidence']:.1f}%")
            
            # Show feature importance
            importance = self.price_model.get_feature_importance().head(5)
            print("   Top Features:")
            for _, row in importance.iterrows():
                print(f"     • {row['feature']}: {row['avg_importance']:.3f}")
                
        except Exception as e:
            print(f"⚠️ Price model training failed: {e}")
        
        # Train volatility forecaster
        try:
            vol_metrics = self.vol_forecaster.train(stock_data)
            print(f"✅ Volatility Forecaster Trained:")
            print(f"   MAE: {vol_metrics['mae']:.6f}")
            print(f"   R²: {vol_metrics['r2']:.4f}")
            
            vol_prediction = self.vol_forecaster.predict_volatility(stock_data)
            print(f"   Volatility Prediction: {vol_prediction:.2%}")
            
        except Exception as e:
            print(f"⚠️ Volatility model training failed: {e}")
        
        # Step 4: Options Analysis
        print(f"\n📋 Step 4: Options Analysis")
        print("-" * 30)
        
        if not options_data['calls'].empty:
            # Analyze options chain
            calls = options_data['calls']
            puts = options_data['puts']
            current_price = latest_data['Close']
            
            # Find ATM options
            atm_calls = calls.iloc[(calls['strike'] - current_price).abs().argsort()[:3]]
            atm_puts = puts.iloc[(puts['strike'] - current_price).abs().argsort()[:3]]
            
            print(f"✅ Options Analysis (Current Price: ${current_price:.2f}):")
            print("   ATM Call Options:")
            for _, option in atm_calls.iterrows():
                print(f"     • Strike ${option['strike']:.0f}: ${option['lastPrice']:.2f} "
                     f"(IV: {option['impliedVolatility']*100:.1f}%, Volume: {option['volume'] or 0})")
            
            print("   ATM Put Options:")
            for _, option in atm_puts.iterrows():
                print(f"     • Strike ${option['strike']:.0f}: ${option['lastPrice']:.2f} "
                     f"(IV: {option['impliedVolatility']*100:.1f}%, Volume: {option['volume'] or 0})")
        
        # Step 5: Strategy Recommendations
        print(f"\n💡 Step 5: Strategy Recommendations")
        print("-" * 30)
        
        self.generate_strategy_recommendations(stock_data, options_data, opportunities)
        
        print(f"\n✅ System Test Complete!")
        print("=" * 60)
    
    def generate_strategy_recommendations(self, stock_data, options_data, opportunities):
        """Generate final strategy recommendations."""
        latest = stock_data.iloc[-1]
        
        print("🎯 Recommended Trading Strategies:")
        
        if opportunities:
            for i, opp in enumerate(opportunities, 1):
                print(f"\n{i}. {opp['strategy'].upper().replace('_', ' ')}")
                print(f"   Rationale: {opp['reason']}")
                print(f"   Confidence: {opp['confidence']:.1f}%")
                
                # Add specific recommendations based on strategy type
                if 'straddle' in opp['strategy']:
                    print("   Implementation: Buy ATM call + ATM put")
                    print("   Risk: Limited to premium paid")
                    print("   Reward: Unlimited (both directions)")
                elif 'call' in opp['strategy']:
                    print("   Implementation: Buy OTM calls or sell OTM puts")
                    print("   Risk: Premium paid (calls) or assignment risk (puts)")
                    print("   Reward: Upside participation")
                elif 'put' in opp['strategy']:
                    print("   Implementation: Buy OTM puts or sell OTM calls")
                    print("   Risk: Premium paid (puts) or assignment risk (calls)")
                    print("   Reward: Downside protection/profit")
                elif 'condor' in opp['strategy']:
                    print("   Implementation: Sell ATM straddle + buy OTM strangle")
                    print("   Risk: Limited to max spread width")
                    print("   Reward: Premium collected")
        
        else:
            print("   No clear opportunities identified at current market conditions")
            print("   Consider waiting for better setups or implementing market-neutral strategies")
        
        # Risk management guidelines
        print(f"\n⚠️ Risk Management Guidelines:")
        print("   • Position size: Max 2-5% of portfolio per trade")
        print("   • Time decay: Avoid options with <15 days to expiry")
        print("   • Liquidity: Only trade options with volume >50 and tight spreads")
        print("   • Stop loss: Set at 50% of premium paid for long options")
        print("   • Profit taking: Consider closing at 100-200% profit")


def main():
    """Run the system test."""
    tester = SystemTester("AAPL")  # Test with Apple stock
    tester.run_comprehensive_test()


if __name__ == "__main__":
    main()