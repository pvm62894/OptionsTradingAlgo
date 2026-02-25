"""Tests for Black-Scholes pricing engine."""

import pytest
import numpy as np
from app.services.pricing.black_scholes import BlackScholesEngine, BinomialTreePricer


bs = BlackScholesEngine()

# Standard test parameters
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestBlackScholes:
    def test_call_price_positive(self):
        price = bs.price(S, K, T, r, sigma, is_call=True)
        assert price > 0, "Call price should be positive"

    def test_put_price_positive(self):
        price = bs.price(S, K, T, r, sigma, is_call=False)
        assert price > 0, "Put price should be positive"

    def test_put_call_parity(self):
        """C - P = S - K * exp(-rT)"""
        call = bs.price(S, K, T, r, sigma, is_call=True)
        put = bs.price(S, K, T, r, sigma, is_call=False)
        parity_rhs = S - K * np.exp(-r * T)
        assert abs((call - put) - parity_rhs) < 1e-6, f"Put-call parity violated: {call - put} != {parity_rhs}"

    def test_atm_call_delta_near_half(self):
        g = bs.greeks(S, K, T, r, sigma, is_call=True)
        assert 0.45 < g.delta < 0.65, f"ATM call delta should be near 0.5, got {g.delta}"

    def test_put_delta_negative(self):
        g = bs.greeks(S, K, T, r, sigma, is_call=False)
        assert g.delta < 0, f"Put delta should be negative, got {g.delta}"

    def test_gamma_positive(self):
        g = bs.greeks(S, K, T, r, sigma, is_call=True)
        assert g.gamma > 0, "Gamma should be positive"

    def test_gamma_same_for_call_and_put(self):
        call_g = bs.greeks(S, K, T, r, sigma, is_call=True)
        put_g = bs.greeks(S, K, T, r, sigma, is_call=False)
        assert abs(call_g.gamma - put_g.gamma) < 1e-10, "Gamma should be equal for calls and puts"

    def test_theta_negative_for_long(self):
        g = bs.greeks(S, K, T, r, sigma, is_call=True)
        assert g.theta < 0, f"Call theta should be negative for ATM, got {g.theta}"

    def test_vega_positive(self):
        g = bs.greeks(S, K, T, r, sigma, is_call=True)
        assert g.vega > 0, "Vega should be positive"

    def test_deep_itm_call_delta_near_one(self):
        g = bs.greeks(100, 60, 0.5, r, sigma, is_call=True)
        assert g.delta > 0.95, f"Deep ITM call delta should be near 1, got {g.delta}"

    def test_deep_otm_call_delta_near_zero(self):
        g = bs.greeks(100, 150, 0.1, r, sigma, is_call=True)
        assert g.delta < 0.05, f"Deep OTM call delta should be near 0, got {g.delta}"

    def test_expired_call_intrinsic(self):
        price = bs.price(105, 100, 0, r, sigma, is_call=True)
        assert abs(price - 5.0) < 1e-10

    def test_expired_otm_call_zero(self):
        price = bs.price(95, 100, 0, r, sigma, is_call=True)
        assert abs(price) < 1e-10


class TestImpliedVolatility:
    def test_iv_roundtrip(self):
        """Price with known vol, then solve for IV. Should match."""
        target_vol = 0.25
        price = bs.price(S, K, T, r, target_vol, is_call=True)
        iv = bs.implied_volatility(price, S, K, T, r, is_call=True)
        assert iv is not None
        assert abs(iv - target_vol) < 1e-6, f"IV roundtrip failed: {iv} != {target_vol}"

    def test_iv_put_roundtrip(self):
        target_vol = 0.30
        price = bs.price(S, K, T, r, target_vol, is_call=False)
        iv = bs.implied_volatility(price, S, K, T, r, is_call=False)
        assert iv is not None
        assert abs(iv - target_vol) < 1e-6

    def test_iv_otm_option(self):
        target_vol = 0.25  # Use higher vol so the option has meaningful value
        price = bs.price(S, 110, 0.5, r, target_vol, is_call=True)
        iv = bs.implied_volatility(price, S, 110, 0.5, r, is_call=True)
        assert iv is not None
        assert abs(iv - target_vol) < 1e-4


class TestBinomialTree:
    def test_american_put_geq_european(self):
        """American put should be >= European put (early exercise premium)."""
        eu_put = bs.price(S, K, T, r, sigma, is_call=False)
        am_put = BinomialTreePricer.price(S, K, T, r, sigma, is_call=False, steps=200)
        assert am_put >= eu_put - 0.01, f"American put {am_put} should be >= European {eu_put}"

    def test_american_call_no_dividend_equals_european(self):
        """Without dividends, American call = European call."""
        eu_call = bs.price(S, K, T, r, sigma, is_call=True)
        am_call = BinomialTreePricer.price(S, K, T, r, sigma, is_call=True, steps=200)
        assert abs(am_call - eu_call) < 0.5, f"American call {am_call} should ≈ European {eu_call}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
