"""
Realistic mock data generator for development and demo.

Generates option chains, OHLC data, and simulated live ticks
with realistic characteristics (skew, term structure, volume distributions).
"""
from __future__ import annotations

import numpy as np
from datetime import date, datetime, timedelta
from ...models.options import (
    OptionContract, OptionChain, OptionType, Greeks, Quote,
    OHLCV, Position, OrderSide, VolatilityRegime,
)
from ..pricing.black_scholes import BlackScholesEngine


# Fixed seed for reproducible demo data
_RNG = np.random.default_rng(42)


# ─── Stock Universe ───────────────────────────────────────

MOCK_STOCKS = {
    "SPY": {"price": 587.32, "vol": 0.16, "div_yield": 0.013, "beta": 1.0},
    "QQQ": {"price": 512.75, "vol": 0.22, "div_yield": 0.006, "beta": 1.15},
    "AAPL": {"price": 237.45, "vol": 0.25, "div_yield": 0.005, "beta": 1.20},
    "MSFT": {"price": 448.90, "vol": 0.23, "div_yield": 0.007, "beta": 1.10},
    "NVDA": {"price": 138.60, "vol": 0.48, "div_yield": 0.0002, "beta": 1.65},
    "TSLA": {"price": 352.80, "vol": 0.55, "div_yield": 0.0, "beta": 1.80},
    "AMZN": {"price": 225.30, "vol": 0.30, "div_yield": 0.0, "beta": 1.25},
    "META": {"price": 695.40, "vol": 0.35, "div_yield": 0.003, "beta": 1.30},
    "JPM": {"price": 268.50, "vol": 0.22, "div_yield": 0.022, "beta": 1.05},
    "GLD": {"price": 265.15, "vol": 0.14, "div_yield": 0.0, "beta": 0.05},
}

bs = BlackScholesEngine()


def generate_option_chain(
    symbol: str = "SPY",
    expiry: date | None = None,
    risk_free_rate: float = 0.045,
) -> OptionChain:
    """Generate a realistic option chain with proper skew and Greeks."""
    stock = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])
    spot = stock["price"]
    base_vol = stock["vol"]

    if expiry is None:
        # Next monthly expiry (3rd Friday)
        today = date.today()
        expiry = _next_monthly_expiry(today)

    dte = (expiry - date.today()).days
    T = max(dte / 365.0, 1 / 365.0)

    # Generate strikes: ±20% from spot, rounded to standard increments
    increment = _strike_increment(spot)
    min_strike = round(spot * 0.80 / increment) * increment
    max_strike = round(spot * 1.20 / increment) * increment
    strikes = np.arange(min_strike, max_strike + increment, increment)

    calls = []
    puts = []

    for strike in strikes:
        moneyness = np.log(strike / spot)

        # Realistic IV with skew: higher for OTM puts (negative skew)
        skew_factor = -0.12 * moneyness  # Negative skew
        smile_factor = 0.04 * moneyness ** 2  # Smile curvature
        term_adj = 0.02 * (30 / max(dte, 1) - 1)  # Term structure effect
        noise = _RNG.normal(0, 0.005)

        iv = base_vol + skew_factor + smile_factor + term_adj + noise
        iv = max(iv, 0.05)  # Floor at 5%

        # Calculate prices using BS
        call_price = bs.price(spot, strike, T, risk_free_rate, iv, is_call=True)
        put_price = bs.price(spot, strike, T, risk_free_rate, iv, is_call=False)

        # Generate realistic bid-ask spreads
        call_spread = _bid_ask_spread(call_price, iv, dte)
        put_spread = _bid_ask_spread(put_price, iv, dte)

        # Volume and OI distributions (peak near ATM)
        atm_distance = abs(strike - spot) / spot
        volume_base = int(max(1, 5000 * np.exp(-20 * atm_distance ** 2)))
        oi_base = volume_base * _RNG.integers(3, 15)

        call_greeks = bs.greeks(spot, strike, T, risk_free_rate, iv, is_call=True)
        put_greeks = bs.greeks(spot, strike, T, risk_free_rate, iv, is_call=False)

        calls.append(OptionContract(
            symbol=f"{symbol}{expiry.strftime('%y%m%d')}C{int(strike*1000):08d}",
            underlying=symbol,
            strike=strike,
            expiry=expiry,
            option_type=OptionType.CALL,
            bid=round(max(call_price - call_spread / 2, 0.01), 2),
            ask=round(call_price + call_spread / 2, 2),
            last=round(call_price + _RNG.normal(0, call_spread * 0.2), 2),
            volume=int(volume_base * _RNG.uniform(0.5, 2.0)),
            open_interest=int(oi_base * _RNG.uniform(0.5, 2.0)),
            implied_volatility=round(iv * 100, 2),
            greeks=Greeks(
                delta=round(call_greeks.delta, 4),
                gamma=round(call_greeks.gamma, 6),
                theta=round(call_greeks.theta, 4),
                vega=round(call_greeks.vega, 4),
                rho=round(call_greeks.rho, 4),
            ),
            in_the_money=strike < spot,
        ))

        puts.append(OptionContract(
            symbol=f"{symbol}{expiry.strftime('%y%m%d')}P{int(strike*1000):08d}",
            underlying=symbol,
            strike=strike,
            expiry=expiry,
            option_type=OptionType.PUT,
            bid=round(max(put_price - put_spread / 2, 0.01), 2),
            ask=round(put_price + put_spread / 2, 2),
            last=round(put_price + _RNG.normal(0, put_spread * 0.2), 2),
            volume=int(volume_base * _RNG.uniform(0.3, 1.5)),
            open_interest=int(oi_base * _RNG.uniform(0.5, 1.5)),
            implied_volatility=round(iv * 100, 2),
            greeks=Greeks(
                delta=round(put_greeks.delta, 4),
                gamma=round(put_greeks.gamma, 6),
                theta=round(put_greeks.theta, 4),
                vega=round(put_greeks.vega, 4),
                rho=round(put_greeks.rho, 4),
            ),
            in_the_money=strike > spot,
        ))

    return OptionChain(
        underlying=symbol,
        underlying_price=spot,
        expiry=expiry,
        calls=sorted(calls, key=lambda c: c.strike),
        puts=sorted(puts, key=lambda p: p.strike),
        timestamp=datetime.utcnow(),
    )


def generate_historical_ohlcv(
    symbol: str = "SPY",
    days: int = 504,  # ~2 years
) -> list[OHLCV]:
    """Generate 2 years of realistic OHLCV data using GBM."""
    stock = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])
    initial_price = stock["price"] * 0.75  # Start lower for uptrend
    vol = stock["vol"]
    mu = 0.08  # Annual drift

    dt = 1 / 252
    prices = [initial_price]
    for _ in range(days - 1):
        ret = mu * dt + vol * np.sqrt(dt) * _RNG.normal()
        prices.append(prices[-1] * np.exp(ret))

    # Scale so last price matches current
    scale = stock["price"] / prices[-1]
    prices = [p * scale for p in prices]

    data = []
    start_date = date.today() - timedelta(days=days)

    for i, close in enumerate(prices):
        current_date = start_date + timedelta(days=i)
        # Skip weekends
        if current_date.weekday() >= 5:
            continue

        intraday_vol = vol * np.sqrt(dt)
        high = close * (1 + abs(_RNG.normal(0, intraday_vol)))
        low = close * (1 - abs(_RNG.normal(0, intraday_vol)))
        open_price = close * (1 + _RNG.normal(0, intraday_vol * 0.5))
        volume = int(_RNG.lognormal(17, 0.5))  # ~24M avg for SPY

        data.append(OHLCV(
            timestamp=datetime.combine(current_date, datetime.min.time()),
            open=round(open_price, 2),
            high=round(max(high, open_price, close), 2),
            low=round(min(low, open_price, close), 2),
            close=round(close, 2),
            volume=volume,
        ))

    return data


def generate_live_tick(symbol: str = "SPY") -> Quote:
    """Generate a single simulated tick for WebSocket streaming."""
    stock = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])
    spot = stock["price"]

    # Small random price movement
    change = _RNG.normal(0, spot * 0.0003)
    new_price = round(spot + change, 2)
    spread = round(spot * 0.0001, 2)  # Tight spread for liquid names

    # Update the stored price
    MOCK_STOCKS[symbol]["price"] = new_price

    return Quote(
        symbol=symbol,
        bid=round(new_price - spread, 2),
        ask=round(new_price + spread, 2),
        last=new_price,
        volume=_RNG.integers(100, 50000),
        timestamp=datetime.utcnow(),
    )


def generate_mock_positions(account_value: float = 100000.0) -> list[Position]:
    """Generate sample portfolio positions (Iron Condor on SPY)."""
    spy_price = MOCK_STOCKS["SPY"]["price"]
    expiry = _next_monthly_expiry(date.today())

    return [
        # Short Iron Condor: sell wings, buy protection
        Position(
            id="pos_1",
            symbol=f"SPY{expiry.strftime('%y%m%d')}P{int((spy_price-15)*1000):08d}",
            underlying="SPY",
            option_type=OptionType.PUT,
            strike=round(spy_price - 15, 0),
            expiry=expiry,
            side=OrderSide.SELL,
            quantity=5,
            entry_price=2.50,
        ),
        Position(
            id="pos_2",
            symbol=f"SPY{expiry.strftime('%y%m%d')}P{int((spy_price-25)*1000):08d}",
            underlying="SPY",
            option_type=OptionType.PUT,
            strike=round(spy_price - 25, 0),
            expiry=expiry,
            side=OrderSide.BUY,
            quantity=5,
            entry_price=1.20,
        ),
        Position(
            id="pos_3",
            symbol=f"SPY{expiry.strftime('%y%m%d')}C{int((spy_price+15)*1000):08d}",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=round(spy_price + 15, 0),
            expiry=expiry,
            side=OrderSide.SELL,
            quantity=5,
            entry_price=2.30,
        ),
        Position(
            id="pos_4",
            symbol=f"SPY{expiry.strftime('%y%m%d')}C{int((spy_price+25)*1000):08d}",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=round(spy_price + 25, 0),
            expiry=expiry,
            side=OrderSide.BUY,
            quantity=5,
            entry_price=1.10,
        ),
    ]


def generate_vol_surface_data(symbol: str = "SPY") -> list[dict]:
    """Generate mock option data suitable for vol surface construction."""
    stock = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])
    spot = stock["price"]
    base_vol = stock["vol"]
    today = date.today()

    contracts = []
    expiries_dte = [7, 14, 30, 45, 60, 90, 120, 180, 365]

    for dte in expiries_dte:
        expiry = today + timedelta(days=dte)
        T = dte / 365.0
        increment = _strike_increment(spot)
        strikes = np.arange(spot * 0.85, spot * 1.15, increment)

        for strike in strikes:
            moneyness = np.log(strike / spot)
            iv = base_vol - 0.12 * moneyness + 0.04 * moneyness**2 + _RNG.normal(0, 0.003)
            iv = max(iv, 0.05)

            for opt_type in ["call", "put"]:
                is_call = opt_type == "call"
                price = bs.price(spot, strike, T, 0.045, iv, is_call)
                spread = max(0.02, price * 0.02)

                contracts.append({
                    "strike": round(float(strike), 2),
                    "expiry": expiry.isoformat(),
                    "option_type": opt_type,
                    "bid": round(max(price - spread / 2, 0.01), 2),
                    "ask": round(price + spread / 2, 2),
                    "volume": int(_RNG.integers(10, 5000)),
                    "open_interest": int(_RNG.integers(100, 50000)),
                })

    return contracts


# ─── Helpers ──────────────────────────────────────────────

def _next_monthly_expiry(from_date: date) -> date:
    """Find the next monthly options expiry (3rd Friday)."""
    year, month = from_date.year, from_date.month
    if from_date.day > 20:
        month += 1
        if month > 12:
            month = 1
            year += 1

    # Find 3rd Friday
    first_day = date(year, month, 1)
    # Friday is weekday 4
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third_friday = first_friday + timedelta(weeks=2)
    return third_friday


def _strike_increment(spot: float) -> float:
    """Standard strike increment based on price level."""
    if spot < 50:
        return 0.5
    elif spot < 200:
        return 1.0
    elif spot < 500:
        return 5.0
    return 5.0


def _bid_ask_spread(price: float, iv: float, dte: int) -> float:
    """Realistic bid-ask spread based on option characteristics."""
    base_spread = max(0.01, price * 0.015)
    vol_adj = iv * 0.1
    dte_adj = max(0, (60 - dte) * 0.002)
    return round(base_spread + vol_adj + dte_adj, 2)
