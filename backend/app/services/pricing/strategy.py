"""
Strategy analysis engine.

Calculates payoff diagrams, breakeven points, and time-decay
visualization for multi-leg option strategies.
"""

import numpy as np
from datetime import date
from ...models.options import (
    StrategyLeg, StrategyAnalysis, Greeks, OrderSide, OptionType,
    StrategyAnalysisRequest,
)
from .black_scholes import BlackScholesEngine


class StrategyAnalyzer:
    """Analyze multi-leg option strategies with payoff diagrams and Greeks."""

    def __init__(self, risk_free_rate: float = 0.045):
        self.r = risk_free_rate
        self.bs = BlackScholesEngine()

    def analyze(
        self,
        legs: list[StrategyLeg],
        spot: float,
        volatility: float = 0.20,
        price_range_pct: float = 20.0,
        time_steps: int = 5,
    ) -> StrategyAnalysis:
        """
        Full strategy analysis with payoff at multiple time points.

        Returns payoff data suitable for charting with time slider.
        """
        today = date.today()

        # Price range for payoff diagram
        price_min = spot * (1 - price_range_pct / 100)
        price_max = spot * (1 + price_range_pct / 100)
        prices = np.linspace(price_min, price_max, 200)

        # Calculate net premium
        net_premium = sum(
            leg.premium * leg.quantity * (1 if leg.side == OrderSide.SELL else -1)
            for leg in legs
        )

        # Find max DTE across legs
        max_dte = max((leg.expiry - today).days for leg in legs)

        # Generate payoff curves at different time points
        payoff_data = []
        time_fractions = np.linspace(0, 1, time_steps)

        for t_frac in time_fractions:
            days_elapsed = int(t_frac * max_dte)
            days_remaining = max_dte - days_elapsed
            label = f"T-{days_remaining}d" if days_remaining > 0 else "Expiry"

            curve = {"label": label, "days_remaining": days_remaining, "points": []}

            for underlying_price in prices:
                total_pnl = net_premium * 100  # Start with premium received/paid

                for leg in legs:
                    leg_dte = (leg.expiry - today).days - days_elapsed
                    T = max(leg_dte / 365.0, 0.0)
                    is_call = leg.option_type == OptionType.CALL
                    multiplier = leg.quantity * (1 if leg.side == OrderSide.BUY else -1)

                    if T <= 0:
                        # At expiry: intrinsic value only
                        if is_call:
                            value = max(underlying_price - leg.strike, 0)
                        else:
                            value = max(leg.strike - underlying_price, 0)
                    else:
                        # Before expiry: Black-Scholes value
                        value = self.bs.price(
                            underlying_price, leg.strike, T, self.r, volatility, is_call
                        )

                    total_pnl += multiplier * value * 100

                # Subtract what we paid for long positions
                for leg in legs:
                    if leg.side == OrderSide.BUY:
                        total_pnl -= leg.premium * leg.quantity * 100

                curve["points"].append({
                    "price": round(float(underlying_price), 2),
                    "pnl": round(float(total_pnl), 2),
                })

            payoff_data.append(curve)

        # Calculate at-expiry metrics
        expiry_payoffs = np.array([p["pnl"] for p in payoff_data[-1]["points"]])
        max_profit = float(np.max(expiry_payoffs))
        max_loss = float(np.min(expiry_payoffs))

        # Breakeven points (where PnL crosses zero)
        breakevens = self._find_breakevens(
            prices, expiry_payoffs
        )

        # Total Greeks at current spot
        total_greeks = self._compute_total_greeks(legs, spot, volatility, today)

        return StrategyAnalysis(
            legs=legs,
            underlying_price=spot,
            max_profit=round(max_profit, 2) if max_profit < 1e6 else None,
            max_loss=round(max_loss, 2) if max_loss > -1e6 else None,
            breakeven_points=breakevens,
            payoff_data=[
                {
                    "label": curve["label"],
                    "days_remaining": curve["days_remaining"],
                    "points": curve["points"],
                }
                for curve in payoff_data
            ],
            total_greeks=total_greeks,
            net_premium=round(net_premium, 2),
        )

    def _find_breakevens(self, prices: np.ndarray, payoffs: np.ndarray) -> list[float]:
        """Find price points where payoff crosses zero."""
        breakevens = []
        for i in range(1, len(payoffs)):
            if payoffs[i - 1] * payoffs[i] < 0:  # Sign change
                # Linear interpolation
                x = prices[i - 1] + (0 - payoffs[i - 1]) * (prices[i] - prices[i - 1]) / (payoffs[i] - payoffs[i - 1])
                breakevens.append(round(float(x), 2))
        return breakevens

    def _compute_total_greeks(
        self,
        legs: list[StrategyLeg],
        spot: float,
        volatility: float,
        today: date,
    ) -> Greeks:
        """Compute aggregate Greeks for the strategy."""
        total = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0, "rho": 0}

        for leg in legs:
            dte = (leg.expiry - today).days
            T = max(dte / 365.0, 1 / 365.0)
            is_call = leg.option_type == OptionType.CALL
            multiplier = leg.quantity * (1 if leg.side == OrderSide.BUY else -1) * 100

            g = self.bs.greeks(spot, leg.strike, T, self.r, volatility, is_call)
            total["delta"] += g.delta * multiplier
            total["gamma"] += g.gamma * multiplier
            total["theta"] += g.theta * multiplier
            total["vega"] += g.vega * multiplier
            total["rho"] += g.rho * multiplier

        return Greeks(**{k: round(v, 4) for k, v in total.items()})


# ─── Pre-built Strategies ────────────────────────────────

def iron_condor_legs(
    spot: float,
    expiry: date,
    wing_width: float = 10.0,
    protection_width: float = 10.0,
) -> list[StrategyLeg]:
    """Generate Iron Condor legs."""
    return [
        StrategyLeg(
            option_type=OptionType.PUT,
            strike=round(spot - wing_width - protection_width),
            expiry=expiry,
            side=OrderSide.BUY,
            quantity=1,
            premium=0.80,
        ),
        StrategyLeg(
            option_type=OptionType.PUT,
            strike=round(spot - wing_width),
            expiry=expiry,
            side=OrderSide.SELL,
            quantity=1,
            premium=1.80,
        ),
        StrategyLeg(
            option_type=OptionType.CALL,
            strike=round(spot + wing_width),
            expiry=expiry,
            side=OrderSide.SELL,
            quantity=1,
            premium=1.70,
        ),
        StrategyLeg(
            option_type=OptionType.CALL,
            strike=round(spot + wing_width + protection_width),
            expiry=expiry,
            side=OrderSide.BUY,
            quantity=1,
            premium=0.75,
        ),
    ]


def bull_call_spread_legs(
    spot: float,
    expiry: date,
    width: float = 10.0,
) -> list[StrategyLeg]:
    """Generate Bull Call Spread legs."""
    return [
        StrategyLeg(
            option_type=OptionType.CALL,
            strike=round(spot),
            expiry=expiry,
            side=OrderSide.BUY,
            quantity=1,
            premium=5.50,
        ),
        StrategyLeg(
            option_type=OptionType.CALL,
            strike=round(spot + width),
            expiry=expiry,
            side=OrderSide.SELL,
            quantity=1,
            premium=2.20,
        ),
    ]


def straddle_legs(
    spot: float,
    expiry: date,
) -> list[StrategyLeg]:
    """Generate Long Straddle legs."""
    atm = round(spot)
    return [
        StrategyLeg(
            option_type=OptionType.CALL,
            strike=atm,
            expiry=expiry,
            side=OrderSide.BUY,
            quantity=1,
            premium=6.00,
        ),
        StrategyLeg(
            option_type=OptionType.PUT,
            strike=atm,
            expiry=expiry,
            side=OrderSide.BUY,
            quantity=1,
            premium=5.50,
        ),
    ]
