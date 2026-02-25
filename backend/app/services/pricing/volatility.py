"""
Volatility surface construction and analysis.

Builds IV surfaces from market data, calculates IV rank/percentile,
and provides skew/term structure analytics.
"""
from __future__ import annotations

import numpy as np
from datetime import date, datetime, timedelta
from dataclasses import dataclass
from ..pricing.black_scholes import BlackScholesEngine
from ...models.options import VolSurfacePoint, VolatilitySurface


@dataclass
class SkewMetrics:
    put_25d_iv: float
    call_25d_iv: float
    atm_iv: float
    skew: float  # 25Δ Put IV - 25Δ Call IV
    butterfly: float  # (25Δ Put IV + 25Δ Call IV) / 2 - ATM IV
    risk_reversal: float  # 25Δ Call IV - 25Δ Put IV


@dataclass
class TermStructure:
    short_term_iv: float  # 30D
    mid_term_iv: float  # 60D
    long_term_iv: float  # 90D+
    ratio_30_90: float  # Contango/Backwardation indicator
    is_inverted: bool


class VolatilitySurfaceBuilder:
    """Constructs and analyzes implied volatility surfaces."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.bs = BlackScholesEngine()

    def build_surface(
        self,
        underlying: str,
        spot: float,
        chains: list[dict],
        historical_ivs: list[float] | None = None,
    ) -> VolatilitySurface:
        """
        Build IV surface from option chain data.

        Args:
            underlying: Ticker symbol
            spot: Current spot price
            chains: List of option contracts with fields:
                {strike, expiry, option_type, bid, ask, volume, open_interest}
            historical_ivs: Historical IV values for rank/percentile calc
        """
        points = []
        today = date.today()

        for contract in chains:
            strike = contract["strike"]
            expiry = contract["expiry"] if isinstance(contract["expiry"], date) else date.fromisoformat(contract["expiry"])
            is_call = contract["option_type"] == "call"
            mid_price = (contract["bid"] + contract["ask"]) / 2

            if mid_price <= 0 or contract["bid"] <= 0:
                continue

            dte = (expiry - today).days
            if dte <= 0:
                continue

            T = dte / 365.0
            moneyness = strike / spot

            # Skip deep ITM/OTM (noisy IVs)
            if moneyness < 0.7 or moneyness > 1.3:
                continue

            iv = self.bs.implied_volatility(mid_price, spot, strike, T, self.r, is_call)
            if iv is None or iv <= 0.01 or iv > 3.0:
                continue

            points.append(VolSurfacePoint(
                strike=strike,
                expiry=expiry,
                days_to_expiry=dte,
                iv=round(iv * 100, 2),  # Store as percentage
                moneyness=round(moneyness, 4),
            ))

        # Calculate IV rank and percentile
        iv_rank = 50.0
        iv_percentile = 50.0
        if historical_ivs and len(historical_ivs) > 20:
            atm_points = [p for p in points if 0.97 < p.moneyness < 1.03]
            if atm_points:
                current_atm_iv = np.mean([p.iv for p in atm_points])
                sorted_hist = sorted(historical_ivs)
                iv_min = sorted_hist[0]
                iv_max = sorted_hist[-1]
                if iv_max > iv_min:
                    iv_rank = ((current_atm_iv - iv_min) / (iv_max - iv_min)) * 100
                iv_percentile = (np.searchsorted(sorted_hist, current_atm_iv) / len(sorted_hist)) * 100

        return VolatilitySurface(
            underlying=underlying,
            spot_price=spot,
            points=points,
            iv_rank=round(iv_rank, 1),
            iv_percentile=round(iv_percentile, 1),
            timestamp=datetime.utcnow(),
        )

    def compute_skew(self, surface: VolatilitySurface, target_dte: int = 30) -> SkewMetrics | None:
        """Calculate volatility skew metrics for a given expiry."""
        # Filter points near target DTE
        tolerance = max(5, target_dte * 0.2)
        nearby = [p for p in surface.points if abs(p.days_to_expiry - target_dte) < tolerance]

        if len(nearby) < 5:
            return None

        # ATM IV (moneyness ~ 1.0)
        atm_points = [p for p in nearby if 0.98 < p.moneyness < 1.02]
        if not atm_points:
            return None
        atm_iv = np.mean([p.iv for p in atm_points])

        # 25-delta approximate moneyness: ~0.93 for puts, ~1.07 for calls
        put_25d = [p for p in nearby if 0.91 < p.moneyness < 0.95]
        call_25d = [p for p in nearby if 1.05 < p.moneyness < 1.09]

        put_25d_iv = np.mean([p.iv for p in put_25d]) if put_25d else atm_iv
        call_25d_iv = np.mean([p.iv for p in call_25d]) if call_25d else atm_iv

        skew = put_25d_iv - call_25d_iv
        butterfly = (put_25d_iv + call_25d_iv) / 2 - atm_iv
        risk_reversal = call_25d_iv - put_25d_iv

        return SkewMetrics(
            put_25d_iv=round(put_25d_iv, 2),
            call_25d_iv=round(call_25d_iv, 2),
            atm_iv=round(atm_iv, 2),
            skew=round(skew, 2),
            butterfly=round(butterfly, 2),
            risk_reversal=round(risk_reversal, 2),
        )

    def compute_term_structure(self, surface: VolatilitySurface) -> TermStructure | None:
        """Calculate term structure from ATM options across expiries."""
        atm = [p for p in surface.points if 0.97 < p.moneyness < 1.03]
        if len(atm) < 3:
            return None

        def avg_iv_near_dte(target: int, tol: int = 10) -> float | None:
            pts = [p.iv for p in atm if abs(p.days_to_expiry - target) < tol]
            return np.mean(pts) if pts else None

        short = avg_iv_near_dte(30, 15)
        mid = avg_iv_near_dte(60, 15)
        long = avg_iv_near_dte(90, 30)

        if short is None or long is None:
            return None

        mid = mid or (short + long) / 2

        return TermStructure(
            short_term_iv=round(short, 2),
            mid_term_iv=round(mid, 2),
            long_term_iv=round(long, 2),
            ratio_30_90=round(short / long, 3) if long > 0 else 1.0,
            is_inverted=short > long,
        )

    def variance_risk_premium(
        self,
        current_iv: float,
        realized_vol: float,
    ) -> float:
        """IV - Realized Vol. Positive = options are expensive."""
        return round(current_iv - realized_vol, 2)


def realized_volatility(prices: np.ndarray, window: int = 30) -> float:
    """Calculate annualized realized volatility from price series."""
    if len(prices) < window + 1:
        return 0.0
    log_returns = np.diff(np.log(prices[-window - 1:]))
    return float(np.std(log_returns) * np.sqrt(252) * 100)
