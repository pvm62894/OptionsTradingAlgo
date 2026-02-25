"""
Black-Scholes-Merton pricing engine with full Greeks calculation.

All formulas follow Hull (Options, Futures, and Other Derivatives).
Uses Numba JIT compilation for hot-path performance.
"""

import numpy as np
from scipy.stats import norm
from numba import njit, float64
from dataclasses import dataclass
from typing import Optional


# ─── Numba-accelerated core functions ─────────────────────

@njit(float64(float64), cache=True)
def _norm_cdf(x: float) -> float:
    """Cumulative normal distribution using Hart's approximation (|error| < 7.5e-8)."""
    # Hart's rational approximation for erfc
    if x >= 0:
        k = 1.0 / (1.0 + 0.2316419 * x)
    else:
        k = 1.0 / (1.0 - 0.2316419 * x)

    poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))))

    if x >= 0:
        return 1.0 - poly * np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
    else:
        return poly * np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@njit(float64(float64), cache=True)
def _norm_pdf(x: float) -> float:
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


@njit(cache=True)
def _d1d2(S: float, K: float, T: float, r: float, sigma: float):
    """Calculate d1 and d2 parameters."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


@njit(cache=True)
def _bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return S * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)


@njit(cache=True)
def _bs_put_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


@njit(cache=True)
def _call_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return _norm_cdf(d1)


@njit(cache=True)
def _put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return _norm_cdf(d1) - 1.0


@njit(cache=True)
def _gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Gamma is the same for calls and puts."""
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return _norm_pdf(d1) / (S * sigma * np.sqrt(T))


@njit(cache=True)
def _call_theta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Call theta (per calendar day)."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    term1 = -(S * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * _norm_cdf(d2)
    return (term1 + term2) / 365.0


@njit(cache=True)
def _put_theta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Put theta (per calendar day)."""
    if T <= 0:
        return 0.0
    d1, d2 = _d1d2(S, K, T, r, sigma)
    term1 = -(S * _norm_pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * _norm_cdf(-d2)
    return (term1 + term2) / 365.0


@njit(cache=True)
def _vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Vega (per 1% change in vol, same for calls and puts)."""
    if T <= 0:
        return 0.0
    d1, _ = _d1d2(S, K, T, r, sigma)
    return S * _norm_pdf(d1) * np.sqrt(T) / 100.0


@njit(cache=True)
def _call_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    _, d2 = _d1d2(S, K, T, r, sigma)
    return K * T * np.exp(-r * T) * _norm_cdf(d2) / 100.0


@njit(cache=True)
def _put_rho(S: float, K: float, T: float, r: float, sigma: float) -> float:
    if T <= 0:
        return 0.0
    _, d2 = _d1d2(S, K, T, r, sigma)
    return -K * T * np.exp(-r * T) * _norm_cdf(-d2) / 100.0


# ─── Implied Volatility Solver ────────────────────────────

@njit(cache=True)
def _implied_vol_newton(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    is_call: bool,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> float:
    """Newton-Raphson IV solver. Returns -1 on failure."""
    if T <= 0 or market_price <= 0:
        return -1.0

    # Initial guess using Brenner-Subrahmanyam approximation
    sigma = np.sqrt(2.0 * np.pi / T) * market_price / S

    if sigma <= 0.001:
        sigma = 0.20  # fallback

    for _ in range(max_iter):
        if is_call:
            price = _bs_call_price(S, K, T, r, sigma)
        else:
            price = _bs_put_price(S, K, T, r, sigma)

        diff = price - market_price

        if abs(diff) < tol:
            return sigma

        # Vega for Newton step (raw, not /100)
        d1, _ = _d1d2(S, K, T, r, sigma)
        vega_raw = S * _norm_pdf(d1) * np.sqrt(T)

        if vega_raw < 1e-12:
            break

        sigma = sigma - diff / vega_raw

        if sigma <= 0.001:
            sigma = 0.001

    return sigma if abs(diff) < 0.01 else -1.0


# ─── Public API ───────────────────────────────────────────

@dataclass
class GreeksResult:
    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class BlackScholesEngine:
    """High-performance Black-Scholes pricing engine."""

    @staticmethod
    def price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
        """
        Calculate European option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate (annualized, e.g. 0.05 for 5%)
            sigma: Volatility (annualized, e.g. 0.20 for 20%)
            is_call: True for call, False for put
        """
        if is_call:
            return _bs_call_price(S, K, T, r, sigma)
        return _bs_put_price(S, K, T, r, sigma)

    @staticmethod
    def greeks(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> GreeksResult:
        """Calculate all Greeks for an option."""
        if is_call:
            price = _bs_call_price(S, K, T, r, sigma)
            delta = _call_delta(S, K, T, r, sigma)
            theta = _call_theta(S, K, T, r, sigma)
            rho = _call_rho(S, K, T, r, sigma)
        else:
            price = _bs_put_price(S, K, T, r, sigma)
            delta = _put_delta(S, K, T, r, sigma)
            theta = _put_theta(S, K, T, r, sigma)
            rho = _put_rho(S, K, T, r, sigma)

        return GreeksResult(
            price=price,
            delta=delta,
            gamma=_gamma(S, K, T, r, sigma),
            theta=theta,
            vega=_vega(S, K, T, r, sigma),
            rho=rho,
        )

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        is_call: bool,
    ) -> Optional[float]:
        """
        Solve for implied volatility using Newton-Raphson.

        Returns None if solver fails to converge.
        """
        result = _implied_vol_newton(market_price, S, K, T, r, is_call)
        return result if result > 0 else None

    @staticmethod
    def price_array(
        S: np.ndarray,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool,
    ) -> np.ndarray:
        """Vectorized pricing for payoff diagrams."""
        prices = np.empty(len(S))
        for i in range(len(S)):
            if is_call:
                prices[i] = _bs_call_price(S[i], K, T, r, sigma)
            else:
                prices[i] = _bs_put_price(S[i], K, T, r, sigma)
        return prices


class BinomialTreePricer:
    """Cox-Ross-Rubinstein binomial tree for American options."""

    @staticmethod
    def price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        is_call: bool,
        steps: int = 200,
        dividend_yield: float = 0.0,
    ) -> float:
        """Price American option using CRR binomial tree."""
        dt = T / steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - dividend_yield) * dt) - d) / (u - d)
        disc = np.exp(-r * dt)

        # Build full price tree at terminal nodes (j = 0..steps)
        # Node (steps, j) has price S * u^j * d^(steps-j)
        terminal_prices = np.array([S * u**j * d**(steps - j) for j in range(steps + 1)])

        # Terminal payoffs
        if is_call:
            values = np.maximum(terminal_prices - K, 0.0)
        else:
            values = np.maximum(K - terminal_prices, 0.0)

        # Backward induction with early exercise check
        for i in range(steps - 1, -1, -1):
            # Prices at step i: S * u^j * d^(i-j) for j = 0..i
            step_prices = np.array([S * u**j * d**(i - j) for j in range(i + 1)])
            continuation = disc * (p * values[1:i+2] + (1 - p) * values[0:i+1])

            if is_call:
                exercise = np.maximum(step_prices - K, 0.0)
            else:
                exercise = np.maximum(K - step_prices, 0.0)

            values = np.maximum(continuation, exercise)

        return float(values[0])
