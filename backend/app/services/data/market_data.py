"""
Market data service with caching layer.

Uses YFinance as primary data source with mock data fallback.
Falls back to mock data when yfinance fails or is disabled.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime
from typing import Any

import numpy as np
from ...core.config import get_settings
from ...core.logging import get_logger
from ...models.options import OptionChain, Quote, OHLCV, VolatilitySurface
from .mock_generator import (
    generate_option_chain,
    generate_historical_ohlcv,
    generate_live_tick,
    generate_vol_surface_data,
    generate_mock_positions,
    MOCK_STOCKS,
)
from .yfinance_provider import YFinanceProvider, SUPPORTED_SYMBOLS
from ..pricing.volatility import VolatilitySurfaceBuilder

log = get_logger(__name__)

# Config flag: set QF_USE_LIVE_DATA=false to disable yfinance
USE_LIVE_DATA = os.environ.get("QF_USE_LIVE_DATA", "true").lower() in ("true", "1", "yes")


class MarketDataService:
    """
    Unified market data interface.

    Uses YFinance for live data with automatic fallback to mock data.
    """

    def __init__(self):
        self.settings = get_settings()
        self._redis = None
        self._yf = YFinanceProvider() if USE_LIVE_DATA else None
        self._vol_builder = VolatilitySurfaceBuilder()

    async def initialize(self):
        """Initialize connections."""
        if self.settings.REDIS_URL and self._yf:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(
                    self.settings.REDIS_URL,
                    decode_responses=True,
                )
                await self._redis.ping()
                log.info("redis_connected")
            except Exception as e:
                log.warning("redis_connection_failed", error=str(e))
                self._redis = None

        mode = "yfinance" if self._yf else "mock"
        log.info("market_data_service_initialized", mode=mode)

    async def close(self):
        if self._redis:
            await self._redis.close()

    # ─── Option Chains ────────────────────────────────────

    async def get_option_chain(
        self,
        symbol: str,
        expiry: date | None = None,
    ) -> OptionChain:
        """Fetch option chain, using cache if available."""
        cache_key = f"chain:{symbol}:{expiry or 'next'}"

        # Check Redis cache
        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                return OptionChain.model_validate_json(cached)

        # Try yfinance first
        if self._yf:
            chain = await self._yf.get_option_chain(symbol, expiry)
            if chain is not None:
                log.debug("chain_from_yfinance", symbol=symbol)
                if self._redis:
                    await self._redis.setex(
                        cache_key,
                        self.settings.OPTION_CHAIN_CACHE_TTL,
                        chain.model_dump_json(),
                    )
                return chain

        # Fallback to mock
        log.debug("chain_from_mock", symbol=symbol)
        return generate_option_chain(symbol, expiry)

    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol."""
        if self._yf:
            quote = await self._yf.get_quote(symbol)
            if quote is not None:
                return quote
        return generate_live_tick(symbol)

    async def get_historical(
        self,
        symbol: str,
        days: int = 504,
    ) -> list[OHLCV]:
        """Get historical OHLCV data."""
        if self._yf:
            data = await self._yf.get_historical(symbol, days)
            if data is not None:
                return data
        return generate_historical_ohlcv(symbol, days)

    async def get_volatility_surface(self, symbol: str) -> VolatilitySurface:
        """Build volatility surface from option chain data."""
        cache_key = f"volsurf:{symbol}"

        if self._redis:
            cached = await self._redis.get(cache_key)
            if cached:
                return VolatilitySurface.model_validate_json(cached)

        # Get spot price
        spot = None
        if self._yf:
            quote = await self._yf.get_quote(symbol)
            if quote is not None:
                spot = quote.last

        if spot is None:
            stock = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])
            spot = stock["price"]

        # Get chain contracts for surface
        contracts = None
        if self._yf:
            contracts = await self._yf.get_volatility_surface(symbol)

        if contracts is None:
            contracts = generate_vol_surface_data(symbol)

        # Get historical IVs for rank calculation
        historical_ivs = None
        if self._yf:
            hist = await self._yf.get_historical(symbol, 252)
            if hist and len(hist) > 30:
                prices = [h.close for h in hist]
                log_rets = [
                    abs(prices[i] / prices[i - 1] - 1)
                    for i in range(1, len(prices))
                ]
                # Rolling 30-day realized vol as proxy for historical IV
                arr = np.array(log_rets)
                historical_ivs = []
                for i in range(30, len(arr)):
                    rv = float(np.std(arr[i - 30:i]) * np.sqrt(252) * 100)
                    historical_ivs.append(rv)

        if not historical_ivs or len(historical_ivs) < 20:
            stock = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])
            historical_ivs = [stock["vol"] * 100 * (0.8 + i * 0.002) for i in range(252)]

        surface = self._vol_builder.build_surface(
            underlying=symbol,
            spot=spot,
            chains=contracts,
            historical_ivs=historical_ivs,
        )

        if self._redis:
            await self._redis.setex(
                cache_key,
                self.settings.VOLATILITY_SURFACE_CACHE_TTL,
                surface.model_dump_json(),
            )

        return surface

    async def get_volatility_surface_3d(self, symbol: str) -> list[dict]:
        """
        Get flat list of vol surface points for 3D plotting.
        Returns: [{strike, dte, iv, volume, open_interest}, ...]
        """
        if self._yf:
            raw = await self._yf.get_volatility_surface(symbol)
            if raw is not None:
                spot_quote = await self._yf.get_quote(symbol)
                spot = spot_quote.last if spot_quote else MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])["price"]
                points = self._build_3d_points(raw, spot)
                if points:
                    return points

        # Fallback to mock
        raw = generate_vol_surface_data(symbol)
        spot = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])["price"]
        return self._build_3d_points(raw, spot)

    def _build_3d_points(self, raw: list[dict], spot: float) -> list[dict]:
        """Convert raw chain data to flat 3D surface points."""
        today = date.today()
        points = []
        for c in raw:
            expiry = c["expiry"] if isinstance(c["expiry"], date) else date.fromisoformat(c["expiry"])
            dte = (expiry - today).days
            if dte <= 0:
                continue
            mid = (c["bid"] + c["ask"]) / 2
            if mid <= 0:
                continue
            T = dte / 365.0
            is_call = c["option_type"] == "call"
            iv = self._vol_builder.bs.implied_volatility(mid, spot, float(c["strike"]), T, 0.05, is_call)
            if iv is None or iv <= 0.01 or iv > 3.0:
                continue
            points.append({
                "strike": c["strike"],
                "dte": dte,
                "iv": round(iv * 100, 2),
                "volume": c.get("volume", 0),
                "open_interest": c.get("open_interest", 0),
            })
        return points

    def get_available_symbols(self) -> list[dict]:
        """Return list of available symbols with current prices."""
        # This is called synchronously from the route, so use mock data
        # as the baseline. Live prices are fetched via /quote endpoint.
        return [
            {
                "symbol": sym,
                "price": data["price"],
                "volatility": round(data["vol"] * 100, 1),
            }
            for sym, data in MOCK_STOCKS.items()
        ]

    async def get_available_symbols_live(self) -> list[dict]:
        """Return list of available symbols with real prices (async)."""
        if self._yf:
            result = await self._yf.get_available_symbols()
            if result:
                return result
        return self.get_available_symbols()

    def get_mock_positions(self):
        return generate_mock_positions()

    def is_valid_symbol(self, symbol: str) -> bool:
        """Check if a symbol is supported."""
        return symbol in MOCK_STOCKS or symbol in SUPPORTED_SYMBOLS
