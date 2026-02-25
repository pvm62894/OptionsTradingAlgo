"""
YFinance data provider for real market data.

Fetches live quotes, option chains, and historical data from Yahoo Finance.
Falls back to mock data on any failure. Uses in-memory TTL caching to
avoid hammering the Yahoo Finance API.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
from cachetools import TTLCache

from ...core.logging import get_logger
from ...models.options import (
    Greeks,
    OHLCV,
    OptionChain,
    OptionContract,
    OptionType,
    Quote,
)

log = get_logger(__name__)

# Thread pool for blocking yfinance calls
_executor = ThreadPoolExecutor(max_workers=4)

# Cache configuration: max 256 entries, 60s TTL by default
_quote_cache: TTLCache = TTLCache(maxsize=256, ttl=15)
_chain_cache: TTLCache = TTLCache(maxsize=64, ttl=60)
_historical_cache: TTLCache = TTLCache(maxsize=64, ttl=300)
_surface_cache: TTLCache = TTLCache(maxsize=32, ttl=300)

# Supported symbols (same universe as mock data)
SUPPORTED_SYMBOLS = {
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA",
    "TSLA", "AMZN", "META", "JPM", "GLD",
}


def _import_yfinance():
    """Lazy import to avoid startup cost if not used."""
    import yfinance as yf
    return yf


class YFinanceProvider:
    """Provides real market data via yfinance with in-memory caching."""

    async def get_quote(self, symbol: str) -> Quote | None:
        """Fetch current quote for a symbol. Returns None on failure."""
        cache_key = f"quote:{symbol}"
        cached = _quote_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            loop = asyncio.get_event_loop()
            quote = await loop.run_in_executor(_executor, self._fetch_quote_sync, symbol)
            if quote is not None:
                _quote_cache[cache_key] = quote
            return quote
        except Exception as e:
            log.warning("yfinance_quote_failed", symbol=symbol, error=str(e))
            return None

    async def get_option_chain(self, symbol: str, expiry: date | None = None) -> OptionChain | None:
        """Fetch option chain. Returns None on failure."""
        cache_key = f"chain:{symbol}:{expiry or 'next'}"
        cached = _chain_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            loop = asyncio.get_event_loop()
            chain = await loop.run_in_executor(
                _executor, self._fetch_chain_sync, symbol, expiry
            )
            if chain is not None:
                _chain_cache[cache_key] = chain
            return chain
        except Exception as e:
            log.warning("yfinance_chain_failed", symbol=symbol, error=str(e))
            return None

    async def get_historical(self, symbol: str, days: int = 504) -> list[OHLCV] | None:
        """Fetch historical OHLCV data. Returns None on failure."""
        cache_key = f"hist:{symbol}:{days}"
        cached = _historical_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                _executor, self._fetch_historical_sync, symbol, days
            )
            if data is not None:
                _historical_cache[cache_key] = data
            return data
        except Exception as e:
            log.warning("yfinance_historical_failed", symbol=symbol, error=str(e))
            return None

    async def get_volatility_surface(self, symbol: str) -> list[dict] | None:
        """Fetch option data across multiple expiries for vol surface. Returns None on failure."""
        cache_key = f"surface:{symbol}"
        cached = _surface_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                _executor, self._fetch_surface_sync, symbol
            )
            if data is not None:
                _surface_cache[cache_key] = data
            return data
        except Exception as e:
            log.warning("yfinance_surface_failed", symbol=symbol, error=str(e))
            return None

    async def get_available_symbols(self) -> list[dict] | None:
        """Return watchlist with real prices. Returns None on failure."""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor, self._fetch_symbols_sync
            )
            return result
        except Exception as e:
            log.warning("yfinance_symbols_failed", error=str(e))
            return None

    # ─── Synchronous fetch methods (run in executor) ─────

    @staticmethod
    def _fetch_quote_sync(symbol: str) -> Quote | None:
        yf = _import_yfinance()
        ticker = yf.Ticker(symbol)
        info = ticker.fast_info

        last_price = float(info.last_price)
        prev_close = float(info.previous_close) if hasattr(info, 'previous_close') else last_price

        # Try to get bid/ask from info; fall back to approximation
        try:
            bid = float(ticker.info.get("bid", last_price - 0.01))
            ask = float(ticker.info.get("ask", last_price + 0.01))
        except Exception:
            spread = last_price * 0.0001
            bid = round(last_price - spread, 2)
            ask = round(last_price + spread, 2)

        volume = int(info.last_volume) if hasattr(info, 'last_volume') and info.last_volume else 0

        return Quote(
            symbol=symbol,
            bid=round(bid, 2),
            ask=round(ask, 2),
            last=round(last_price, 2),
            volume=volume,
            timestamp=datetime.utcnow(),
        )

    @staticmethod
    def _fetch_chain_sync(symbol: str, expiry: date | None) -> OptionChain | None:
        yf = _import_yfinance()
        ticker = yf.Ticker(symbol)
        spot = float(ticker.fast_info.last_price)

        # Get available expiries
        available_expiries = ticker.options  # list of date strings
        if not available_expiries:
            return None

        if expiry is not None:
            expiry_str = expiry.isoformat()
            # Find closest available expiry
            closest = min(available_expiries, key=lambda e: abs(date.fromisoformat(e) - expiry))
            target_expiry = closest
        else:
            # Use the nearest expiry
            target_expiry = available_expiries[0]

        expiry_date = date.fromisoformat(target_expiry)
        chain = ticker.option_chain(target_expiry)

        calls = []
        for _, row in chain.calls.iterrows():
            contract = _row_to_contract(row, symbol, expiry_date, OptionType.CALL, spot)
            if contract is not None:
                calls.append(contract)

        puts = []
        for _, row in chain.puts.iterrows():
            contract = _row_to_contract(row, symbol, expiry_date, OptionType.PUT, spot)
            if contract is not None:
                puts.append(contract)

        return OptionChain(
            underlying=symbol,
            underlying_price=spot,
            expiry=expiry_date,
            calls=sorted(calls, key=lambda c: c.strike),
            puts=sorted(puts, key=lambda p: p.strike),
            timestamp=datetime.utcnow(),
        )

    @staticmethod
    def _fetch_historical_sync(symbol: str, days: int) -> list[OHLCV] | None:
        yf = _import_yfinance()
        ticker = yf.Ticker(symbol)

        # Map days to yfinance period
        if days <= 5:
            period = "5d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        elif days <= 730:
            period = "2y"
        else:
            period = "5y"

        df = ticker.history(period=period)
        if df.empty:
            return None

        result = []
        for ts, row in df.iterrows():
            result.append(OHLCV(
                timestamp=ts.to_pydatetime().replace(tzinfo=None),
                open=round(float(row["Open"]), 2),
                high=round(float(row["High"]), 2),
                low=round(float(row["Low"]), 2),
                close=round(float(row["Close"]), 2),
                volume=int(row["Volume"]),
            ))

        return result

    @staticmethod
    def _fetch_surface_sync(symbol: str) -> list[dict] | None:
        yf = _import_yfinance()
        ticker = yf.Ticker(symbol)
        spot = float(ticker.fast_info.last_price)

        available_expiries = ticker.options
        if not available_expiries:
            return None

        # Use up to 8 expiries for surface construction
        expiries_to_use = available_expiries[:8]
        contracts = []
        today = date.today()

        for exp_str in expiries_to_use:
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if dte <= 0:
                continue

            try:
                chain = ticker.option_chain(exp_str)
            except Exception:
                continue

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                for _, row in df.iterrows():
                    strike = float(row["strike"])
                    moneyness = strike / spot
                    if moneyness < 0.7 or moneyness > 1.3:
                        continue

                    bid = float(row.get("bid", 0) or 0)
                    ask = float(row.get("ask", 0) or 0)
                    if bid <= 0 or ask <= 0:
                        continue

                    vol = int(row.get("volume", 0) or 0)
                    oi = int(row.get("openInterest", 0) or 0)

                    contracts.append({
                        "strike": round(strike, 2),
                        "expiry": exp_date.isoformat(),
                        "option_type": opt_type,
                        "bid": round(bid, 2),
                        "ask": round(ask, 2),
                        "volume": vol,
                        "open_interest": oi,
                    })

        return contracts if contracts else None

    @staticmethod
    def _fetch_symbols_sync() -> list[dict]:
        yf = _import_yfinance()
        result = []
        for symbol in sorted(SUPPORTED_SYMBOLS):
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.fast_info
                price = float(info.last_price)
                # Approximate implied vol from 30-day historical vol
                hist = ticker.history(period="1mo")
                if not hist.empty and len(hist) > 5:
                    log_returns = np.diff(np.log(hist["Close"].values))
                    vol = float(np.std(log_returns) * np.sqrt(252) * 100)
                else:
                    vol = 20.0

                result.append({
                    "symbol": symbol,
                    "price": round(price, 2),
                    "volatility": round(vol, 1),
                })
            except Exception as e:
                log.warning("yfinance_symbol_fetch_failed", symbol=symbol, error=str(e))
                continue
        return result


def _row_to_contract(
    row: Any,
    underlying: str,
    expiry: date,
    option_type: OptionType,
    spot: float,
) -> OptionContract | None:
    """Convert a yfinance option chain DataFrame row to OptionContract."""
    try:
        strike = float(row["strike"])
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        last = float(row.get("lastPrice", 0) or 0)
        volume = int(row.get("volume", 0) or 0)
        oi = int(row.get("openInterest", 0) or 0)
        iv = float(row.get("impliedVolatility", 0) or 0) * 100  # Convert to percentage

        contract_symbol = str(row.get("contractSymbol", f"{underlying}{expiry.strftime('%y%m%d')}"))

        is_call = option_type == OptionType.CALL
        itm = (strike < spot) if is_call else (strike > spot)

        # yfinance does not always provide Greeks; set what's available
        delta = 0.0
        gamma = 0.0
        theta = 0.0
        vega = 0.0

        return OptionContract(
            symbol=contract_symbol,
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            bid=round(bid, 2),
            ask=round(ask, 2),
            last=round(last, 2),
            volume=volume,
            open_interest=oi,
            implied_volatility=round(iv, 2),
            greeks=Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=0.0),
            in_the_money=itm,
        )
    except Exception as e:
        log.debug("row_to_contract_failed", error=str(e))
        return None
