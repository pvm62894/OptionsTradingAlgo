"""
FastAPI routes for the QuantumFlow API.

All endpoints are prefixed with /api/v1/.
"""
from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, HTTPException
from ...models.options import (
    OptionChainRequest, StrategyAnalysisRequest, BacktestConfig,
    PredictionRequest, StrategyLeg, SignalResponse,
)
from ...services.data.market_data import MarketDataService
from ...services.data.mock_generator import MOCK_STOCKS, generate_live_tick
from ...services.pricing.strategy import StrategyAnalyzer, iron_condor_legs, straddle_legs, bull_call_spread_legs
from ...services.pricing.greeks import PortfolioGreeksAggregator
from ...services.ml.regime_classifier import VolatilityRegimeClassifier
from ...services.execution.backtester import EventDrivenBacktester
from ...services.signals.signal_engine import AlgorithmicSignalEngine
from ...core.logging import get_logger
import pandas as pd
import numpy as np

log = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

# ─── Service instances (initialized in lifespan) ─────────
market_data = MarketDataService()
strategy_analyzer = StrategyAnalyzer()
greeks_aggregator = PortfolioGreeksAggregator()
regime_classifier = VolatilityRegimeClassifier()
backtester = EventDrivenBacktester()
signal_engine = AlgorithmicSignalEngine(market_data)

# WebSocket connection manager
_ws_connections: set[WebSocket] = set()


# ─── Health ───────────────────────────────────────────────

@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "QuantumFlow API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


# ─── Market Data ─────────────────────────────────────────

@router.get("/symbols")
async def list_symbols():
    """List available symbols with current prices."""
    return await market_data.get_available_symbols_live()


@router.get("/options/chain")
async def get_option_chain(
    symbol: str = Query("SPY", description="Underlying symbol"),
    expiry: date | None = Query(None, description="Expiry date (YYYY-MM-DD)"),
):
    """Fetch option chain with Greeks for a symbol."""
    symbol = symbol.upper()
    if not market_data.is_valid_symbol(symbol):
        raise HTTPException(404, f"Symbol {symbol} not found")

    chain = await market_data.get_option_chain(symbol, expiry)
    return chain.model_dump()


@router.get("/quote/{symbol}")
async def get_quote(symbol: str):
    """Get current quote for a symbol."""
    symbol = symbol.upper()
    if not market_data.is_valid_symbol(symbol):
        raise HTTPException(404, f"Symbol {symbol} not found")

    quote = await market_data.get_quote(symbol)
    return quote.model_dump()


@router.get("/historical/{symbol}")
async def get_historical(
    symbol: str,
    days: int = Query(252, ge=20, le=1000),
):
    """Get historical OHLCV data."""
    symbol = symbol.upper()
    data = await market_data.get_historical(symbol, days)
    return [d.model_dump() for d in data]


# ─── Strategy Analysis ───────────────────────────────────

@router.post("/strategy/analyze")
async def analyze_strategy(request: StrategyAnalysisRequest):
    """Analyze a multi-leg option strategy with payoff diagram."""
    symbol = request.symbol.upper()
    if not market_data.is_valid_symbol(symbol):
        raise HTTPException(404, f"Symbol {symbol} not found")

    quote = await market_data.get_quote(symbol)
    spot = quote.last
    vol = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])["vol"]

    result = strategy_analyzer.analyze(
        legs=request.legs,
        spot=spot,
        volatility=vol,
        price_range_pct=request.price_range_pct,
        time_steps=request.time_steps,
    )
    return result.model_dump()


@router.get("/strategy/templates")
async def get_strategy_templates(
    symbol: str = Query("SPY"),
):
    """Get pre-built strategy templates."""
    symbol = symbol.upper()
    quote = await market_data.get_quote(symbol)
    spot = quote.last
    vol = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])["vol"]

    from ...services.data.mock_generator import _next_monthly_expiry
    expiry = _next_monthly_expiry(date.today())

    templates = {
        "iron_condor": {
            "name": "Iron Condor",
            "description": "Neutral strategy profiting from low volatility",
            "legs": [l.model_dump() for l in iron_condor_legs(spot, expiry)],
        },
        "bull_call_spread": {
            "name": "Bull Call Spread",
            "description": "Bullish debit spread with defined risk",
            "legs": [l.model_dump() for l in bull_call_spread_legs(spot, expiry)],
        },
        "straddle": {
            "name": "Long Straddle",
            "description": "Volatility play profiting from large moves",
            "legs": [l.model_dump() for l in straddle_legs(spot, expiry)],
        },
    }
    return templates


# ─── Portfolio / Greeks ───────────────────────────────────

@router.get("/greeks/portfolio")
async def get_portfolio_greeks():
    """Get aggregate portfolio Greeks and risk metrics."""
    positions = market_data.get_mock_positions()
    spot_prices = {sym: data["price"] for sym, data in MOCK_STOCKS.items()}
    ivs = {sym: data["vol"] for sym, data in MOCK_STOCKS.items()}

    summary = greeks_aggregator.portfolio_summary(positions, spot_prices, ivs)
    return summary.model_dump()


# ─── Volatility Surface ──────────────────────────────────

@router.get("/volatility/surface")
async def get_volatility_surface(
    symbol: str = Query("SPY"),
):
    """Get implied volatility surface data."""
    symbol = symbol.upper()
    if not market_data.is_valid_symbol(symbol):
        raise HTTPException(404, f"Symbol {symbol} not found")

    surface = await market_data.get_volatility_surface(symbol)
    return surface.model_dump()


@router.get("/volatility/surface3d")
async def get_volatility_surface_3d(
    symbol: str = Query("SPY"),
):
    """
    Get flat list of vol surface points for 3D plotting.

    Returns: [{strike, dte, iv, volume, open_interest}, ...]
    """
    symbol = symbol.upper()
    if not market_data.is_valid_symbol(symbol):
        raise HTTPException(404, f"Symbol {symbol} not found")

    points = await market_data.get_volatility_surface_3d(symbol)
    return points


# ─── ML Predictions ───────────────────────────────────────

@router.post("/predict/volatility")
async def predict_volatility(request: PredictionRequest):
    """Predict volatility regime using ML model."""
    symbol = request.symbol.upper()
    if not market_data.is_valid_symbol(symbol):
        raise HTTPException(404, f"Symbol {symbol} not found")

    # Get historical data for prediction
    historical = await market_data.get_historical(symbol, 504)
    df = pd.DataFrame([h.model_dump() for h in historical])
    df.set_index("timestamp", inplace=True)

    prediction = regime_classifier.predict(df, symbol=symbol)
    return prediction.model_dump()


# ─── Backtesting ──────────────────────────────────────────

@router.post("/backtest/run")
async def run_backtest(config: BacktestConfig):
    """Execute a backtest with the given configuration."""
    symbol = config.symbol.upper()
    if not market_data.is_valid_symbol(symbol):
        raise HTTPException(404, f"Symbol {symbol} not found")

    # Get historical data
    days = (config.end_date - config.start_date).days
    historical = await market_data.get_historical(symbol, max(days + 50, 300))
    df = pd.DataFrame([h.model_dump() for h in historical])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)

    vol = MOCK_STOCKS.get(symbol, MOCK_STOCKS["SPY"])["vol"]

    try:
        result = backtester.run(config, df, volatility=vol)
        return result.model_dump()
    except ValueError as e:
        raise HTTPException(400, str(e))


# ─── Algorithmic Signals ────────────────────────────────

@router.get("/signals")
async def get_signals(
    symbols: str = Query("SPY,QQQ", description="Comma-separated symbols"),
):
    """Get algorithmic trade signals for specified symbols."""
    symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    valid = [s for s in symbol_list if s in MOCK_STOCKS]
    if not valid:
        raise HTTPException(400, "No valid symbols provided")

    response = await signal_engine.scan_watchlist(valid)
    return response.model_dump()


@router.get("/signals/scan")
async def scan_all_signals():
    """Full watchlist scan — generate signals for all available symbols."""
    all_symbols = list(MOCK_STOCKS.keys())
    response = await signal_engine.scan_watchlist(all_symbols)
    return response.model_dump()


# ─── WebSocket Stream ────────────────────────────────────

@router.websocket("/stream/prices")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time price streaming.

    Client sends: {"subscribe": ["SPY", "AAPL", "NVDA"]}
    Server sends: Tick updates every 5s (live) or 500ms (mock)
    """
    await websocket.accept()
    _ws_connections.add(websocket)
    log.info("ws_client_connected", total=len(_ws_connections))

    subscribed_symbols = ["SPY"]  # Default

    try:
        # Start streaming in background
        stream_task = asyncio.create_task(
            _stream_ticks(websocket, subscribed_symbols)
        )

        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if "subscribe" in msg:
                    valid = [s.upper() for s in msg["subscribe"] if market_data.is_valid_symbol(s.upper())]
                    if valid:
                        subscribed_symbols.clear()
                        subscribed_symbols.extend(valid)
                        await websocket.send_json({"type": "subscribed", "symbols": valid})
                elif msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        stream_task.cancel()
        _ws_connections.discard(websocket)
        log.info("ws_client_disconnected", total=len(_ws_connections))


async def _stream_ticks(websocket: WebSocket, symbols: list[str]):
    """Stream price ticks for subscribed symbols.

    When yfinance is active, polls every 5 seconds for real prices.
    Falls back to mock ticks at 500ms if yfinance is unavailable.
    """
    prev_prices = {}
    use_live = market_data._yf is not None

    while True:
        for symbol in symbols:
            try:
                if use_live:
                    quote = await market_data.get_quote(symbol)
                else:
                    quote = generate_live_tick(symbol)

                prev = prev_prices.get(symbol, quote.last)
                change = quote.last - prev
                change_pct = (change / prev * 100) if prev > 0 else 0

                await websocket.send_json({
                    "type": "tick",
                    "symbol": symbol,
                    "price": quote.last,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "change": round(change, 2),
                    "change_pct": round(change_pct, 3),
                    "volume": quote.volume,
                    "timestamp": quote.timestamp.isoformat(),
                })

                prev_prices[symbol] = quote.last
            except Exception as e:
                log.warning("ws_tick_error", symbol=symbol, error=str(e))

        # Live: poll every 5s to avoid rate limits. Mock: 500ms for responsiveness.
        await asyncio.sleep(5.0 if use_live else 0.5)
