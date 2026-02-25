"""Pydantic models for options data."""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pydantic import BaseModel, Field


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class VolatilityRegime(str, Enum):
    LOW_VOL_TRENDING = "low_vol_trending"
    HIGH_VOL_MEAN_REVERTING = "high_vol_mean_reverting"
    CRISIS = "crisis"


# ─── Market Data ──────────────────────────────────────────

class Quote(BaseModel):
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime


class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


# ─── Greeks ───────────────────────────────────────────────

class Greeks(BaseModel):
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0


class OptionContract(BaseModel):
    symbol: str
    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    greeks: Greeks = Field(default_factory=Greeks)
    in_the_money: bool = False


class OptionChain(BaseModel):
    underlying: str
    underlying_price: float
    expiry: date
    calls: list[OptionContract]
    puts: list[OptionContract]
    timestamp: datetime


class OptionChainRequest(BaseModel):
    symbol: str
    expiry: date | None = None


# ─── Strategy ─────────────────────────────────────────────

class StrategyLeg(BaseModel):
    option_type: OptionType
    strike: float
    expiry: date
    side: OrderSide
    quantity: int = 1
    premium: float = 0.0


class StrategyAnalysis(BaseModel):
    legs: list[StrategyLeg]
    underlying_price: float
    max_profit: float | None = None
    max_loss: float | None = None
    breakeven_points: list[float] = []
    payoff_data: list[dict] = []
    total_greeks: Greeks = Field(default_factory=Greeks)
    net_premium: float = 0.0


class StrategyAnalysisRequest(BaseModel):
    symbol: str
    legs: list[StrategyLeg]
    price_range_pct: float = 20.0
    time_steps: int = 5


# ─── Portfolio ────────────────────────────────────────────

class Position(BaseModel):
    id: str
    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiry: date
    side: OrderSide
    quantity: int
    entry_price: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    greeks: Greeks = Field(default_factory=Greeks)


class PortfolioSummary(BaseModel):
    positions: list[Position]
    total_greeks: Greeks
    total_pnl: float
    margin_used: float
    buying_power: float
    max_loss: float


# ─── Volatility ───────────────────────────────────────────

class VolSurfacePoint(BaseModel):
    strike: float
    expiry: date
    days_to_expiry: int
    iv: float
    moneyness: float  # strike / spot


class VolatilitySurface(BaseModel):
    underlying: str
    spot_price: float
    points: list[VolSurfacePoint]
    iv_rank: float  # 0-100 percentile
    iv_percentile: float
    timestamp: datetime


# ─── Backtest ─────────────────────────────────────────────

class BacktestConfig(BaseModel):
    strategy_name: str
    symbol: str
    start_date: date
    end_date: date
    initial_capital: float = 100000.0
    legs: list[StrategyLeg] = []
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None


class BacktestResult(BaseModel):
    config: BacktestConfig
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    equity_curve: list[dict]
    trades: list[dict]
    greeks_over_time: list[dict]


# ─── ML ───────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    symbol: str
    horizon_days: int = 5


class VolatilityPrediction(BaseModel):
    symbol: str
    regime: VolatilityRegime
    regime_probabilities: dict[str, float]
    predicted_iv_30d: float
    confidence: float
    features_importance: dict[str, float] = {}
    timestamp: datetime


# ─── WebSocket ────────────────────────────────────────────

class WSTickMessage(BaseModel):
    type: str = "tick"
    symbol: str
    price: float
    change: float
    change_pct: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime


class WSGreeksUpdate(BaseModel):
    type: str = "greeks_update"
    portfolio_greeks: Greeks
    total_pnl: float
    timestamp: datetime


# ─── Algorithmic Signals ─────────────────────────────────

class TradeSignal(BaseModel):
    ticker: str
    strategy_name: str  # "Iron Condor", "Bull Call Spread", "Straddle", etc.
    direction: str  # "bullish", "bearish", "neutral"
    confidence: str  # "HIGH", "MEDIUM", "LOW"
    suggested_legs: list  # list of dicts with strike, expiry, option_type, side
    expected_value: float
    max_risk: float
    probability_of_profit: float
    iv_rank: float
    reasoning: str
    timestamp: str


class SignalResponse(BaseModel):
    signals: list[TradeSignal]
    scan_timestamp: str
    symbols_scanned: int
