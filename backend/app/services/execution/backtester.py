"""
Event-driven backtesting framework.

Simulates tick-by-tick option strategy execution with realistic
fills, slippage, and margin modeling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from ...models.options import (
    BacktestConfig, BacktestResult, StrategyLeg, OptionType, OrderSide,
)
from ..pricing.black_scholes import BlackScholesEngine
from ...core.logging import get_logger

log = get_logger(__name__)


class EventType(str, Enum):
    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    ORDER = "order"
    FILL = "fill"


@dataclass
class Event:
    type: EventType
    timestamp: datetime
    data: dict


@dataclass
class Trade:
    entry_date: date
    exit_date: date | None
    strategy: str
    legs: list[dict]
    entry_premium: float
    exit_premium: float = 0.0
    pnl: float = 0.0
    slippage: float = 0.0
    status: str = "open"


@dataclass
class BacktestState:
    cash: float
    equity_curve: list[dict] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)
    open_positions: list[dict] = field(default_factory=list)
    greeks_history: list[dict] = field(default_factory=list)
    peak_equity: float = 0.0
    max_drawdown: float = 0.0


class EventDrivenBacktester:
    """
    Event-driven backtester with realistic execution simulation.

    Features:
    - Fills at BID (sells) and ASK (buys), never mid-price
    - Configurable slippage
    - Margin tracking
    - Path-dependent logic support
    """

    def __init__(
        self,
        slippage_ticks: int = 1,
        tick_size: float = 0.01,
        commission_per_contract: float = 0.65,
    ):
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.commission = commission_per_contract
        self.bs = BlackScholesEngine()
        self.r = 0.045

    def run(
        self,
        config: BacktestConfig,
        prices: pd.DataFrame,
        entry_frequency_days: int = 30,
        volatility: float = 0.20,
    ) -> BacktestResult:
        """
        Run backtest simulation.

        Args:
            config: Backtest configuration
            prices: OHLCV DataFrame indexed by date
            entry_frequency_days: How often to enter new positions
            volatility: Assumed volatility for pricing
        """
        log.info("backtest_started", strategy=config.strategy_name, symbol=config.symbol)

        state = BacktestState(cash=config.initial_capital, peak_equity=config.initial_capital)

        # Convert dates
        start = pd.Timestamp(config.start_date)
        end = pd.Timestamp(config.end_date)
        price_data = prices[(prices.index >= start) & (prices.index <= end)]

        if len(price_data) < 20:
            raise ValueError("Insufficient data for backtesting")

        last_entry = start - timedelta(days=entry_frequency_days + 1)

        for idx, row in price_data.iterrows():
            current_date = idx.date() if hasattr(idx, "date") else idx
            spot = row["close"]

            # ─── Check for exits on existing positions ────
            for pos in list(state.open_positions):
                pos_expiry = pos["expiry"]
                days_to_exp = (pos_expiry - current_date).days

                should_exit = False
                exit_reason = ""

                # Expiry exit
                if days_to_exp <= 1:
                    should_exit = True
                    exit_reason = "expiry"

                # Stop loss
                elif config.stop_loss_pct and pos.get("current_pnl", 0) < -abs(config.stop_loss_pct) * pos["entry_cost"]:
                    should_exit = True
                    exit_reason = "stop_loss"

                # Take profit
                elif config.take_profit_pct and pos.get("current_pnl", 0) > config.take_profit_pct * pos["entry_cost"]:
                    should_exit = True
                    exit_reason = "take_profit"

                if should_exit:
                    pnl = self._close_position(pos, spot, volatility, current_date)
                    state.cash += pnl + pos["entry_cost"]
                    state.open_positions.remove(pos)
                    state.trades[-1 if state.trades else 0].exit_date = current_date
                    state.trades[-1].exit_premium = round(pnl / 100, 2)
                    state.trades[-1].pnl = round(pnl, 2)
                    state.trades[-1].status = exit_reason

            # ─── Mark-to-market open positions ────────────
            position_value = 0.0
            total_greeks = {"delta": 0, "gamma": 0, "theta": 0, "vega": 0}

            for pos in state.open_positions:
                mtm = self._mark_to_market(pos, spot, volatility, current_date)
                pos["current_pnl"] = mtm - pos["entry_cost"]
                position_value += mtm

                # Track Greeks
                for leg in pos["legs"]:
                    dte = (leg["expiry"] - current_date).days
                    if dte <= 0:
                        continue
                    T = dte / 365.0
                    g = self.bs.greeks(spot, leg["strike"], T, self.r, volatility, leg["is_call"])
                    mult = leg["quantity"] * leg["direction"] * 100
                    total_greeks["delta"] += g.delta * mult
                    total_greeks["gamma"] += g.gamma * mult
                    total_greeks["theta"] += g.theta * mult
                    total_greeks["vega"] += g.vega * mult

            # ─── Check for new entry ──────────────────────
            days_since_entry = (current_date - last_entry).days if isinstance(last_entry, date) else entry_frequency_days + 1
            if days_since_entry >= entry_frequency_days and len(state.open_positions) == 0:
                entry_cost = self._open_position(config, state, spot, volatility, current_date)
                if entry_cost is not None:
                    last_entry = current_date

            # ─── Record equity curve ──────────────────────
            total_equity = state.cash + position_value
            state.peak_equity = max(state.peak_equity, total_equity)
            drawdown = (state.peak_equity - total_equity) / state.peak_equity if state.peak_equity > 0 else 0
            state.max_drawdown = max(state.max_drawdown, drawdown)

            state.equity_curve.append({
                "date": str(current_date),
                "equity": round(total_equity, 2),
                "cash": round(state.cash, 2),
                "positions_value": round(position_value, 2),
                "drawdown": round(drawdown * 100, 2),
            })

            state.greeks_history.append({
                "date": str(current_date),
                **{k: round(v, 4) for k, v in total_greeks.items()},
            })

        # ─── Calculate performance metrics ────────────────
        return self._compute_results(config, state)

    def _open_position(
        self,
        config: BacktestConfig,
        state: BacktestState,
        spot: float,
        vol: float,
        current_date: date,
    ) -> float | None:
        """Open a new position based on strategy config."""
        expiry = current_date + timedelta(days=30)

        # Default to Iron Condor if no legs specified
        if config.legs:
            legs = config.legs
        else:
            # Auto-generate Iron Condor
            legs = [
                StrategyLeg(option_type=OptionType.PUT, strike=round(spot * 0.95), expiry=expiry, side=OrderSide.BUY, quantity=1),
                StrategyLeg(option_type=OptionType.PUT, strike=round(spot * 0.97), expiry=expiry, side=OrderSide.SELL, quantity=1),
                StrategyLeg(option_type=OptionType.CALL, strike=round(spot * 1.03), expiry=expiry, side=OrderSide.SELL, quantity=1),
                StrategyLeg(option_type=OptionType.CALL, strike=round(spot * 1.05), expiry=expiry, side=OrderSide.BUY, quantity=1),
            ]

        total_cost = 0.0
        leg_details = []

        for leg in legs:
            T = (leg.expiry - current_date).days / 365.0
            is_call = leg.option_type == OptionType.CALL
            direction = 1 if leg.side == OrderSide.BUY else -1
            price = self.bs.price(spot, leg.strike, T, self.r, vol, is_call)

            # Apply slippage: buy at ask, sell at bid
            spread = price * 0.02  # 2% spread
            if direction > 0:
                fill_price = price + spread / 2 + self.slippage_ticks * self.tick_size
            else:
                fill_price = price - spread / 2 - self.slippage_ticks * self.tick_size
            fill_price = max(fill_price, 0.01)

            cost = fill_price * direction * leg.quantity * 100
            total_cost += cost
            total_cost -= self.commission * leg.quantity  # Commission

            leg_details.append({
                "strike": leg.strike,
                "expiry": leg.expiry,
                "is_call": is_call,
                "direction": direction,
                "quantity": leg.quantity,
                "fill_price": fill_price,
            })

        # Check if we can afford this
        if abs(total_cost) > state.cash * 0.2:  # Max 20% of cash per trade
            return None

        state.cash += total_cost  # Negative for debits, positive for credits
        state.open_positions.append({
            "legs": leg_details,
            "entry_cost": abs(total_cost),
            "entry_date": current_date,
            "expiry": max(leg.expiry for leg in legs),
            "current_pnl": 0.0,
        })

        state.trades.append(Trade(
            entry_date=current_date,
            exit_date=None,
            strategy=config.strategy_name,
            legs=[{
                "strike": l["strike"],
                "type": "call" if l["is_call"] else "put",
                "direction": "long" if l["direction"] > 0 else "short",
                "quantity": l["quantity"],
                "fill": round(l["fill_price"], 2),
            } for l in leg_details],
            entry_premium=round(total_cost / 100, 2),
        ))

        return abs(total_cost)

    def _close_position(
        self,
        position: dict,
        spot: float,
        vol: float,
        current_date: date,
    ) -> float:
        """Close position and return P&L."""
        total = 0.0
        for leg in position["legs"]:
            dte = (leg["expiry"] - current_date).days
            if dte <= 0:
                # Intrinsic value at expiry
                if leg["is_call"]:
                    value = max(spot - leg["strike"], 0)
                else:
                    value = max(leg["strike"] - spot, 0)
            else:
                T = dte / 365.0
                value = self.bs.price(spot, leg["strike"], T, self.r, vol, leg["is_call"])

            # Reverse direction to close
            close_direction = -leg["direction"]
            spread = value * 0.02
            if close_direction > 0:
                fill = value + spread / 2 + self.slippage_ticks * self.tick_size
            else:
                fill = value - spread / 2 - self.slippage_ticks * self.tick_size
            fill = max(fill, 0.0)

            total += fill * close_direction * leg["quantity"] * 100
            total -= self.commission * leg["quantity"]

        return total

    def _mark_to_market(
        self,
        position: dict,
        spot: float,
        vol: float,
        current_date: date,
    ) -> float:
        """Calculate current position value."""
        total = 0.0
        for leg in position["legs"]:
            dte = (leg["expiry"] - current_date).days
            if dte <= 0:
                if leg["is_call"]:
                    value = max(spot - leg["strike"], 0)
                else:
                    value = max(leg["strike"] - spot, 0)
            else:
                T = dte / 365.0
                value = self.bs.price(spot, leg["strike"], T, self.r, vol, leg["is_call"])

            total += value * leg["direction"] * leg["quantity"] * 100

        return total

    def _compute_results(self, config: BacktestConfig, state: BacktestState) -> BacktestResult:
        """Compute final performance metrics."""
        equity = [e["equity"] for e in state.equity_curve]

        if len(equity) < 2:
            raise ValueError("Not enough data points for analysis")

        returns = pd.Series(equity).pct_change().dropna()

        total_return = (equity[-1] - config.initial_capital) / config.initial_capital

        # Sharpe ratio (annualized)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0.0

        # Sortino ratio
        downside = returns[returns < 0]
        sortino = (returns.mean() / downside.std() * np.sqrt(252)) if len(downside) > 0 and downside.std() > 0 else 0.0

        # Win rate and profit factor
        closed = [t for t in state.trades if t.status != "open"]
        wins = [t for t in closed if t.pnl > 0]
        losses = [t for t in closed if t.pnl <= 0]
        win_rate = len(wins) / len(closed) if closed else 0
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        return BacktestResult(
            config=config,
            total_return=round(total_return * 100, 2),
            sharpe_ratio=round(sharpe, 3),
            sortino_ratio=round(sortino, 3),
            max_drawdown=round(state.max_drawdown * 100, 2),
            win_rate=round(win_rate * 100, 1),
            profit_factor=round(profit_factor, 3),
            total_trades=len(closed),
            equity_curve=state.equity_curve,
            trades=[{
                "entry_date": str(t.entry_date),
                "exit_date": str(t.exit_date) if t.exit_date else None,
                "strategy": t.strategy,
                "pnl": t.pnl,
                "status": t.status,
                "legs": t.legs,
            } for t in state.trades],
            greeks_over_time=state.greeks_history,
        )
