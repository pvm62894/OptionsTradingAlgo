"""
Portfolio-level Greeks aggregation and risk monitoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from ...models.options import Greeks, Position, PortfolioSummary, OrderSide
from .black_scholes import BlackScholesEngine
from datetime import date


@dataclass
class MarginRequirement:
    initial_margin: float
    maintenance_margin: float
    excess_margin: float


class PortfolioGreeksAggregator:
    """Aggregate position-level Greeks to portfolio level with risk monitoring."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.r = risk_free_rate
        self.bs = BlackScholesEngine()

    def calculate_position_greeks(
        self,
        position: Position,
        spot: float,
        sigma: float,
    ) -> Greeks:
        """Calculate current Greeks for a single position."""
        today = date.today()
        dte = (position.expiry - today).days
        T = max(dte / 365.0, 1 / 365.0)  # Minimum 1 day
        is_call = position.option_type.value == "call"

        result = self.bs.greeks(spot, position.strike, T, self.r, sigma, is_call)

        multiplier = position.quantity * (1 if position.side == OrderSide.BUY else -1) * 100

        return Greeks(
            delta=round(result.delta * multiplier, 4),
            gamma=round(result.gamma * multiplier, 6),
            theta=round(result.theta * multiplier, 4),
            vega=round(result.vega * multiplier, 4),
            rho=round(result.rho * multiplier, 4),
        )

    def aggregate_greeks(self, positions_greeks: list[Greeks]) -> Greeks:
        """Sum all position Greeks to portfolio level."""
        return Greeks(
            delta=round(sum(g.delta for g in positions_greeks), 4),
            gamma=round(sum(g.gamma for g in positions_greeks), 6),
            theta=round(sum(g.theta for g in positions_greeks), 4),
            vega=round(sum(g.vega for g in positions_greeks), 4),
            rho=round(sum(g.rho for g in positions_greeks), 4),
        )

    def portfolio_summary(
        self,
        positions: list[Position],
        spot_prices: dict[str, float],
        ivs: dict[str, float],
        account_value: float = 100000.0,
    ) -> PortfolioSummary:
        """Generate full portfolio summary with risk metrics."""
        all_greeks = []
        total_pnl = 0.0
        margin_used = 0.0

        for pos in positions:
            underlying = pos.underlying
            spot = spot_prices.get(underlying, 0.0)
            sigma = ivs.get(underlying, 0.20)

            greeks = self.calculate_position_greeks(pos, spot, sigma)
            all_greeks.append(greeks)

            # Update position P&L
            is_call = pos.option_type.value == "call"
            today = date.today()
            dte = max((pos.expiry - today).days, 1)
            T = dte / 365.0
            current_price = self.bs.price(spot, pos.strike, T, self.r, sigma, is_call)
            pos.current_price = round(current_price, 4)

            multiplier = pos.quantity * (1 if pos.side == OrderSide.BUY else -1)
            pos.unrealized_pnl = round((current_price - pos.entry_price) * multiplier * 100, 2)
            pos.greeks = greeks
            total_pnl += pos.unrealized_pnl

            # Simple margin calc: 20% of notional for sold options
            if pos.side == OrderSide.SELL:
                margin_used += spot * pos.quantity * 100 * 0.20

        portfolio_greeks = self.aggregate_greeks(all_greeks)
        buying_power = account_value - margin_used

        # Max loss estimation (simplified)
        max_loss = sum(
            abs(p.entry_price * p.quantity * 100)
            for p in positions
            if p.side == OrderSide.BUY
        ) + sum(
            p.strike * p.quantity * 100
            for p in positions
            if p.side == OrderSide.SELL and p.option_type.value == "put"
        )

        return PortfolioSummary(
            positions=positions,
            total_greeks=portfolio_greeks,
            total_pnl=round(total_pnl, 2),
            margin_used=round(margin_used, 2),
            buying_power=round(buying_power, 2),
            max_loss=round(max_loss, 2),
        )

    def calculate_margin(
        self,
        position: Position,
        spot: float,
    ) -> MarginRequirement:
        """Calculate margin requirements for a position (Reg-T simplified)."""
        notional = spot * position.quantity * 100

        if position.side == OrderSide.BUY:
            # Long options: full premium
            initial = position.entry_price * position.quantity * 100
            maintenance = initial
        else:
            # Short options: 20% of underlying + premium - OTM amount
            is_call = position.option_type.value == "call"
            if is_call:
                otm_amount = max(position.strike - spot, 0) * position.quantity * 100
            else:
                otm_amount = max(spot - position.strike, 0) * position.quantity * 100

            initial = notional * 0.20 + position.entry_price * position.quantity * 100 - otm_amount
            initial = max(initial, notional * 0.10)  # Minimum 10%
            maintenance = initial * 0.75

        return MarginRequirement(
            initial_margin=round(initial, 2),
            maintenance_margin=round(maintenance, 2),
            excess_margin=0.0,  # Filled by caller with account info
        )
