"""
Algorithmic signal engine for options trade detection.

Scans option chains and volatility surfaces to generate actionable
trade signals based on quantitative criteria.
"""
from __future__ import annotations

import numpy as np
from datetime import date, datetime

from ...models.options import TradeSignal, SignalResponse, OptionContract
from ..data.market_data import MarketDataService
from ..data.mock_generator import MOCK_STOCKS
from ..pricing.volatility import VolatilitySurfaceBuilder, realized_volatility
from ...core.logging import get_logger

log = get_logger(__name__)


class AlgorithmicSignalEngine:
    """Scans option chains and generates algorithmic trade signals."""

    def __init__(self, market_data_service: MarketDataService):
        self._mds = market_data_service
        self._vol_builder = VolatilitySurfaceBuilder()

    async def scan_symbol(self, symbol: str) -> list[TradeSignal]:
        """Analyze a single symbol and return detected trade signals."""
        symbol = symbol.upper()
        stock = MOCK_STOCKS.get(symbol)
        if stock is None:
            return []

        spot = stock["price"]
        base_vol = stock["vol"]

        # Gather data
        chain = await self._mds.get_option_chain(symbol)
        surface = await self._mds.get_volatility_surface(symbol)
        historical = await self._mds.get_historical(symbol, 252)

        iv_rank = surface.iv_rank
        prices_array = np.array([bar.close for bar in historical])
        rv = realized_volatility(prices_array, window=30)

        # Compute skew and term structure
        skew = self._vol_builder.compute_skew(surface, target_dte=30)
        term = self._vol_builder.compute_term_structure(surface)

        signals: list[TradeSignal] = []
        now = datetime.utcnow().isoformat()

        # ── Strategy 1: High IV Rank Premium Selling ──────────
        if iv_rank > 50:
            sig = self._high_iv_premium_sell(symbol, spot, chain, iv_rank, skew, now)
            if sig:
                signals.append(sig)

        # ── Strategy 2: Volatility Skew Anomaly ──────────────
        if skew and skew.skew > 5.0:
            sig = self._skew_anomaly(symbol, spot, chain, iv_rank, skew, now)
            if sig:
                signals.append(sig)

        # ── Strategy 3: Unusual Volume ───────────────────────
        vol_signals = self._unusual_volume(symbol, spot, chain, iv_rank, now)
        signals.extend(vol_signals)

        # ── Strategy 4: Cheap Volatility (negative VRP) ──────
        atm_iv = surface.iv_rank  # use rank as proxy; get actual ATM IV below
        atm_points = [p for p in surface.points if 0.97 < p.moneyness < 1.03 and 25 < p.days_to_expiry < 45]
        if atm_points:
            current_iv = np.mean([p.iv for p in atm_points])
            if rv > 0 and current_iv < rv:
                sig = self._cheap_vol(symbol, spot, chain, iv_rank, current_iv, rv, now)
                if sig:
                    signals.append(sig)

        # ── Strategy 5: Term Structure Plays ─────────────────
        if term and term.is_inverted and term.ratio_30_90 > 1.10:
            sig = self._term_structure_play(symbol, spot, chain, iv_rank, term, now)
            if sig:
                signals.append(sig)

        return signals

    async def scan_watchlist(self, symbols: list[str]) -> SignalResponse:
        """Scan multiple symbols and aggregate signals."""
        all_signals: list[TradeSignal] = []
        for sym in symbols:
            try:
                sigs = await self.scan_symbol(sym)
                all_signals.extend(sigs)
            except Exception as e:
                log.warning("signal_scan_error", symbol=sym, error=str(e))

        # Sort by confidence then expected value
        confidence_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        all_signals.sort(key=lambda s: (confidence_order.get(s.confidence, 3), -s.expected_value))

        return SignalResponse(
            signals=all_signals,
            scan_timestamp=datetime.utcnow().isoformat(),
            symbols_scanned=len(symbols),
        )

    # ─── Signal Generators ───────────────────────────────────

    def _high_iv_premium_sell(self, symbol, spot, chain, iv_rank, skew, now) -> TradeSignal | None:
        """Iron Condor or Short Strangle when IV rank is elevated."""
        calls = sorted(chain.calls, key=lambda c: c.strike)
        puts = sorted(chain.puts, key=lambda p: p.strike)

        # Find ~16 delta short strikes
        short_call = self._find_by_delta(calls, target=0.16, is_call=True)
        short_put = self._find_by_delta(puts, target=-0.16, is_call=False)
        # Find ~5 delta long strikes (wings)
        long_call = self._find_by_delta(calls, target=0.05, is_call=True)
        long_put = self._find_by_delta(puts, target=-0.05, is_call=False)

        if not all([short_call, short_put, long_call, long_put]):
            return None

        # Credit received (use mid prices)
        credit = (
            _mid(short_call) + _mid(short_put)
            - _mid(long_call) - _mid(long_put)
        )
        if credit <= 0:
            return None

        call_width = long_call.strike - short_call.strike
        put_width = short_put.strike - long_put.strike
        max_width = max(call_width, put_width)
        max_risk = max_width - credit

        # Approximate P(profit): probability of staying between short strikes
        # Use short deltas as rough proxy
        prob_profit = 1.0 - abs(short_call.greeks.delta) - abs(short_put.greeks.delta)
        prob_profit = max(0.1, min(prob_profit, 0.95))

        expected_value = prob_profit * credit - (1 - prob_profit) * max_risk

        # Confidence: HIGH if IV rank > 70 and confirming skew, else MEDIUM
        confirming = iv_rank > 70 or (skew and abs(skew.skew) < 8)
        confidence = "HIGH" if iv_rank > 70 and confirming else "MEDIUM"

        expiry_str = chain.expiry.isoformat()
        return TradeSignal(
            ticker=symbol,
            strategy_name="Iron Condor",
            direction="neutral",
            confidence=confidence,
            suggested_legs=[
                {"strike": long_put.strike, "expiry": expiry_str, "option_type": "put", "side": "buy"},
                {"strike": short_put.strike, "expiry": expiry_str, "option_type": "put", "side": "sell"},
                {"strike": short_call.strike, "expiry": expiry_str, "option_type": "call", "side": "sell"},
                {"strike": long_call.strike, "expiry": expiry_str, "option_type": "call", "side": "buy"},
            ],
            expected_value=round(expected_value, 2),
            max_risk=round(max_risk, 2),
            probability_of_profit=round(prob_profit * 100, 1),
            iv_rank=round(iv_rank, 1),
            reasoning=f"IV Rank at {iv_rank:.0f}% — elevated premium makes Iron Condor attractive. "
                      f"Credit ${credit:.2f}, max risk ${max_risk:.2f}, ~{prob_profit*100:.0f}% P(profit).",
            timestamp=now,
        )

    def _skew_anomaly(self, symbol, spot, chain, iv_rank, skew, now) -> TradeSignal | None:
        """Put spread when put skew is abnormally steep."""
        puts = sorted(chain.puts, key=lambda p: p.strike)

        # Sell the rich 25-delta put, buy the cheaper far-OTM put
        short_put = self._find_by_delta(puts, target=-0.25, is_call=False)
        long_put = self._find_by_delta(puts, target=-0.10, is_call=False)

        if not short_put or not long_put or long_put.strike >= short_put.strike:
            return None

        credit = _mid(short_put) - _mid(long_put)
        if credit <= 0:
            return None

        width = short_put.strike - long_put.strike
        max_risk = width - credit
        prob_profit = 1.0 - abs(short_put.greeks.delta)
        prob_profit = max(0.1, min(prob_profit, 0.95))
        expected_value = prob_profit * credit - (1 - prob_profit) * max_risk

        confidence = "HIGH" if skew.skew > 8 and iv_rank > 40 else "MEDIUM"
        expiry_str = chain.expiry.isoformat()

        return TradeSignal(
            ticker=symbol,
            strategy_name="Bull Put Spread",
            direction="bullish",
            confidence=confidence,
            suggested_legs=[
                {"strike": long_put.strike, "expiry": expiry_str, "option_type": "put", "side": "buy"},
                {"strike": short_put.strike, "expiry": expiry_str, "option_type": "put", "side": "sell"},
            ],
            expected_value=round(expected_value, 2),
            max_risk=round(max_risk, 2),
            probability_of_profit=round(prob_profit * 100, 1),
            iv_rank=round(iv_rank, 1),
            reasoning=f"Put skew at {skew.skew:.1f}% — puts are rich relative to calls. "
                      f"Sell the steep skew via bull put spread for ${credit:.2f} credit.",
            timestamp=now,
        )

    def _unusual_volume(self, symbol, spot, chain, iv_rank, now) -> list[TradeSignal]:
        """Flag contracts with volume > 3x open interest."""
        signals: list[TradeSignal] = []

        all_contracts = list(chain.calls) + list(chain.puts)
        for c in all_contracts:
            if c.open_interest < 100:
                continue
            if c.volume > 3 * c.open_interest:
                is_call = c.option_type.value == "call"
                direction = "bullish" if is_call else "bearish"

                # Simple directional play: just flag the contract
                premium = _mid(c)
                max_risk = premium
                # For a long option, profit is unlimited (call) or strike-premium (put)
                # Approximate expected value conservatively
                delta = abs(c.greeks.delta)
                prob_profit = delta  # rough proxy
                expected_profit = premium * 1.5  # target 150% of premium
                expected_value = prob_profit * expected_profit - (1 - prob_profit) * max_risk

                expiry_str = chain.expiry.isoformat()
                signals.append(TradeSignal(
                    ticker=symbol,
                    strategy_name="Directional (Unusual Volume)",
                    direction=direction,
                    confidence="LOW",
                    suggested_legs=[
                        {"strike": c.strike, "expiry": expiry_str,
                         "option_type": c.option_type.value, "side": "buy"},
                    ],
                    expected_value=round(expected_value, 2),
                    max_risk=round(max_risk, 2),
                    probability_of_profit=round(prob_profit * 100, 1),
                    iv_rank=round(iv_rank, 1),
                    reasoning=f"Unusual volume on {symbol} {c.strike} {c.option_type.value}: "
                              f"{c.volume:,} vol vs {c.open_interest:,} OI ({c.volume/c.open_interest:.1f}x). "
                              f"Possible informed flow.",
                    timestamp=now,
                ))

        return signals[:3]  # Cap at 3 unusual volume signals per symbol

    def _cheap_vol(self, symbol, spot, chain, iv_rank, current_iv, rv, now) -> TradeSignal | None:
        """Long straddle when IV is below realized vol."""
        calls = sorted(chain.calls, key=lambda c: c.strike)
        puts = sorted(chain.puts, key=lambda p: p.strike)

        atm_call = self._find_by_delta(calls, target=0.50, is_call=True)
        atm_put = self._find_by_delta(puts, target=-0.50, is_call=False)

        if not atm_call or not atm_put:
            return None

        debit = _mid(atm_call) + _mid(atm_put)
        if debit <= 0:
            return None

        # Max risk is the debit paid
        max_risk = debit
        # Estimate breakeven range
        upper_be = atm_call.strike + debit
        lower_be = atm_put.strike - debit
        move_needed_pct = (debit / spot) * 100

        # P(profit): rough estimate based on how cheap vol is
        vol_discount = (rv - current_iv) / rv if rv > 0 else 0
        prob_profit = min(0.55, 0.35 + vol_discount * 0.5)
        expected_profit = debit * 1.0  # target 100% return on debit
        expected_value = prob_profit * expected_profit - (1 - prob_profit) * max_risk

        confidence = "MEDIUM" if vol_discount > 0.15 else "LOW"
        expiry_str = chain.expiry.isoformat()

        return TradeSignal(
            ticker=symbol,
            strategy_name="Long Straddle",
            direction="neutral",
            confidence=confidence,
            suggested_legs=[
                {"strike": atm_call.strike, "expiry": expiry_str, "option_type": "call", "side": "buy"},
                {"strike": atm_put.strike, "expiry": expiry_str, "option_type": "put", "side": "buy"},
            ],
            expected_value=round(expected_value, 2),
            max_risk=round(max_risk, 2),
            probability_of_profit=round(prob_profit * 100, 1),
            iv_rank=round(iv_rank, 1),
            reasoning=f"IV ({current_iv:.1f}%) is below realized vol ({rv:.1f}%) — options are cheap. "
                      f"Long straddle costs ${debit:.2f}, needs {move_needed_pct:.1f}% move to profit.",
            timestamp=now,
        )

    def _term_structure_play(self, symbol, spot, chain, iv_rank, term, now) -> TradeSignal | None:
        """Calendar spread when term structure is inverted (backwardation)."""
        calls = sorted(chain.calls, key=lambda c: c.strike)

        atm_call = self._find_by_delta(calls, target=0.50, is_call=True)
        if not atm_call:
            return None

        # Calendar: sell near-term, buy far-term at same strike
        # Since we only have one chain, approximate the debit
        near_premium = _mid(atm_call)
        # Far-term premium is higher due to more time value
        far_premium = near_premium * 1.4  # approximate
        debit = far_premium - near_premium
        if debit <= 0:
            return None

        max_risk = debit
        # Calendar profits if near-term decays faster; roughly 40-50% success rate
        prob_profit = 0.45
        expected_profit = debit * 0.8
        expected_value = prob_profit * expected_profit - (1 - prob_profit) * max_risk

        confidence = "MEDIUM" if term.ratio_30_90 > 1.15 else "LOW"
        expiry_str = chain.expiry.isoformat()

        return TradeSignal(
            ticker=symbol,
            strategy_name="Calendar Spread",
            direction="neutral",
            confidence=confidence,
            suggested_legs=[
                {"strike": atm_call.strike, "expiry": expiry_str, "option_type": "call", "side": "sell"},
                {"strike": atm_call.strike, "expiry": "far_month", "option_type": "call", "side": "buy"},
            ],
            expected_value=round(expected_value, 2),
            max_risk=round(max_risk, 2),
            probability_of_profit=round(prob_profit * 100, 1),
            iv_rank=round(iv_rank, 1),
            reasoning=f"Term structure inverted — short-term IV ({term.short_term_iv:.1f}%) >> "
                      f"long-term IV ({term.long_term_iv:.1f}%), ratio {term.ratio_30_90:.2f}x. "
                      f"Calendar spread exploits mean-reversion of near-term vol.",
            timestamp=now,
        )

    # ─── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _find_by_delta(
        contracts: list[OptionContract],
        target: float,
        is_call: bool,
    ) -> OptionContract | None:
        """Find the contract closest to a target delta."""
        best = None
        best_dist = float("inf")

        for c in contracts:
            delta = c.greeks.delta
            dist = abs(delta - target)
            if dist < best_dist:
                best_dist = dist
                best = c

        return best


def _mid(contract: OptionContract) -> float:
    """Mid-price of a contract."""
    return round((contract.bid + contract.ask) / 2, 2)
