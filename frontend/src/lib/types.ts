// ─── Core Types for QuantumFlow ──────────────────────────

export interface Greeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
}

export interface OptionContract {
  symbol: string;
  underlying: string;
  strike: number;
  expiry: string;
  option_type: "call" | "put";
  bid: number;
  ask: number;
  last: number;
  volume: number;
  open_interest: number;
  implied_volatility: number;
  greeks: Greeks;
  in_the_money: boolean;
}

export interface OptionChain {
  underlying: string;
  underlying_price: number;
  expiry: string;
  calls: OptionContract[];
  puts: OptionContract[];
  timestamp: string;
}

export interface Quote {
  symbol: string;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  timestamp: string;
}

export interface StrategyLeg {
  option_type: "call" | "put";
  strike: number;
  expiry: string;
  side: "buy" | "sell";
  quantity: number;
  premium: number;
}

export interface PayoffPoint {
  price: number;
  pnl: number;
}

export interface PayoffCurve {
  label: string;
  days_remaining: number;
  points: PayoffPoint[];
}

export interface StrategyAnalysis {
  legs: StrategyLeg[];
  underlying_price: number;
  max_profit: number | null;
  max_loss: number | null;
  breakeven_points: number[];
  payoff_data: PayoffCurve[];
  total_greeks: Greeks;
  net_premium: number;
}

export interface Position {
  id: string;
  symbol: string;
  underlying: string;
  option_type: "call" | "put";
  strike: number;
  expiry: string;
  side: "buy" | "sell";
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  greeks: Greeks;
}

export interface PortfolioSummary {
  positions: Position[];
  total_greeks: Greeks;
  total_pnl: number;
  margin_used: number;
  buying_power: number;
  max_loss: number;
}

export interface VolSurfacePoint {
  strike: number;
  expiry: string;
  days_to_expiry: number;
  iv: number;
  moneyness: number;
}

export interface VolatilitySurface {
  underlying: string;
  spot_price: number;
  points: VolSurfacePoint[];
  iv_rank: number;
  iv_percentile: number;
  timestamp: string;
}

export interface TickMessage {
  type: "tick";
  symbol: string;
  price: number;
  bid: number;
  ask: number;
  change: number;
  change_pct: number;
  volume: number;
  timestamp: string;
}

export interface SymbolInfo {
  symbol: string;
  price: number;
  volatility: number;
}

export type VolatilityRegime =
  | "low_vol_trending"
  | "high_vol_mean_reverting"
  | "crisis";

export interface VolatilityPrediction {
  symbol: string;
  regime: VolatilityRegime;
  regime_probabilities: Record<string, number>;
  predicted_iv_30d: number;
  confidence: number;
  features_importance: Record<string, number>;
  timestamp: string;
}

export interface BacktestConfig {
  strategy_name: string;
  symbol: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  stop_loss_pct?: number;
  take_profit_pct?: number;
}

export interface VolSurface3DPoint {
  strike: number;
  dte: number;
  iv: number;
  volume: number;
  open_interest: number;
}

export interface VolSurface3DResponse {
  symbol: string;
  points: VolSurface3DPoint[];
  spot_price: number;
}

// ─── Algorithmic Signals ──────────────────────────────────

export interface SignalLeg {
  strike: number;
  expiry: string;
  option_type: "call" | "put";
  side: "buy" | "sell";
}

export interface TradeSignal {
  ticker: string;
  strategy_name: string;
  direction: "bullish" | "bearish" | "neutral";
  confidence: "HIGH" | "MEDIUM" | "LOW";
  suggested_legs: SignalLeg[];
  expected_value: number;
  max_risk: number;
  probability_of_profit: number;
  iv_rank: number;
  reasoning: string;
  timestamp: string;
}

export interface SignalResponse {
  signals: TradeSignal[];
  scan_timestamp: string;
  symbols_scanned: number;
}

export interface BacktestResult {
  config: BacktestConfig;
  total_return: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  max_drawdown: number;
  win_rate: number;
  profit_factor: number;
  total_trades: number;
  equity_curve: { date: string; equity: number; drawdown: number }[];
  trades: Record<string, unknown>[];
  greeks_over_time: Record<string, unknown>[];
}
