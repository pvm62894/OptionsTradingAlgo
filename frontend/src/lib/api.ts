// ─── API Client for QuantumFlow Backend ──────────────────

import type {
  OptionChain,
  PortfolioSummary,
  VolatilitySurface,
  StrategyAnalysis,
  StrategyLeg,
  SymbolInfo,
  VolatilityPrediction,
  BacktestConfig,
  BacktestResult,
  VolSurface3DResponse,
  SignalResponse,
} from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

async function fetchApi<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${await res.text()}`);
  }
  return res.json();
}

// ─── Market Data ─────────────────────────────────────────

export const api = {
  getSymbols: () => fetchApi<SymbolInfo[]>("/symbols"),

  getOptionChain: (symbol: string, expiry?: string) => {
    const params = new URLSearchParams({ symbol });
    if (expiry) params.set("expiry", expiry);
    return fetchApi<OptionChain>(`/options/chain?${params}`);
  },

  getQuote: (symbol: string) => fetchApi<Record<string, unknown>>(`/quote/${symbol}`),

  getHistorical: (symbol: string, days = 252) =>
    fetchApi<Record<string, unknown>[]>(`/historical/${symbol}?days=${days}`),

  // ─── Strategy ────────────────────────────────────────

  analyzeStrategy: (symbol: string, legs: StrategyLeg[], priceRangePct = 20, timeSteps = 6) =>
    fetchApi<StrategyAnalysis>("/strategy/analyze", {
      method: "POST",
      body: JSON.stringify({
        symbol,
        legs,
        price_range_pct: priceRangePct,
        time_steps: timeSteps,
      }),
    }),

  getStrategyTemplates: (symbol: string) =>
    fetchApi<Record<string, { name: string; description: string; legs: StrategyLeg[] }>>(
      `/strategy/templates?symbol=${symbol}`
    ),

  // ─── Portfolio ───────────────────────────────────────

  getPortfolioGreeks: () => fetchApi<PortfolioSummary>("/greeks/portfolio"),

  // ─── Volatility ──────────────────────────────────────

  getVolatilitySurface: (symbol: string) =>
    fetchApi<VolatilitySurface>(`/volatility/surface?symbol=${symbol}`),

  getVolSurface3D: (symbol: string) =>
    fetchApi<VolSurface3DResponse>(`/volatility/surface3d?symbol=${symbol}`),

  // ─── ML ──────────────────────────────────────────────

  predictVolatility: (symbol: string) =>
    fetchApi<VolatilityPrediction>("/predict/volatility", {
      method: "POST",
      body: JSON.stringify({ symbol }),
    }),

  // ─── Backtest ────────────────────────────────────────

  runBacktest: (config: BacktestConfig) =>
    fetchApi<BacktestResult>("/backtest/run", {
      method: "POST",
      body: JSON.stringify(config),
    }),

  // ─── Signals ────────────────────────────────────────

  getSignals: (symbols: string[]) =>
    fetchApi<SignalResponse>(`/signals?symbols=${symbols.join(",")}`),

  scanAllSignals: () => fetchApi<SignalResponse>("/signals/scan"),

  // ─── Health ──────────────────────────────────────────

  health: () => fetchApi<{ status: string }>("/health"),
};
