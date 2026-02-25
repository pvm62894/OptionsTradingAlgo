"use client";

import { useQuery, useMutation } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { BacktestConfig, StrategyLeg, VolSurface3DResponse, SignalResponse } from "@/lib/types";

export function useOptionChain(symbol: string, expiry?: string) {
  return useQuery({
    queryKey: ["optionChain", symbol, expiry],
    queryFn: () => api.getOptionChain(symbol, expiry),
    refetchInterval: 60_000, // Refresh every minute
    staleTime: 30_000,
  });
}

export function useVolatilitySurface(symbol: string) {
  return useQuery({
    queryKey: ["volSurface", symbol],
    queryFn: () => api.getVolatilitySurface(symbol),
    staleTime: 5 * 60_000,
  });
}

export function usePortfolio() {
  return useQuery({
    queryKey: ["portfolio"],
    queryFn: () => api.getPortfolioGreeks(),
    refetchInterval: 30_000,
  });
}

export function useSymbols() {
  return useQuery({
    queryKey: ["symbols"],
    queryFn: () => api.getSymbols(),
    staleTime: 5 * 60_000,
  });
}

export function useStrategyTemplates(symbol: string) {
  return useQuery({
    queryKey: ["strategyTemplates", symbol],
    queryFn: () => api.getStrategyTemplates(symbol),
    staleTime: 60_000,
  });
}

export function useVolatilityPrediction(symbol: string) {
  return useQuery({
    queryKey: ["volPrediction", symbol],
    queryFn: () => api.predictVolatility(symbol),
    staleTime: 5 * 60_000,
  });
}

export function useStrategyAnalysis() {
  return useMutation({
    mutationFn: ({ symbol, legs }: { symbol: string; legs: StrategyLeg[] }) =>
      api.analyzeStrategy(symbol, legs),
  });
}

export function useVolSurface3D(symbol: string) {
  return useQuery({
    queryKey: ["volSurface3D", symbol],
    queryFn: async () => {
      try {
        const res = await api.getVolSurface3D(symbol);
        return transformSurface3D(res);
      } catch {
        // Fallback: fetch 2D surface and transform it
        const surface = await api.getVolatilitySurface(symbol);
        return transformFlatSurface(surface);
      }
    },
    staleTime: 60_000,
    gcTime: 5 * 60_000,
  });
}

function transformSurface3D(res: VolSurface3DResponse) {
  const strikeSet = new Set<number>();
  const dteSet = new Set<number>();
  const lookup = new Map<string, { iv: number; volume: number; oi: number }>();

  for (const p of res.points) {
    strikeSet.add(p.strike);
    dteSet.add(p.dte);
    lookup.set(`${p.strike}_${p.dte}`, { iv: p.iv, volume: p.volume, oi: p.open_interest });
  }

  const strikes = [...strikeSet].sort((a, b) => a - b);
  const dtes = [...dteSet].sort((a, b) => a - b);

  const ivGrid: number[][] = [];
  const volumeGrid: number[][] = [];
  const oiGrid: number[][] = [];

  for (const dte of dtes) {
    const ivRow: number[] = [];
    const volRow: number[] = [];
    const oiRow: number[] = [];
    for (const strike of strikes) {
      const entry = lookup.get(`${strike}_${dte}`);
      ivRow.push(entry?.iv ?? 0);
      volRow.push(entry?.volume ?? 0);
      oiRow.push(entry?.oi ?? 0);
    }
    ivGrid.push(ivRow);
    volumeGrid.push(volRow);
    oiGrid.push(oiRow);
  }

  return { strikes, dtes, ivGrid, volumeGrid, oiGrid };
}

function transformFlatSurface(surface: Awaited<ReturnType<typeof api.getVolatilitySurface>>) {
  const strikeSet = new Set<number>();
  const dteSet = new Set<number>();
  const lookup = new Map<string, number>();

  for (const p of surface.points) {
    strikeSet.add(p.strike);
    dteSet.add(p.days_to_expiry);
    lookup.set(`${p.strike}_${p.days_to_expiry}`, p.iv);
  }

  const strikes = [...strikeSet].sort((a, b) => a - b);
  const dtes = [...dteSet].sort((a, b) => a - b);

  const ivGrid: number[][] = [];
  const volumeGrid: number[][] = [];
  const oiGrid: number[][] = [];

  for (const dte of dtes) {
    const ivRow: number[] = [];
    const volRow: number[] = [];
    const oiRow: number[] = [];
    for (const strike of strikes) {
      ivRow.push(lookup.get(`${strike}_${dte}`) ?? 0);
      volRow.push(0);
      oiRow.push(0);
    }
    ivGrid.push(ivRow);
    volumeGrid.push(volRow);
    oiGrid.push(oiRow);
  }

  return { strikes, dtes, ivGrid, volumeGrid, oiGrid };
}

export function useAlgorithmicSignals(symbols: string[]) {
  return useQuery({
    queryKey: ["signals", symbols.join(",")],
    queryFn: () => api.getSignals(symbols),
    refetchInterval: 30_000,
    staleTime: 15_000,
  });
}

export function useFullScan() {
  return useQuery({
    queryKey: ["signals", "fullScan"],
    queryFn: () => api.scanAllSignals(),
    refetchInterval: 60_000,
    staleTime: 30_000,
  });
}

export function useBacktest() {
  return useMutation({
    mutationFn: (config: BacktestConfig) => api.runBacktest(config),
  });
}
