import { create } from "zustand";
import type { TickMessage, VolatilityRegime } from "@/lib/types";

interface MarketState {
  prices: Record<string, TickMessage>;
  selectedSymbol: string;
  regime: VolatilityRegime;
  connected: boolean;

  updateTick: (tick: TickMessage) => void;
  setSelectedSymbol: (symbol: string) => void;
  setRegime: (regime: VolatilityRegime) => void;
  setConnected: (connected: boolean) => void;
}

export const useMarketStore = create<MarketState>((set) => ({
  prices: {},
  selectedSymbol: "SPY",
  regime: "low_vol_trending",
  connected: false,

  updateTick: (tick) =>
    set((state) => ({
      prices: { ...state.prices, [tick.symbol]: tick },
    })),

  setSelectedSymbol: (symbol) => set({ selectedSymbol: symbol }),
  setRegime: (regime) => set({ regime }),
  setConnected: (connected) => set({ connected }),
}));
