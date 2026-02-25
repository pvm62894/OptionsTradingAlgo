"use client";

import { useMarketStore } from "@/stores/market-store";
import { useSymbols, useVolatilityPrediction } from "@/hooks/use-queries";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useEffect, useRef, useState } from "react";
import { Activity, Search, Wifi, WifiOff } from "lucide-react";

const REGIME_CONFIG = {
  low_vol_trending: { label: "LOW VOL", className: "bg-emerald-500/20 text-emerald-400 border-emerald-500/40 glow-green" },
  high_vol_mean_reverting: { label: "HIGH VOL", className: "bg-rose-500/20 text-rose-400 border-rose-500/40 glow-red" },
  crisis: { label: "CRISIS", className: "bg-amber-500/20 text-amber-400 border-amber-500/40 glow-amber" },
};

export function MarketPulseHeader() {
  const { selectedSymbol, setSelectedSymbol, prices, connected, regime } = useMarketStore();
  const { data: symbols } = useSymbols();
  const { data: prediction } = useVolatilityPrediction(selectedSymbol);
  const [flashClass, setFlashClass] = useState("");
  const prevPrice = useRef<number | null>(null);

  const tick = prices[selectedSymbol];
  const currentPrice = tick?.price ?? 0;
  const change = tick?.change ?? 0;
  const changePct = tick?.change_pct ?? 0;

  // Flash on price change
  useEffect(() => {
    if (prevPrice.current !== null && currentPrice !== prevPrice.current) {
      setFlashClass(currentPrice > prevPrice.current ? "tick-up" : "tick-down");
      const timer = setTimeout(() => setFlashClass(""), 150);
      return () => clearTimeout(timer);
    }
    prevPrice.current = currentPrice;
  }, [currentPrice]);

  const currentRegime = prediction?.regime ?? regime;
  const regimeInfo = REGIME_CONFIG[currentRegime];

  return (
    <header className="h-12 border-b border-grid bg-surface flex items-center px-4 gap-4 shrink-0">
      {/* Left: Symbol selector */}
      <div className="flex items-center gap-2">
        <Activity className="w-4 h-4 text-active" />
        <select
          value={selectedSymbol}
          onChange={(e) => setSelectedSymbol(e.target.value)}
          className="bg-void border border-grid text-foreground font-data text-sm px-2 py-1 focus:outline-none focus:border-active"
        >
          {(symbols ?? []).map((s) => (
            <option key={s.symbol} value={s.symbol}>
              {s.symbol}
            </option>
          ))}
          {!symbols && <option value="SPY">SPY</option>}
        </select>
      </div>

      {/* Center: Large price display */}
      <div className="flex-1 flex items-center justify-center gap-3">
        <span className="font-heading text-sm text-muted-foreground tracking-wider">
          {selectedSymbol}
        </span>
        <span className={cn("font-data text-3xl font-bold tracking-tight transition-colors", flashClass)}>
          {currentPrice > 0 ? currentPrice.toFixed(2) : "---"}
        </span>
        {currentPrice > 0 && (
          <Badge
            variant="outline"
            className={cn(
              "font-data text-xs border px-2 py-0.5",
              change >= 0
                ? "text-profit border-emerald-500/40 bg-emerald-500/10"
                : "text-loss border-rose-500/40 bg-rose-500/10"
            )}
          >
            {change >= 0 ? "+" : ""}
            {change.toFixed(2)} ({changePct >= 0 ? "+" : ""}
            {changePct.toFixed(2)}%)
          </Badge>
        )}
      </div>

      {/* Right: Regime badge + connection status */}
      <div className="flex items-center gap-3">
        <Badge
          variant="outline"
          className={cn("font-heading text-[10px] tracking-widest px-3 py-1 border", regimeInfo.className)}
        >
          {regimeInfo.label}
        </Badge>

        <div className="flex items-center gap-1">
          {connected ? (
            <Wifi className="w-3.5 h-3.5 text-profit" />
          ) : (
            <WifiOff className="w-3.5 h-3.5 text-loss" />
          )}
          <span className="font-data text-[10px] text-muted-foreground">
            {connected ? "LIVE" : "DISCONNECTED"}
          </span>
        </div>
      </div>
    </header>
  );
}
