"use client";

import { useVolatilitySurface } from "@/hooks/use-queries";
import { useMarketStore } from "@/stores/market-store";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useMemo, useState } from "react";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { TooltipProvider } from "@/components/ui/tooltip";

function ivToColor(iv: number, minIV: number, maxIV: number): string {
  const range = maxIV - minIV;
  const t = range > 0 ? (iv - minIV) / range : 0.5;

  // 4-stop gradient: deep blue → cyan → yellow → red
  // [0] #1e3a5f  [0.33] #22d3ee  [0.66] #eab308  [1.0] #f43f5e
  const stops = [
    { pos: 0, r: 30, g: 58, b: 95 },
    { pos: 0.33, r: 34, g: 211, b: 238 },
    { pos: 0.66, r: 234, g: 179, b: 8 },
    { pos: 1.0, r: 244, g: 63, b: 94 },
  ];

  let lo = stops[0], hi = stops[1];
  for (let i = 1; i < stops.length; i++) {
    if (t <= stops[i].pos) {
      lo = stops[i - 1];
      hi = stops[i];
      break;
    }
    if (i === stops.length - 1) {
      lo = stops[i - 1];
      hi = stops[i];
    }
  }

  const segT = hi.pos === lo.pos ? 0 : (t - lo.pos) / (hi.pos - lo.pos);
  const r = Math.round(lo.r + segT * (hi.r - lo.r));
  const g = Math.round(lo.g + segT * (hi.g - lo.g));
  const b = Math.round(lo.b + segT * (hi.b - lo.b));
  return `rgb(${r},${g},${b})`;
}

export function VolatilitySurfaceHeatmap() {
  const symbol = useMarketStore((s) => s.selectedSymbol);
  const { data: surface, isLoading } = useVolatilitySurface(symbol);
  const [hoveredPoint, setHoveredPoint] = useState<{ strike: number; dte: number; iv: number } | null>(null);

  // Build grid: X = DTE buckets, Y = moneyness buckets
  const { grid, dteBuckets, moneynessBuckets, minIV, maxIV } = useMemo(() => {
    if (!surface?.points?.length) {
      return { grid: new Map(), dteBuckets: [], moneynessBuckets: [], minIV: 0, maxIV: 50 };
    }

    // Group by DTE and moneyness
    const dteBucketMap = new Map<number, number[]>();
    let mn = Infinity, mx = 0;

    for (const p of surface.points) {
      const dteBucket = nearestDTE(p.days_to_expiry);
      const moneyBucket = Math.round(p.moneyness * 100) / 100;

      if (!dteBucketMap.has(dteBucket)) dteBucketMap.set(dteBucket, []);
      dteBucketMap.get(dteBucket)!.push(moneyBucket);

      mn = Math.min(mn, p.iv);
      mx = Math.max(mx, p.iv);
    }

    const dtes = [...new Set([...dteBucketMap.keys()])].sort((a, b) => a - b);
    const moneys = [...new Set(surface.points.map((p) => Math.round(p.moneyness * 100) / 100))].sort((a, b) => a - b);

    // Build lookup
    const g = new Map<string, number>();
    for (const p of surface.points) {
      const key = `${nearestDTE(p.days_to_expiry)}_${Math.round(p.moneyness * 100) / 100}`;
      const existing = g.get(key);
      if (!existing || Math.abs(p.moneyness - 1) < 0.01) {
        g.set(key, p.iv);
      }
    }

    return { grid: g, dteBuckets: dtes, moneynessBuckets: moneys, minIV: mn === Infinity ? 0 : mn, maxIV: mx };
  }, [surface]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground font-data text-sm">
        Loading surface...
      </div>
    );
  }

  if (!surface) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground font-data text-sm">
        No volatility data
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-grid bg-surface shrink-0">
        <h2 className="font-heading text-xs text-muted-foreground">IV SURFACE</h2>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="font-data text-[10px] border-grid">
            RANK {surface.iv_rank.toFixed(0)}
          </Badge>
          <Badge
            variant="outline"
            className={cn(
              "font-data text-[10px] border",
              surface.iv_rank > 70 ? "text-loss border-rose-500/40" :
              surface.iv_rank > 30 ? "text-warning border-amber-500/40" :
              "text-profit border-emerald-500/40"
            )}
          >
            {surface.iv_rank > 70 ? "EXPENSIVE" : surface.iv_rank > 30 ? "FAIR" : "CHEAP"}
          </Badge>
        </div>
      </div>

      {/* Heatmap */}
      <ScrollArea className="flex-1 terminal-scroll">
        <TooltipProvider delayDuration={0}>
          <div className="p-3">
            {/* Column headers (DTE) */}
            <div className="flex ml-14">
              {dteBuckets.map((dte) => (
                <div key={dte} className="w-10 text-center font-data text-[9px] text-muted-foreground">
                  {dte}d
                </div>
              ))}
            </div>

            {/* Rows (moneyness) */}
            {moneynessBuckets.map((money) => (
              <div key={money} className="flex items-center">
                <div className="w-14 text-right pr-2 font-data text-[9px] text-muted-foreground shrink-0">
                  {((money - 1) * 100).toFixed(0)}%
                </div>
                {dteBuckets.map((dte) => {
                  const key = `${dte}_${money}`;
                  const iv = grid.get(key);
                  const isATM = Math.abs(money - 1) < 0.015;

                  return (
                    <Tooltip key={key}>
                      <TooltipTrigger asChild>
                        <div
                          className={cn(
                            "w-10 h-7 border border-void/50 flex items-center justify-center cursor-crosshair transition-transform hover:scale-110 hover:z-10",
                            isATM && "ring-1 ring-active/40"
                          )}
                          style={{
                            backgroundColor: iv ? ivToColor(iv, minIV, maxIV) : "#18181b",
                            opacity: iv ? 0.85 : 0.2,
                          }}
                          onMouseEnter={() => iv && setHoveredPoint({ strike: money, dte, iv })}
                          onMouseLeave={() => setHoveredPoint(null)}
                        >
                          {iv && (
                            <span className="font-data text-[8px] text-void font-bold mix-blend-difference">
                              {iv.toFixed(0)}
                            </span>
                          )}
                        </div>
                      </TooltipTrigger>
                      {iv && (
                        <TooltipContent className="bg-surface border-grid font-data text-[11px] p-2" side="top">
                          <div>IV: {iv.toFixed(1)}%</div>
                          <div>DTE: {dte}d</div>
                          <div>Moneyness: {((money - 1) * 100).toFixed(1)}%</div>
                        </TooltipContent>
                      )}
                    </Tooltip>
                  );
                })}
              </div>
            ))}

            {/* Color legend */}
            <div className="flex items-center justify-center mt-3 gap-2">
              <span className="text-[9px] font-data text-muted-foreground">{minIV.toFixed(0)}%</span>
              <div className="flex h-3 w-40">
                {Array.from({ length: 20 }, (_, i) => {
                  const pct = i / 19;
                  const iv = minIV + pct * (maxIV - minIV);
                  return (
                    <div
                      key={i}
                      className="flex-1"
                      style={{ backgroundColor: ivToColor(iv, minIV, maxIV) }}
                    />
                  );
                })}
              </div>
              <span className="text-[9px] font-data text-muted-foreground">{maxIV.toFixed(0)}%</span>
            </div>
          </div>
        </TooltipProvider>
      </ScrollArea>
    </div>
  );
}

function nearestDTE(dte: number): number {
  const buckets = [7, 14, 30, 45, 60, 90, 120, 180, 365];
  let nearest = buckets[0];
  let minDiff = Math.abs(dte - buckets[0]);
  for (const b of buckets) {
    const diff = Math.abs(dte - b);
    if (diff < minDiff) {
      minDiff = diff;
      nearest = b;
    }
  }
  return nearest;
}
