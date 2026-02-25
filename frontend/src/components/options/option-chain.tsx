"use client";

import { useOptionChain } from "@/hooks/use-queries";
import { useMarketStore } from "@/stores/market-store";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { OptionContract } from "@/lib/types";
import { useMemo, useRef, useEffect } from "react";
import { motion } from "framer-motion";

function formatNum(n: number, decimals = 2): string {
  return n.toFixed(decimals);
}

function VolBar({ value, max, color }: { value: number; max: number; color: string }) {
  const pct = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  return (
    <div className="relative w-full h-full">
      <div
        className="absolute inset-y-0 left-0 opacity-20"
        style={{ width: `${pct}%`, backgroundColor: color }}
      />
      <span className="relative z-10 font-data text-[11px]">
        {value > 0 ? value.toLocaleString() : "—"}
      </span>
    </div>
  );
}

function IVHeatCell({ iv, minIV, maxIV }: { iv: number; minIV: number; maxIV: number }) {
  const range = maxIV - minIV;
  const intensity = range > 0 ? (iv - minIV) / range : 0.5;
  // Cool (cyan) to Hot (rose)
  const r = Math.round(intensity * 244);
  const g = Math.round((1 - intensity) * 100 + intensity * 63);
  const b = Math.round((1 - intensity) * 238 + intensity * 94);

  return (
    <td
      className="px-2 text-right font-data text-[11px] border-r border-grid"
      style={{ backgroundColor: `rgba(${r},${g},${b},0.12)` }}
    >
      {formatNum(iv, 1)}%
    </td>
  );
}

function ContractRow({
  call,
  put,
  strike,
  spotPrice,
  maxVolume,
  minIV,
  maxIV,
  index,
}: {
  call?: OptionContract;
  put?: OptionContract;
  strike: number;
  spotPrice: number;
  maxVolume: number;
  minIV: number;
  maxIV: number;
  index: number;
}) {
  const isATM = Math.abs(strike - spotPrice) <= 2.5;
  const callITM = strike < spotPrice;
  const putITM = strike > spotPrice;

  return (
    <motion.tr
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.08, delay: index * 0.01 }}
      className={cn(
        "dense-row border-b border-grid hover:bg-zinc-800/50 transition-colors duration-75",
        isATM && "bg-active/5 border-active/20"
      )}
    >
      {/* CALLS side */}
      <td className={cn("px-2 text-right font-data text-[11px] border-r border-grid", callITM ? "bg-zinc-800/40" : "")}>
        <span className={cn(call && call.greeks.delta > 0 ? "text-emerald-400" : "text-zinc-500")}>
          {call ? formatNum(call.greeks.delta, 3) : "—"}
        </span>
      </td>
      <td className={cn("px-2 text-right font-data text-[11px] border-r border-grid", callITM ? "bg-zinc-800/40" : "")}>
        {call ? formatNum(call.bid) : "—"}
      </td>
      <td className={cn("px-2 text-right font-data text-[11px] border-r border-grid", callITM ? "bg-zinc-800/40" : "")}>
        {call ? formatNum(call.ask) : "—"}
      </td>
      {call ? (
        <IVHeatCell iv={call.implied_volatility} minIV={minIV} maxIV={maxIV} />
      ) : (
        <td className="px-2 text-right font-data text-[11px] border-r border-grid">—</td>
      )}
      <td className={cn("px-2 text-right border-r border-grid", callITM ? "bg-zinc-800/40" : "")}>
        <VolBar value={call?.volume ?? 0} max={maxVolume} color="#22d3ee" />
      </td>

      {/* STRIKE column */}
      <td
        className={cn(
          "px-3 text-center font-data text-[12px] font-bold border-r border-l border-grid",
          isATM ? "text-active bg-active/10" : "text-foreground"
        )}
      >
        {formatNum(strike, strike % 1 === 0 ? 0 : 2)}
      </td>

      {/* PUTS side */}
      <td className={cn("px-2 text-right border-r border-grid", putITM ? "bg-zinc-800/40" : "")}>
        <VolBar value={put?.volume ?? 0} max={maxVolume} color="#f43f5e" />
      </td>
      {put ? (
        <IVHeatCell iv={put.implied_volatility} minIV={minIV} maxIV={maxIV} />
      ) : (
        <td className="px-2 text-right font-data text-[11px] border-r border-grid">—</td>
      )}
      <td className={cn("px-2 text-right font-data text-[11px] border-r border-grid", putITM ? "bg-zinc-800/40" : "")}>
        {put ? formatNum(put.bid) : "—"}
      </td>
      <td className={cn("px-2 text-right font-data text-[11px] border-r border-grid", putITM ? "bg-zinc-800/40" : "")}>
        {put ? formatNum(put.ask) : "—"}
      </td>
      <td className={cn("px-2 text-right font-data text-[11px]", putITM ? "bg-zinc-800/40" : "")}>
        <span className={cn(put && put.greeks.delta < 0 ? "text-rose-400" : "text-zinc-500")}>
          {put ? formatNum(put.greeks.delta, 3) : "—"}
        </span>
      </td>
    </motion.tr>
  );
}

export function OptionChainTable() {
  const symbol = useMarketStore((s) => s.selectedSymbol);
  const { data: chain, isLoading } = useOptionChain(symbol);
  const atmRef = useRef<HTMLTableRowElement>(null);

  // Build strike-indexed maps
  const { strikes, callMap, putMap, maxVolume, minIV, maxIV } = useMemo(() => {
    if (!chain) return { strikes: [], callMap: {}, putMap: {}, maxVolume: 0, minIV: 0, maxIV: 100 };

    const cm: Record<number, OptionContract> = {};
    const pm: Record<number, OptionContract> = {};
    let maxVol = 0;
    let mnIV = Infinity;
    let mxIV = 0;

    for (const c of chain.calls) {
      cm[c.strike] = c;
      maxVol = Math.max(maxVol, c.volume);
      if (c.implied_volatility > 0) {
        mnIV = Math.min(mnIV, c.implied_volatility);
        mxIV = Math.max(mxIV, c.implied_volatility);
      }
    }
    for (const p of chain.puts) {
      pm[p.strike] = p;
      maxVol = Math.max(maxVol, p.volume);
      if (p.implied_volatility > 0) {
        mnIV = Math.min(mnIV, p.implied_volatility);
        mxIV = Math.max(mxIV, p.implied_volatility);
      }
    }

    const allStrikes = new Set([...Object.keys(cm), ...Object.keys(pm)].map(Number));
    const sorted = [...allStrikes].sort((a, b) => a - b);

    return { strikes: sorted, callMap: cm, putMap: pm, maxVolume: maxVol, minIV: mnIV === Infinity ? 0 : mnIV, maxIV: mxIV };
  }, [chain]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground font-data text-sm">
        Loading chain...
      </div>
    );
  }

  if (!chain) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground font-data text-sm">
        No data available
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-grid bg-surface shrink-0">
        <h2 className="font-heading text-xs text-muted-foreground">OPTION CHAIN</h2>
        <span className="font-data text-[10px] text-muted-foreground">
          EXP {chain.expiry} · {strikes.length} STRIKES
        </span>
      </div>

      {/* Table */}
      <ScrollArea className="flex-1 terminal-scroll">
        <table className="w-full border-collapse text-[11px]">
          <thead className="sticky top-0 z-10 bg-surface">
            <tr className="border-b-2 border-grid">
              <th colSpan={5} className="text-center font-heading text-[10px] text-emerald-400 tracking-widest py-1 border-r border-grid">
                CALLS
              </th>
              <th className="font-heading text-[10px] text-muted-foreground tracking-widest py-1 border-r border-l border-grid">
                STRIKE
              </th>
              <th colSpan={5} className="text-center font-heading text-[10px] text-rose-400 tracking-widest py-1">
                PUTS
              </th>
            </tr>
            <tr className="border-b border-grid text-muted-foreground">
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">Δ</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">BID</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">ASK</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">IV</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">VOL</th>
              <th className="px-2 py-1 text-center font-heading text-[9px] tracking-wider border-r border-l border-grid">$</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">VOL</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">IV</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">BID</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider border-r border-grid">ASK</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] tracking-wider">Δ</th>
            </tr>
          </thead>
          <tbody>
            {strikes.map((strike, i) => (
              <ContractRow
                key={strike}
                call={callMap[strike]}
                put={putMap[strike]}
                strike={strike}
                spotPrice={chain.underlying_price}
                maxVolume={maxVolume}
                minIV={minIV}
                maxIV={maxIV}
                index={i}
              />
            ))}
          </tbody>
        </table>
      </ScrollArea>
    </div>
  );
}
