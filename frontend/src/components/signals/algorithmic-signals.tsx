"use client";

import { useFullScan } from "@/hooks/use-queries";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useState } from "react";
import type { TradeSignal } from "@/lib/types";
import { ChevronDown, ChevronRight, Zap, TrendingUp, TrendingDown, Minus } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

const DIRECTION_CONFIG = {
  bullish: { color: "text-profit", bg: "bg-emerald-500/10", border: "border-emerald-500/30", icon: TrendingUp },
  bearish: { color: "text-loss", bg: "bg-rose-500/10", border: "border-rose-500/30", icon: TrendingDown },
  neutral: { color: "text-active", bg: "bg-cyan-500/10", border: "border-cyan-500/30", icon: Minus },
} as const;

const CONFIDENCE_CONFIG = {
  HIGH: { color: "text-profit", bg: "bg-emerald-500/15", border: "border-emerald-500/40" },
  MEDIUM: { color: "text-warning", bg: "bg-amber-500/15", border: "border-amber-500/40" },
  LOW: { color: "text-zinc-400", bg: "bg-zinc-500/10", border: "border-zinc-600" },
} as const;

function SignalRow({ signal, index }: { signal: TradeSignal; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const dir = DIRECTION_CONFIG[signal.direction];
  const conf = CONFIDENCE_CONFIG[signal.confidence];
  const DirIcon = dir.icon;

  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.1, delay: index * 0.03 }}
      className="border-b border-grid"
    >
      {/* Main row */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-zinc-800/40 transition-colors text-left"
      >
        {expanded ? (
          <ChevronDown className="w-3 h-3 text-zinc-500 shrink-0" />
        ) : (
          <ChevronRight className="w-3 h-3 text-zinc-500 shrink-0" />
        )}

        {/* Ticker */}
        <span className="font-data text-[11px] text-foreground w-10 shrink-0 font-bold">
          {signal.ticker}
        </span>

        {/* Strategy */}
        <span className="font-data text-[11px] text-zinc-300 truncate w-32 shrink-0">
          {signal.strategy_name}
        </span>

        {/* Direction */}
        <Badge variant="outline" className={cn("text-[9px] px-1.5 py-0 font-data shrink-0", dir.color, dir.border, dir.bg)}>
          <DirIcon className="w-2.5 h-2.5 mr-0.5" />
          {signal.direction.toUpperCase()}
        </Badge>

        {/* Confidence */}
        <Badge variant="outline" className={cn("text-[9px] px-1.5 py-0 font-data shrink-0", conf.color, conf.border, conf.bg)}>
          {signal.confidence}
        </Badge>

        {/* Spacer */}
        <div className="flex-1" />

        {/* P(Profit) */}
        <span className={cn("font-data text-[11px] w-12 text-right shrink-0", signal.probability_of_profit > 60 ? "text-profit" : "text-zinc-400")}>
          {signal.probability_of_profit.toFixed(0)}%
        </span>

        {/* EV */}
        <span className={cn("font-data text-[11px] w-16 text-right shrink-0", signal.expected_value > 0 ? "text-profit" : "text-loss")}>
          {signal.expected_value > 0 ? "+" : ""}${signal.expected_value.toFixed(2)}
        </span>
      </button>

      {/* Expanded detail */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.15 }}
            className="overflow-hidden"
          >
            <div className="px-3 pb-3 pt-1 ml-5 space-y-2">
              {/* Reasoning */}
              <p className="font-data text-[10px] text-zinc-400 leading-relaxed">
                {signal.reasoning}
              </p>

              {/* Metrics row */}
              <div className="flex gap-4 text-[10px] font-data">
                <div>
                  <span className="text-zinc-500">MAX RISK </span>
                  <span className="text-loss">${signal.max_risk.toFixed(2)}</span>
                </div>
                <div>
                  <span className="text-zinc-500">IV RANK </span>
                  <span className="text-zinc-300">{signal.iv_rank.toFixed(0)}%</span>
                </div>
                <div>
                  <span className="text-zinc-500">P(PROFIT) </span>
                  <span className="text-profit">{signal.probability_of_profit.toFixed(1)}%</span>
                </div>
              </div>

              {/* Legs */}
              {signal.suggested_legs.length > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {signal.suggested_legs.map((leg, i) => (
                    <span
                      key={i}
                      className={cn(
                        "text-[9px] font-data px-1.5 py-0.5 border rounded-sm",
                        leg.side === "buy"
                          ? "text-profit border-emerald-500/30 bg-emerald-500/5"
                          : "text-loss border-rose-500/30 bg-rose-500/5"
                      )}
                    >
                      {leg.side.toUpperCase()} {leg.strike} {leg.option_type.toUpperCase()} {leg.expiry}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export function AlgorithmicSignals() {
  const { data, isLoading, isError } = useFullScan();

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <SignalsHeader count={0} />
        <div className="flex-1 p-4 space-y-2">
          {Array.from({ length: 5 }, (_, i) => (
            <div key={i} className="h-8 bg-zinc-900 animate-pulse rounded" />
          ))}
        </div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex flex-col h-full">
        <SignalsHeader count={0} />
        <div className="flex-1 flex items-center justify-center">
          <p className="text-zinc-500 font-data text-xs">Failed to load signals</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <SignalsHeader count={data.signals.length} scanned={data.symbols_scanned} />

      {data.signals.length === 0 ? (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <Zap className="w-6 h-6 mx-auto mb-2 text-zinc-600" />
            <p className="text-zinc-500 font-data text-xs">No signals detected</p>
            <p className="text-zinc-600 font-data text-[10px] mt-1">Markets are quiet</p>
          </div>
        </div>
      ) : (
        <ScrollArea className="flex-1 terminal-scroll">
          {/* Column headers */}
          <div className="flex items-center gap-2 px-3 py-1 border-b border-grid text-zinc-500 sticky top-0 bg-surface z-10">
            <span className="w-3" />
            <span className="font-heading text-[9px] tracking-wider w-10">TICK</span>
            <span className="font-heading text-[9px] tracking-wider w-32">STRATEGY</span>
            <span className="font-heading text-[9px] tracking-wider w-16">DIR</span>
            <span className="font-heading text-[9px] tracking-wider w-12">CONF</span>
            <div className="flex-1" />
            <span className="font-heading text-[9px] tracking-wider w-12 text-right">P(W)</span>
            <span className="font-heading text-[9px] tracking-wider w-16 text-right">EV</span>
          </div>

          {data.signals.map((signal, i) => (
            <SignalRow key={`${signal.ticker}-${signal.strategy_name}-${i}`} signal={signal} index={i} />
          ))}
        </ScrollArea>
      )}
    </div>
  );
}

function SignalsHeader({ count, scanned }: { count: number; scanned?: number }) {
  return (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-grid bg-surface shrink-0">
      <div className="flex items-center gap-2">
        <Zap className="w-3.5 h-3.5 text-active" />
        <h2 className="font-heading text-xs text-muted-foreground">ALGORITHMIC SIGNALS</h2>
      </div>
      <div className="flex items-center gap-2">
        {scanned !== undefined && (
          <span className="font-data text-[10px] text-zinc-600">{scanned} SCANNED</span>
        )}
        <Badge variant="outline" className={cn(
          "font-data text-[10px] border-grid",
          count > 0 ? "text-active border-cyan-500/30" : "text-zinc-500"
        )}>
          {count} SIGNAL{count !== 1 ? "S" : ""}
        </Badge>
      </div>
    </div>
  );
}
