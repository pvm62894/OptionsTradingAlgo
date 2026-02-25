"use client";

import { usePortfolio } from "@/hooks/use-queries";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import type { Greeks, Position } from "@/lib/types";
import { motion } from "framer-motion";

function GreekGauge({
  label,
  symbol,
  value,
  color,
  maxAbsValue = 500,
}: {
  label: string;
  symbol: string;
  value: number;
  color: string;
  maxAbsValue?: number;
}) {
  const pct = Math.min(Math.abs(value) / maxAbsValue, 1) * 100;
  const isPositive = value >= 0;

  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-heading text-muted-foreground tracking-wider">{label}</span>
        <span className={cn("font-data text-[11px]", `text-${color}`)}>
          {isPositive ? "+" : ""}
          {value.toFixed(2)}
        </span>
      </div>
      <div className="h-1.5 bg-zinc-900 w-full">
        <motion.div
          className="h-full"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.3, ease: "easeOut" }}
          style={{
            backgroundColor: `var(--color-${color})`,
            marginLeft: isPositive ? "50%" : `${50 - pct / 2}%`,
            width: `${pct / 2}%`,
          }}
        />
      </div>
    </div>
  );
}

function PositionRow({ position }: { position: Position }) {
  const pnlPositive = position.unrealized_pnl >= 0;

  return (
    <tr className="border-b border-grid hover:bg-zinc-800/30 transition-colors duration-75 dense-row">
      <td className="px-2 font-data text-[11px] text-foreground">
        {position.underlying}
        <span className="text-muted-foreground ml-1">
          {position.strike.toFixed(0)}{position.option_type === "call" ? "C" : "P"}
        </span>
      </td>
      <td className={cn("px-2 font-data text-[11px] text-center", position.side === "buy" ? "text-profit" : "text-loss")}>
        {position.side === "buy" ? "+" : "-"}{position.quantity}
      </td>
      <td className="px-2 font-data text-[11px] text-right text-muted-foreground">
        {position.entry_price.toFixed(2)}
      </td>
      <td className="px-2 font-data text-[11px] text-right">
        {position.current_price.toFixed(2)}
      </td>
      <td className={cn("px-2 font-data text-[11px] text-right font-bold", pnlPositive ? "text-profit" : "text-loss")}>
        {pnlPositive ? "+" : ""}${position.unrealized_pnl.toFixed(0)}
      </td>
      <td className="px-2 font-data text-[10px] text-right text-muted-foreground">
        {position.greeks.delta.toFixed(1)}
      </td>
    </tr>
  );
}

export function PositionMonitor() {
  const { data: portfolio, isLoading } = usePortfolio();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground font-data text-sm">
        Loading portfolio...
      </div>
    );
  }

  if (!portfolio) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground font-data text-sm">
        No portfolio data
      </div>
    );
  }

  const pnlPositive = portfolio.total_pnl >= 0;

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-grid bg-surface shrink-0">
        <h2 className="font-heading text-xs text-muted-foreground">POSITIONS</h2>
        <div className="flex items-center gap-2">
          <span className="font-data text-[11px] text-muted-foreground">P&L</span>
          <Badge
            variant="outline"
            className={cn(
              "font-data text-xs px-2 border",
              pnlPositive
                ? "text-profit border-emerald-500/40 bg-emerald-500/10"
                : "text-loss border-rose-500/40 bg-rose-500/10"
            )}
          >
            {pnlPositive ? "+" : ""}${portfolio.total_pnl.toFixed(0)}
          </Badge>
        </div>
      </div>

      {/* Greeks gauges */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-2 px-3 py-2 border-b border-grid">
        <GreekGauge
          label="DELTA"
          symbol="Δ"
          value={portfolio.total_greeks.delta}
          color={portfolio.total_greeks.delta >= 0 ? "profit" : "loss"}
        />
        <GreekGauge
          label="GAMMA"
          symbol="Γ"
          value={portfolio.total_greeks.gamma}
          color="warning"
          maxAbsValue={50}
        />
        <GreekGauge
          label="THETA"
          symbol="Θ"
          value={portfolio.total_greeks.theta}
          color={portfolio.total_greeks.theta >= 0 ? "profit" : "loss"}
        />
        <GreekGauge
          label="VEGA"
          symbol="V"
          value={portfolio.total_greeks.vega}
          color="active"
        />
      </div>

      {/* Risk metrics */}
      <div className="flex justify-between px-3 py-1.5 border-b border-grid text-[10px]">
        <div>
          <span className="text-muted-foreground">MARGIN: </span>
          <span className="font-data text-foreground">${portfolio.margin_used.toLocaleString()}</span>
        </div>
        <div>
          <span className="text-muted-foreground">BP: </span>
          <span className={cn("font-data", portfolio.buying_power >= 0 ? "text-foreground" : "text-loss")}>
            ${portfolio.buying_power.toLocaleString()}
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">MAX LOSS: </span>
          <span className="font-data text-loss">${portfolio.max_loss.toLocaleString()}</span>
        </div>
      </div>

      {/* Positions table */}
      <ScrollArea className="flex-1 terminal-scroll">
        <table className="w-full border-collapse">
          <thead className="sticky top-0 bg-surface z-10">
            <tr className="border-b border-grid">
              <th className="px-2 py-1 text-left font-heading text-[9px] text-muted-foreground tracking-wider">CONTRACT</th>
              <th className="px-2 py-1 text-center font-heading text-[9px] text-muted-foreground tracking-wider">QTY</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] text-muted-foreground tracking-wider">ENTRY</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] text-muted-foreground tracking-wider">MARK</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] text-muted-foreground tracking-wider">P&L</th>
              <th className="px-2 py-1 text-right font-heading text-[9px] text-muted-foreground tracking-wider">Δ</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.positions.map((pos) => (
              <PositionRow key={pos.id} position={pos} />
            ))}
          </tbody>
        </table>
      </ScrollArea>
    </div>
  );
}
