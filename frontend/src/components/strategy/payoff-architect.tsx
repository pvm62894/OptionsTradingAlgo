"use client";

import { useStrategyStore } from "@/stores/strategy-store";
import { useMarketStore } from "@/stores/market-store";
import { useStrategyTemplates, useStrategyAnalysis } from "@/hooks/use-queries";
import { Slider } from "@/components/ui/slider";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useEffect, useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import { motion } from "framer-motion";
import { Plus, Trash2, TrendingUp } from "lucide-react";
import type { StrategyLeg } from "@/lib/types";

function LegBuilder() {
  const { legs, addLeg, removeLeg, updateLeg } = useStrategyStore();
  const symbol = useMarketStore((s) => s.selectedSymbol);
  const { data: templates } = useStrategyTemplates(symbol);

  const handleAddLeg = () => {
    addLeg({
      option_type: "call",
      strike: 590,
      expiry: new Date(Date.now() + 30 * 86400000).toISOString().split("T")[0],
      side: "buy",
      quantity: 1,
      premium: 5.0,
    });
  };

  const loadTemplate = (key: string) => {
    if (!templates?.[key]) return;
    const template = templates[key];
    useStrategyStore.getState().setLegs(template.legs);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-grid bg-surface">
        <h3 className="font-heading text-[10px] text-muted-foreground tracking-widest">LEGS</h3>
        <button
          onClick={handleAddLeg}
          className="flex items-center gap-1 text-[10px] font-data text-active hover:text-cyan-300 transition-colors"
        >
          <Plus className="w-3 h-3" /> ADD
        </button>
      </div>

      {/* Template buttons */}
      <div className="flex gap-1 px-3 py-1.5 border-b border-grid">
        {["iron_condor", "bull_call_spread", "straddle"].map((key) => (
          <button
            key={key}
            onClick={() => loadTemplate(key)}
            className="px-2 py-0.5 text-[9px] font-heading tracking-wider border border-grid hover:border-active hover:text-active transition-colors"
          >
            {key.replace(/_/g, " ").toUpperCase()}
          </button>
        ))}
      </div>

      {/* Leg list */}
      <div className="flex-1 overflow-y-auto terminal-scroll">
        {legs.length === 0 ? (
          <div className="flex items-center justify-center h-full text-muted-foreground text-[11px] font-data">
            Add legs or select a template
          </div>
        ) : (
          legs.map((leg, i) => (
            <LegRow key={i} leg={leg} index={i} onUpdate={updateLeg} onRemove={removeLeg} />
          ))
        )}
      </div>

      {/* Strategy Greeks summary */}
      <StrategyGreeksSummary />
    </div>
  );
}

function LegRow({
  leg,
  index,
  onUpdate,
  onRemove,
}: {
  leg: StrategyLeg;
  index: number;
  onUpdate: (i: number, l: StrategyLeg) => void;
  onRemove: (i: number) => void;
}) {
  const update = (partial: Partial<StrategyLeg>) => onUpdate(index, { ...leg, ...partial });

  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.1 }}
      className="flex items-center gap-1.5 px-3 py-1.5 border-b border-grid text-[11px]"
    >
      <select
        value={leg.side}
        onChange={(e) => update({ side: e.target.value as "buy" | "sell" })}
        className={cn(
          "bg-void border border-grid px-1.5 py-0.5 font-data text-[10px] w-14",
          leg.side === "buy" ? "text-profit" : "text-loss"
        )}
      >
        <option value="buy">LONG</option>
        <option value="sell">SHORT</option>
      </select>

      <input
        type="number"
        value={leg.quantity}
        onChange={(e) => update({ quantity: parseInt(e.target.value) || 1 })}
        className="bg-void border border-grid px-1.5 py-0.5 font-data text-[10px] w-10 text-center"
        min={1}
      />

      <input
        type="number"
        value={leg.strike}
        onChange={(e) => update({ strike: parseFloat(e.target.value) || 0 })}
        className="bg-void border border-grid px-1.5 py-0.5 font-data text-[10px] w-16 text-right"
        step={5}
      />

      <select
        value={leg.option_type}
        onChange={(e) => update({ option_type: e.target.value as "call" | "put" })}
        className="bg-void border border-grid px-1.5 py-0.5 font-data text-[10px] w-14"
      >
        <option value="call">CALL</option>
        <option value="put">PUT</option>
      </select>

      <input
        type="number"
        value={leg.premium}
        onChange={(e) => update({ premium: parseFloat(e.target.value) || 0 })}
        className="bg-void border border-grid px-1.5 py-0.5 font-data text-[10px] w-14 text-right"
        step={0.1}
      />

      <button
        onClick={() => onRemove(index)}
        className="text-zinc-500 hover:text-loss transition-colors ml-auto"
      >
        <Trash2 className="w-3 h-3" />
      </button>
    </motion.div>
  );
}

function StrategyGreeksSummary() {
  const analysis = useStrategyStore((s) => s.analysis);
  if (!analysis) return null;

  const g = analysis.total_greeks;
  const greeks = [
    { label: "Δ", value: g.delta, color: g.delta >= 0 ? "text-profit" : "text-loss" },
    { label: "Γ", value: g.gamma, color: "text-warning" },
    { label: "Θ", value: g.theta, color: g.theta >= 0 ? "text-profit" : "text-loss" },
    { label: "V", value: g.vega, color: "text-active" },
  ];

  return (
    <div className="border-t border-grid px-3 py-2 bg-surface">
      <div className="flex items-center gap-3">
        {greeks.map(({ label, value, color }) => (
          <div key={label} className="flex items-center gap-1">
            <span className="text-[9px] font-heading text-muted-foreground tracking-wider">{label}</span>
            <span className={cn("font-data text-[11px]", color)}>{value.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <div className="flex gap-3 mt-1">
        <span className="text-[9px] text-muted-foreground">
          MAX P: <span className="text-profit font-data">${analysis.max_profit?.toLocaleString() ?? "∞"}</span>
        </span>
        <span className="text-[9px] text-muted-foreground">
          MAX L: <span className="text-loss font-data">${analysis.max_loss?.toLocaleString() ?? "∞"}</span>
        </span>
        <span className="text-[9px] text-muted-foreground">
          BE: <span className="text-foreground font-data">{analysis.breakeven_points.map((b) => b.toFixed(1)).join(", ") || "—"}</span>
        </span>
      </div>
    </div>
  );
}

function PayoffChart() {
  const { analysis, timeSliderValue, setTimeSlider } = useStrategyStore();

  const chartData = useMemo(() => {
    if (!analysis?.payoff_data?.length) return [];

    // Select curve based on time slider
    const curveIndex = Math.min(
      Math.round(timeSliderValue * (analysis.payoff_data.length - 1)),
      analysis.payoff_data.length - 1
    );
    const curve = analysis.payoff_data[curveIndex];
    if (!curve) return [];

    return curve.points.map((p) => ({
      price: p.price,
      pnl: p.pnl,
      positive: p.pnl >= 0 ? p.pnl : 0,
      negative: p.pnl < 0 ? p.pnl : 0,
    }));
  }, [analysis, timeSliderValue]);

  const currentCurve = analysis?.payoff_data?.[
    Math.min(
      Math.round(timeSliderValue * ((analysis?.payoff_data?.length ?? 1) - 1)),
      (analysis?.payoff_data?.length ?? 1) - 1
    )
  ];

  if (!analysis) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground">
        <div className="text-center">
          <TrendingUp className="w-8 h-8 mx-auto mb-2 opacity-30" />
          <p className="font-data text-xs">Build a strategy to see payoff</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Chart header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-grid bg-surface shrink-0">
        <h3 className="font-heading text-[10px] text-muted-foreground tracking-widest">
          PAYOFF DIAGRAM
        </h3>
        <Badge variant="outline" className="font-data text-[10px] border-grid">
          {currentCurve?.label ?? "EXPIRY"}
        </Badge>
      </div>

      {/* Chart */}
      <div className="flex-1 px-2 py-2">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <CartesianGrid stroke="#27272a" strokeDasharray="2 2" />
            <XAxis
              dataKey="price"
              tick={{ fontSize: 10, fontFamily: "JetBrains Mono", fill: "#a1a1aa" }}
              tickFormatter={(v: number) => v.toFixed(0)}
              stroke="#27272a"
            />
            <YAxis
              tick={{ fontSize: 10, fontFamily: "JetBrains Mono", fill: "#a1a1aa" }}
              tickFormatter={(v: number) => `$${v.toFixed(0)}`}
              stroke="#27272a"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#18181b",
                border: "1px solid #27272a",
                borderRadius: 0,
                fontSize: 11,
                fontFamily: "JetBrains Mono",
              }}
              formatter={(value) => [`$${Number(value).toFixed(2)}`, "P&L"]}
              labelFormatter={(label) => `Price: $${Number(label).toFixed(2)}`}
            />
            <ReferenceLine y={0} stroke="#52525b" strokeWidth={1} />
            {analysis.breakeven_points.map((be, i) => (
              <ReferenceLine
                key={i}
                x={be}
                stroke="#22d3ee"
                strokeDasharray="4 4"
                strokeWidth={1}
              />
            ))}
            <defs>
              <linearGradient id="profitGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#10b981" stopOpacity={0.05} />
              </linearGradient>
              <linearGradient id="lossGrad" x1="0" y1="1" x2="0" y2="0">
                <stop offset="0%" stopColor="#f43f5e" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#f43f5e" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            <Area
              type="monotone"
              dataKey="positive"
              stroke="#10b981"
              fill="url(#profitGrad)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="negative"
              stroke="#f43f5e"
              fill="url(#lossGrad)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Time slider */}
      <div className="px-4 py-2 border-t border-grid bg-surface shrink-0">
        <div className="flex items-center gap-3">
          <span className="text-[9px] font-heading text-muted-foreground tracking-wider w-6">NOW</span>
          <Slider
            value={[timeSliderValue * 100]}
            onValueChange={(v) => setTimeSlider(v[0] / 100)}
            max={100}
            step={1}
            className="flex-1"
          />
          <span className="text-[9px] font-heading text-muted-foreground tracking-wider w-8">EXP</span>
        </div>
      </div>
    </div>
  );
}

export function PayoffArchitect() {
  const symbol = useMarketStore((s) => s.selectedSymbol);
  const { legs } = useStrategyStore();
  const { mutate: analyze, isPending } = useStrategyAnalysis();
  const setAnalysis = useStrategyStore((s) => s.setAnalysis);

  // Auto-analyze when legs change
  useEffect(() => {
    if (legs.length === 0) {
      setAnalysis(null);
      return;
    }

    const timer = setTimeout(() => {
      analyze(
        { symbol, legs },
        { onSuccess: (data) => setAnalysis(data) }
      );
    }, 300);

    return () => clearTimeout(timer);
  }, [legs, symbol, analyze, setAnalysis]);

  return (
    <div className="flex h-full border-l border-grid">
      {/* Left pane: Leg builder (30%) */}
      <div className="w-[280px] min-w-[240px] border-r border-grid flex flex-col">
        <LegBuilder />
      </div>

      {/* Right pane: Chart (70%) */}
      <div className="flex-1 flex flex-col">
        <PayoffChart />
      </div>
    </div>
  );
}
