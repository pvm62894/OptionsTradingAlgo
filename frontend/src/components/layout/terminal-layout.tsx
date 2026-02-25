"use client";

import { Panel, Group, Separator } from "react-resizable-panels";
import { MarketPulseHeader } from "@/components/market/market-pulse-header";
import { OptionChainTable } from "@/components/options/option-chain";
import { VolSurface3D } from "@/components/volatility/vol-surface-3d";
import { VolatilitySurfaceHeatmap } from "@/components/volatility/vol-surface";
import { AlgorithmicSignals } from "@/components/signals/algorithmic-signals";
import { useMarketStream } from "@/hooks/use-market-stream";
import { useMarketStore } from "@/stores/market-store";

export function TerminalLayout() {
  const selectedSymbol = useMarketStore((s) => s.selectedSymbol);

  // Connect to WebSocket stream
  useMarketStream([selectedSymbol, "SPY", "QQQ", "NVDA"]);

  return (
    <div className="h-screen w-screen flex flex-col bg-void overflow-hidden">
      {/* Market Pulse Header */}
      <MarketPulseHeader />

      {/* Main Content Area */}
      <Group orientation="vertical" className="flex-1">
        {/* Top row: Option Chain + 3D Vol Surface */}
        <Panel defaultSize="60%" minSize="30%">
          <Group orientation="horizontal" className="h-full">
            {/* Option Chain (40%) */}
            <Panel defaultSize="40%" minSize="25%">
              <div className="h-full border-r border-grid">
                <OptionChainTable />
              </div>
            </Panel>

            <Separator className="w-px bg-grid hover:bg-active transition-colors" />

            {/* 3D Volatility Surface (60%) */}
            <Panel defaultSize="60%" minSize="30%">
              <VolSurface3D />
            </Panel>
          </Group>
        </Panel>

        <Separator className="h-px bg-grid hover:bg-active transition-colors" />

        {/* Bottom row: IV Surface Heatmap + Algorithmic Signals */}
        <Panel defaultSize="40%" minSize="20%">
          <Group orientation="horizontal" className="h-full">
            {/* Volatility Surface Heatmap (50%) */}
            <Panel defaultSize="50%" minSize="25%">
              <div className="h-full border-r border-grid">
                <VolatilitySurfaceHeatmap />
              </div>
            </Panel>

            <Separator className="w-px bg-grid hover:bg-active transition-colors" />

            {/* Algorithmic Signals (50%) */}
            <Panel defaultSize="50%" minSize="25%">
              <AlgorithmicSignals />
            </Panel>
          </Group>
        </Panel>
      </Group>
    </div>
  );
}
