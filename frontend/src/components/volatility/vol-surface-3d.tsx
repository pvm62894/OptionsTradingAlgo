"use client";

import dynamic from "next/dynamic";
import { useMarketStore } from "@/stores/market-store";
import { useVolSurface3D } from "@/hooks/use-queries";
import { useMemo, useState, memo } from "react";
import { RefreshCw } from "lucide-react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type ViewMode = "iv_surface" | "options_flow";
type FlowMetric = "volume" | "oi";

const IV_COLORSCALE: [number, string][] = [
  [0, "#1e3a5f"],
  [0.25, "#22d3ee"],
  [0.5, "#eab308"],
  [0.75, "#f97316"],
  [1, "#f43f5e"],
];

const FLOW_COLORSCALE: [number, string][] = [
  [0, "#0a0a0a"],
  [0.25, "#1e3a5f"],
  [0.5, "#22d3ee"],
  [0.75, "#10b981"],
  [1, "#eab308"],
];

const SCENE_AXES = {
  gridcolor: "#27272a",
  linecolor: "#3f3f46",
  title: { font: { color: "#a1a1aa" } },
  tickfont: { color: "#71717a", size: 9 },
};

const BASE_LAYOUT = {
  paper_bgcolor: "#0a0a0a",
  plot_bgcolor: "#0a0a0a",
  font: { family: "JetBrains Mono, monospace", color: "#a1a1aa", size: 10 },
  margin: { l: 0, r: 0, t: 30, b: 0 },
  scene: {
    camera: { eye: { x: 1.5, y: 1.5, z: 1.2 } },
    xaxis: { ...SCENE_AXES, title: { text: "Strike", font: { color: "#a1a1aa" } } },
    yaxis: { ...SCENE_AXES, title: { text: "DTE", font: { color: "#a1a1aa" } } },
    zaxis: { ...SCENE_AXES, title: { text: "IV %", font: { color: "#a1a1aa" } } },
    bgcolor: "#0a0a0a",
  } as any,
  autosize: true,
};

const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
};

function VolSurface3DInner({ symbol }: { symbol: string }) {
  const { data, isLoading, isError, refetch } = useVolSurface3D(symbol);
  const [viewMode, setViewMode] = useState<ViewMode>("iv_surface");
  const [flowMetric, setFlowMetric] = useState<FlowMetric>("volume");

  const plotData = useMemo(() => {
    if (!data) return [];

    const { strikes, dtes, ivGrid, volumeGrid, oiGrid } = data;

    if (viewMode === "iv_surface") {
      return [
        {
          type: "surface" as const,
          x: strikes,
          y: dtes,
          z: ivGrid,
          colorscale: IV_COLORSCALE,
          showscale: false,
          opacity: 0.92,
          contours: {
            z: { show: true, usecolormap: true, highlightcolor: "#22d3ee", project: { z: false } },
          },
        },
      ];
    }

    const zData = flowMetric === "volume" ? volumeGrid : oiGrid;
    return [
      {
        type: "surface" as const,
        x: strikes,
        y: dtes,
        z: zData,
        colorscale: FLOW_COLORSCALE,
        showscale: false,
        opacity: 0.88,
      },
    ];
  }, [data, viewMode, flowMetric]);

  const layout = useMemo(() => {
    if (viewMode === "options_flow") {
      const zLabel = flowMetric === "volume" ? "Volume" : "Open Interest";
      return {
        ...BASE_LAYOUT,
        scene: {
          ...(BASE_LAYOUT.scene as any),
          zaxis: { ...SCENE_AXES, title: { text: zLabel, font: { color: "#a1a1aa" } } },
        },
      };
    }
    return BASE_LAYOUT;
  }, [viewMode, flowMetric]);

  if (isLoading) {
    return (
      <div className="flex flex-col h-full">
        <Header viewMode={viewMode} setViewMode={setViewMode} flowMetric={flowMetric} setFlowMetric={setFlowMetric} />
        <div className="flex-1 p-4 space-y-3">
          <div className="h-1/3 bg-zinc-900 animate-pulse rounded" />
          <div className="h-1/3 bg-zinc-900 animate-pulse rounded" />
          <div className="h-1/4 bg-zinc-900 animate-pulse rounded" />
        </div>
      </div>
    );
  }

  if (isError || !data) {
    return (
      <div className="flex flex-col h-full">
        <Header viewMode={viewMode} setViewMode={setViewMode} flowMetric={flowMetric} setFlowMetric={setFlowMetric} />
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <p className="text-zinc-500 font-mono text-xs mb-3">Failed to load surface data</p>
            <button
              onClick={() => refetch()}
              className="flex items-center gap-1.5 mx-auto px-3 py-1.5 text-[10px] font-mono tracking-wider text-cyan-400 border border-zinc-700 hover:border-cyan-400/50 transition-colors"
            >
              <RefreshCw className="w-3 h-3" /> RETRY
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header viewMode={viewMode} setViewMode={setViewMode} flowMetric={flowMetric} setFlowMetric={setFlowMetric} />
      <div className="flex-1 min-h-0">
        <Plot
          data={plotData as any}
          layout={layout as any}
          config={PLOT_CONFIG}
          style={{ width: "100%", height: "100%" }}
        />
      </div>
    </div>
  );
}

function Header({
  viewMode,
  setViewMode,
  flowMetric,
  setFlowMetric,
}: {
  viewMode: ViewMode;
  setViewMode: (v: ViewMode) => void;
  flowMetric: FlowMetric;
  setFlowMetric: (m: FlowMetric) => void;
}) {
  return (
    <div className="flex items-center justify-between px-3 py-1.5 border-b border-zinc-800 bg-zinc-950/50 shrink-0">
      <h3 className="font-mono text-[10px] text-zinc-500 tracking-widest uppercase">
        3D Vol Surface
      </h3>
      <div className="flex items-center gap-1">
        <ToggleBtn active={viewMode === "iv_surface"} onClick={() => setViewMode("iv_surface")}>
          IV Surface
        </ToggleBtn>
        <ToggleBtn active={viewMode === "options_flow"} onClick={() => setViewMode("options_flow")}>
          Options Flow
        </ToggleBtn>
        {viewMode === "options_flow" && (
          <>
            <div className="w-px h-3 bg-zinc-700 mx-1" />
            <ToggleBtn active={flowMetric === "volume"} onClick={() => setFlowMetric("volume")}>
              Vol
            </ToggleBtn>
            <ToggleBtn active={flowMetric === "oi"} onClick={() => setFlowMetric("oi")}>
              OI
            </ToggleBtn>
          </>
        )}
      </div>
    </div>
  );
}

function ToggleBtn({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-0.5 text-[9px] font-mono tracking-wider border transition-colors ${
        active
          ? "border-cyan-400/50 text-cyan-400 bg-cyan-400/5"
          : "border-zinc-700 text-zinc-500 hover:border-zinc-500 hover:text-zinc-400"
      }`}
    >
      {children}
    </button>
  );
}

export const VolSurface3D = memo(function VolSurface3D() {
  const symbol = useMarketStore((s) => s.selectedSymbol);
  return <VolSurface3DInner symbol={symbol} />;
});
