"use client";

import { useEffect, useRef, useCallback } from "react";
import { useMarketStore } from "@/stores/market-store";
import type { TickMessage } from "@/lib/types";

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/api/v1/stream/prices";
const RECONNECT_BASE_DELAY = 1000;
const MAX_RECONNECT_DELAY = 30000;
const THROTTLE_MS = 100;

export function useMarketStream(symbols: string[] = ["SPY"]) {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttempt = useRef(0);
  const lastUpdate = useRef<Record<string, number>>({});
  const updateTick = useMarketStore((s) => s.updateTick);
  const setConnected = useMarketStore((s) => s.setConnected);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        reconnectAttempt.current = 0;
        // Subscribe to symbols
        ws.send(JSON.stringify({ subscribe: symbols }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TickMessage;
          if (data.type !== "tick") return;

          // Throttle updates per symbol
          const now = Date.now();
          const last = lastUpdate.current[data.symbol] || 0;
          if (now - last < THROTTLE_MS) return;
          lastUpdate.current[data.symbol] = now;

          updateTick(data);
        } catch {
          // Ignore parse errors
        }
      };

      ws.onclose = () => {
        setConnected(false);
        // Exponential backoff reconnection
        const delay = Math.min(
          RECONNECT_BASE_DELAY * Math.pow(2, reconnectAttempt.current),
          MAX_RECONNECT_DELAY
        );
        reconnectAttempt.current++;
        setTimeout(connect, delay);
      };

      ws.onerror = () => {
        ws.close();
      };
    } catch {
      // Connection failed, will retry
    }
  }, [symbols, updateTick, setConnected]);

  useEffect(() => {
    connect();

    // Heartbeat ping
    const pingInterval = setInterval(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "ping" }));
      }
    }, 15000);

    return () => {
      clearInterval(pingInterval);
      wsRef.current?.close();
    };
  }, [connect]);

  // Re-subscribe when symbols change
  useEffect(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ subscribe: symbols }));
    }
  }, [symbols]);
}
