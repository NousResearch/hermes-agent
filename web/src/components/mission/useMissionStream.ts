import { useState, useEffect, useRef, useCallback } from "react";
import type { HealthChip, MissionSnapshot, SpendHistory, QueueHistory } from "./types";
import { api } from "@/lib/api";
import { HERMES_BASE_PATH } from "@/lib/api";

const SSE_URL = `${HERMES_BASE_PATH}/api/dashboard/stream`;
const FALLBACK_POLL_MS = 30_000;
const BACKOFF_START_MS = 1_000;
const BACKOFF_MAX_MS = 30_000;
const FALLBACK_FAILURE_THRESHOLD = 3;

/** Returns live HealthChip array, updated via SSE with poll fallback. */
export function useMissionStream(): HealthChip[] {
  const [chips, setChips] = useState<HealthChip[]>([]);
  const esRef = useRef<EventSource | null>(null);
  const failureCount = useRef(0);
  const backoffMs = useRef(BACKOFF_START_MS);
  const fallbackTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const getToken = () =>
    typeof window !== "undefined" ? window.__HERMES_SESSION_TOKEN__ ?? "" : "";

  const pollFallback = useCallback(async () => {
    try {
      const runtimes = [
        "codex", "claude-code", "ruflo", "hermes", "kanban", "cron",
      ];
      const results = await Promise.all(
        runtimes.map((name) => api.getRuntimeHealth(name).catch(() => null))
      );
      const valid = results.filter(Boolean) as HealthChip[];
      if (valid.length > 0) {
        setChips(valid);
      }
    } catch {
      // Silently ignore poll errors
    }
  }, []);

  const startFallback = useCallback(() => {
    if (fallbackTimerRef.current) return;
    pollFallback();
    fallbackTimerRef.current = setInterval(pollFallback, FALLBACK_POLL_MS);
  }, [pollFallback]);

  const stopFallback = useCallback(() => {
    if (fallbackTimerRef.current) {
      clearInterval(fallbackTimerRef.current);
      fallbackTimerRef.current = null;
    }
  }, []);

  const connect = useCallback(() => {
    if (esRef.current) {
      esRef.current.close();
      esRef.current = null;
    }

    const token = getToken();
    const url = token
      ? `${SSE_URL}?token=${encodeURIComponent(token)}`
      : SSE_URL;

    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (evt) => {
      try {
        const chip = JSON.parse(evt.data) as HealthChip & { eventType?: string };
        if (chip.eventType === "health") {
          setChips((prev) => {
            const idx = prev.findIndex((c) => c.name === chip.name);
            if (idx === -1) return [...prev, chip];
            const next = [...prev];
            next[idx] = chip;
            return next;
          });
        }
        // Reset failure counters on successful message
        failureCount.current = 0;
        backoffMs.current = BACKOFF_START_MS;
        stopFallback();
      } catch {
        // Ignore malformed events
      }
    };

    es.onerror = () => {
      es.close();
      esRef.current = null;
      failureCount.current += 1;

      if (failureCount.current >= FALLBACK_FAILURE_THRESHOLD) {
        startFallback();
      }

      // Exponential backoff reconnect
      const delay = Math.min(backoffMs.current, BACKOFF_MAX_MS);
      backoffMs.current = Math.min(backoffMs.current * 2, BACKOFF_MAX_MS);
      reconnectTimerRef.current = setTimeout(connect, delay);
    };
  }, [startFallback, stopFallback]);

  useEffect(() => {
    connect();
    return () => {
      esRef.current?.close();
      esRef.current = null;
      stopFallback();
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
    };
  }, [connect, stopFallback]);

  return chips;
}

/** Polls /api/dashboard/mission on a fixed interval. */
export function useMissionSnapshot(intervalMs = 30_000): MissionSnapshot | null {
  const [snapshot, setSnapshot] = useState<MissionSnapshot | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetch = async () => {
      try {
        const data = await api.getMissionSnapshot();
        if (!cancelled) setSnapshot(data);
      } catch {
        // Keep last snapshot on error
      }
    };

    fetch();
    const timer = setInterval(fetch, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [intervalMs]);

  return snapshot;
}

/** Polls /api/dashboard/spend for sparkline data. */
export function useSpendHistory(range: "1d" | "7d" | "30d" = "7d"): SpendHistory | null {
  const [history, setHistory] = useState<SpendHistory | null>(null);

  useEffect(() => {
    let cancelled = false;

    const fetch = async () => {
      try {
        const data = await api.getSpendHistory(range);
        if (!cancelled) setHistory(data);
      } catch {
        // Keep last data on error
      }
    };

    fetch();
    const timer = setInterval(fetch, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [range]);

  return history;
}

/** Polls /api/dashboard/queue for the kanban queue-depth sparkline. */
export function useQueueHistory(range: "1d" | "7d" | "30d" = "7d"): QueueHistory | null {
  const [history, setHistory] = useState<QueueHistory | null>(null);

  useEffect(() => {
    let cancelled = false;
    const fetch = async () => {
      try {
        const data = await api.getQueueHistory(range);
        if (!cancelled) setHistory(data);
      } catch {
        // Keep last data on error
      }
    };
    fetch();
    const timer = setInterval(fetch, 30_000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [range]);

  return history;
}
