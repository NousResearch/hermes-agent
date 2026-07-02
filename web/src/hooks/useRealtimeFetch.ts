import { useCallback, useEffect, useRef } from "react";
import { HERMES_BASE_PATH, buildWsAuthParam } from "@/lib/api";
import type { RetryFetchStatus } from "@/hooks/useRetryFetch";

export interface UseRealtimeFetchOptions<T> {
  fetchFn: () => Promise<T>;
  onSuccess: (data: T) => void;
  onError: (message: string, err: unknown) => void;
  setLoading: (loading: boolean) => void;
  onStatusChange?: (status: RetryFetchStatus) => void;
  onNextRetryMsChange?: (delayMs: number | null) => void;
  /** Preferred WebSocket endpoint. If omitted/null, the hook uses polling only. */
  websocketPath?: string | (() => string | null | undefined);
  /** Poll cadence when no WebSocket endpoint is available, or while reconnecting. */
  pollingIntervalMs?: number;
  /** First reconnect delay after a WebSocket drop. */
  retryIntervalMs?: number;
  /** Maximum reconnect delay. */
  maxRetryMs?: number;
  enabled?: boolean;
}

function toErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function browserIsOffline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

function resolvePath(path: string | (() => string | null | undefined) | undefined): string | null {
  if (!path) return null;
  const resolved = typeof path === "function" ? path() : path;
  return resolved || null;
}

function wsUrl(path: string): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  // Contract: callers pass root-relative API paths without HERMES_BASE_PATH.
  // REST uses fetchJSON to prepend the base path; WebSocket URLs are built here.
  const basePath = path.startsWith("/") ? path : `/${path}`;
  return `${proto}//${window.location.host}${HERMES_BASE_PATH}${basePath}`;
}

async function withWsAuth(path: string): Promise<string> {
  const url = new URL(wsUrl(path));
  const [key, value] = await buildWsAuthParam();
  if (value) {
    url.searchParams.set(key, value);
  }
  return url.toString();
}

/**
 * Real-time dashboard data driver.
 *
 * Prefer a WebSocket subscription when `websocketPath` is supplied. Socket
 * messages are invalidation signals that trigger an immediate REST refresh, so
 * consumers keep using the same snapshot normalizers. If no socket exists, or
 * while a socket is reconnecting, the hook falls back to interval polling.
 * All timers/sockets/listeners are cancelled on unmount or option changes.
 */
export function useRealtimeFetch<T>({
  fetchFn,
  onSuccess,
  onError,
  setLoading,
  onStatusChange,
  onNextRetryMsChange,
  websocketPath,
  pollingIntervalMs = 30_000,
  retryIntervalMs = 2_000,
  maxRetryMs = 10_000,
  enabled = true,
}: UseRealtimeFetchOptions<T>): void {
  const fetchSeqRef = useRef(0);
  const websocketRef = useRef<WebSocket | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectDelayRef = useRef(retryIntervalMs);
  const mountedRef = useRef(false);
  const hasLoadedRef = useRef(false);

  const clearPoll = useCallback(() => {
    if (pollTimerRef.current !== null) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const clearReconnect = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    onNextRetryMsChange?.(null);
  }, [onNextRetryMsChange]);

  const closeSocket = useCallback(() => {
    const socket = websocketRef.current;
    websocketRef.current = null;
    if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN)) {
      socket.close();
    }
  }, []);

  const runFetch = useCallback(
    async (showLoading: boolean) => {
      if (!mountedRef.current || !enabled) return;
      if (browserIsOffline()) {
        onStatusChange?.("offline");
        setLoading(false);
        return;
      }
      const seq = ++fetchSeqRef.current;
      if (showLoading) {
        onStatusChange?.("loading");
        setLoading(true);
      }
      try {
        const data = await fetchFn();
        if (!mountedRef.current || seq !== fetchSeqRef.current) return;
        hasLoadedRef.current = true;
        onSuccess(data);
        onStatusChange?.("connected");
        setLoading(false);
      } catch (err) {
        if (!mountedRef.current || seq !== fetchSeqRef.current) return;
        onError(toErrorMessage(err), err);
        onStatusChange?.(browserIsOffline() ? "offline" : "reconnecting");
        setLoading(false);
      }
    },
    [enabled, fetchFn, onError, onStatusChange, onSuccess, setLoading],
  );

  useEffect(() => {
    if (!enabled) return;
    mountedRef.current = true;
    hasLoadedRef.current = false;
    reconnectDelayRef.current = retryIntervalMs;

    let cancelled = false;

    const startPolling = () => {
      if (cancelled || pollTimerRef.current !== null) return;
      pollTimerRef.current = setInterval(() => {
        void runFetch(false);
      }, pollingIntervalMs);
    };

    const stopPolling = () => clearPoll();

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimerRef.current !== null) return;
      const delay = Math.min(reconnectDelayRef.current, maxRetryMs);
      reconnectDelayRef.current = Math.min(delay * 2, maxRetryMs);
      onNextRetryMsChange?.(delay);
      reconnectTimerRef.current = setTimeout(() => {
        reconnectTimerRef.current = null;
        onNextRetryMsChange?.(null);
        void connectSocket();
      }, delay);
    };

    const connectSocket = async () => {
      const path = resolvePath(websocketPath);
      if (!path || typeof WebSocket === "undefined") {
        onStatusChange?.("connected");
        startPolling();
        return;
      }
      if (browserIsOffline()) {
        onStatusChange?.("offline");
        startPolling();
        scheduleReconnect();
        return;
      }
      try {
        closeSocket();
        onStatusChange?.(hasLoadedRef.current ? "reconnecting" : "loading");
        const socket = new WebSocket(await withWsAuth(path));
        websocketRef.current = socket;

        socket.onopen = () => {
          if (cancelled || websocketRef.current !== socket) return;
          reconnectDelayRef.current = retryIntervalMs;
          clearReconnect();
          stopPolling();
          onStatusChange?.("connected");
        };

        socket.onmessage = () => {
          if (cancelled || websocketRef.current !== socket) return;
          void runFetch(false);
        };

        socket.onerror = () => {
          if (cancelled || websocketRef.current !== socket) return;
          onStatusChange?.("reconnecting");
        };

        socket.onclose = () => {
          if (cancelled || websocketRef.current !== socket) return;
          websocketRef.current = null;
          onStatusChange?.(browserIsOffline() ? "offline" : "reconnecting");
          startPolling();
          scheduleReconnect();
        };
      } catch (err) {
        onError(toErrorMessage(err), err);
        onStatusChange?.("reconnecting");
        startPolling();
        scheduleReconnect();
      }
    };

    const handleOnline = () => {
      onStatusChange?.("reconnecting");
      clearReconnect();
      void runFetch(false);
      void connectSocket();
    };

    const handleOffline = () => {
      onStatusChange?.("offline");
      closeSocket();
      startPolling();
    };

    void runFetch(true);
    void connectSocket();

    if (typeof window !== "undefined") {
      window.addEventListener("online", handleOnline);
      window.addEventListener("offline", handleOffline);
    }

    return () => {
      cancelled = true;
      mountedRef.current = false;
      if (typeof window !== "undefined") {
        window.removeEventListener("online", handleOnline);
        window.removeEventListener("offline", handleOffline);
      }
      clearPoll();
      clearReconnect();
      closeSocket();
    };
  }, [
    clearPoll,
    clearReconnect,
    closeSocket,
    enabled,
    maxRetryMs,
    onError,
    onNextRetryMsChange,
    onStatusChange,
    pollingIntervalMs,
    retryIntervalMs,
    runFetch,
    websocketPath,
  ]);
}
