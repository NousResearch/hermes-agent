import { useCallback, useRef, useState } from "react";
import { isAuthUnauthorizedError } from "@/lib/api";
import type { RetryFetchStatus } from "@/hooks/useRetryFetch";
import { useRealtimeFetch } from "@/hooks/useRealtimeFetch";
import {
  buildKanbanEventsPath,
  fetchKanbanState,
} from "@/lib/clients/kanban";
import type {
  KanbanColumn,
  KanbanStateSnapshot,
  KanbanTask,
} from "@/lib/clients/kanban";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type { KanbanColumn, KanbanStateSnapshot, KanbanTask };

/** State shape exposed to consumers. */
export type KanbanStateResult = {
  data: KanbanStateSnapshot | null;
  loading: boolean;
  error: string | null;
  connectionStatus: RetryFetchStatus;
  nextRetryMs: number | null;
  unauthorized: boolean;
  refresh: () => Promise<void>;
};

/** Options accepted by the hook. */
export type KanbanStateOptions = {
  /**
   * Poll the board every N milliseconds on success.
   * Also serves as the base retry delay on error (doubles up to maxRetryMs).
   * @default 30_000
   */
  pollingIntervalMs?: number;
  /**
   * Maximum retry backoff delay (ms).
   * @default 120_000
   */
  maxRetryMs?: number;
  /** Board slug. Omit to use the server's active board. */
  board?: string;
  /** Disable the Kanban WebSocket stream, primarily for tests. */
  disableWebSocket?: boolean;
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

function toErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * useKanbanState — fetches the active Kanban board with exponential-backoff
 * retry so the sidebar recovers automatically after a backend kill/restore
 * or transient network drop.
 *
 * - Polls every `pollingIntervalMs` ms on success (default 30s).
 * - On error: retries with pollingIntervalMs → 2× → 4× backoff (capped at
 *   `maxRetryMs`, default 2 minutes).
 * - Suppresses errors silently so the sidebar doesn't break when the kanban
 *   plugin isn't installed. Callers can inspect `error` for error badges.
 *
 * @example
 *   const { data, loading } = useKanbanState();
 *   const running = data?.columns.find(c => c.status === "running")?.tasks.length ?? 0;
 */
export function useKanbanState(
  options: KanbanStateOptions = {},
): KanbanStateResult {
  const {
    pollingIntervalMs = 30_000,
    maxRetryMs = 120_000,
    board,
    disableWebSocket = false,
  } = options;

  const [data, setData] = useState<KanbanStateSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<RetryFetchStatus>("idle");
  const [nextRetryMs, setNextRetryMs] = useState<number | null>(null);
  const [unauthorized, setUnauthorized] = useState(false);
  const latestEventIdRef = useRef(0);

  const fetchFn = useCallback(
    () => fetchKanbanState({ board }),
    [board],
  );

  const websocketPath = useCallback(
    () => (disableWebSocket ? null : buildKanbanEventsPath({ board, since: latestEventIdRef.current })),
    [board, disableWebSocket],
  );

  const handleSuccess = useCallback((snapshot: KanbanStateSnapshot) => {
    latestEventIdRef.current = snapshot.latest_event_id;
    setData(snapshot);
    setError(null);
    setUnauthorized(false);
  }, []);

  const handleError = useCallback((message: string, err: unknown) => {
    setError(message);
    if (isAuthUnauthorizedError(err)) {
      setUnauthorized(true);
      setData(null);
      return;
    }
    setUnauthorized(false);
    // Keep last known data visible so the sidebar degrades gracefully.
  }, []);

  // Prefer the Kanban plugin WebSocket event stream. Socket messages are
  // invalidation signals that trigger an immediate REST snapshot refresh; if
  // the socket is unavailable/reconnecting, fall back to interval polling.
  useRealtimeFetch({
    fetchFn,
    onSuccess: handleSuccess,
    onError: handleError,
    setLoading,
    websocketPath,
    pollingIntervalMs,
    retryIntervalMs: 2_000,
    maxRetryMs: Math.min(maxRetryMs, 10_000),
    onStatusChange: setConnectionStatus,
    onNextRetryMsChange: setNextRetryMs,
  });

  // Imperative refresh for consumers that want to trigger an immediate poll
  // (e.g. after a mutation, or on a user-initiated "refresh" button click).
  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const snapshot = await fetchKanbanState({ board });
      latestEventIdRef.current = snapshot.latest_event_id;
      setData(snapshot);
      setUnauthorized(false);
      setConnectionStatus("connected");
    } catch (err) {
      setError(toErrorMessage(err));
      if (isAuthUnauthorizedError(err)) {
        setUnauthorized(true);
        setData(null);
      } else {
        setUnauthorized(false);
      }
    } finally {
      setLoading(false);
    }
  }, [board]);

  return { data, loading, error, connectionStatus, nextRetryMs, unauthorized, refresh };
}
