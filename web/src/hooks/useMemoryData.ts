import { useCallback, useMemo, useState } from "react";
import { useRetryFetch } from "@/hooks/useRetryFetch";
import type { RetryFetchStatus } from "@/hooks/useRetryFetch";
import { isAuthUnauthorizedError } from "@/lib/api";
import { fetchMemoryData, updateMemoryData } from "@/lib/clients/memory";
import type { MemoryEntry, MemorySnapshot, MemoryTarget } from "@/lib/clients/memory";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export type { MemoryEntry, MemorySnapshot, MemoryTarget };

/**
 * Thin client/adapter interface.
 *
 * Pass a custom implementation to swap transport (REST → WebSocket/polling/SSE)
 * without touching this hook's internals.
 */
export type MemoryDataClient = {
  fetchMemoryState: (target: MemoryTarget) => Promise<MemorySnapshot>;
  updateMemoryState: (target: MemoryTarget, content: string) => Promise<{ success: boolean; char_count: number; char_limit: number }>;
};

/**
 * State shape exposed to consumers.
 *
 * - `data`    — the last successfully fetched snapshot, or null before the
 *               first successful fetch.
 * - `loading` — true while a fetch is in-flight.
 * - `error`   — human-readable error string if the last fetch failed, else null.
 * - `refetch` — trigger a fresh fetch on demand (e.g. after an edit).
 *               Pass `keepExisting: true` to avoid flicker during background
 *               refreshes.
 * - `save`    — send an updated content string to the backend, then refetch
 *               so state stays consistent.
 */
export type MemoryDataState = {
  data: MemorySnapshot | null;
  loading: boolean;
  error: string | null;
  connectionStatus: RetryFetchStatus;
  nextRetryMs: number | null;
  unauthorized: boolean;
  refetch: (options?: { keepExisting?: boolean }) => Promise<void>;
  /** Backwards-compatible alias for older callers/tests. */
  refresh: (options?: { keepExisting?: boolean }) => Promise<void>;
  save: (content: string) => Promise<void>;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

function makeDefaultClient(): MemoryDataClient {
  return {
    fetchMemoryState: (target) => fetchMemoryData(target),
    updateMemoryState: (target, content) => updateMemoryData(target, content),
  };
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

/**
 * useMemoryData — fetches Hermes agent memory for a given target store with
 * exponential-backoff retry so the UI recovers automatically after a backend
 * kill/restore or transient network drop.
 *
 * - Polls every 60s on success.
 * - On error: retries with 60s → 120s backoff (capped at 2 minutes).
 * - Shows `loading=true` only on the initial fetch; background polls and
 *   retries do not flicker the spinner.
 *
 * Accepts an optional `client` adapter so callers can inject a mocked
 * transport in tests, or swap to WebSocket/SSE/polling later without
 * changing any call sites.
 *
 * @example
 *   const { data, loading, error, refetch, save } = useMemoryData("memory");
 *
 * @example (custom adapter for tests / WS)
 *   const client = { fetchMemoryState: myWsAdapter, updateMemoryState: myWsUpdater };
 *   const state = useMemoryData("memory", client);
 */
export function useMemoryData(
  target: MemoryTarget = "memory",
  client?: MemoryDataClient,
): MemoryDataState {
  const defaultClient = useMemo(() => makeDefaultClient(), []);
  const resolvedClient: MemoryDataClient = client ?? defaultClient;

  const [data, setData] = useState<MemorySnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<RetryFetchStatus>("idle");
  const [nextRetryMs, setNextRetryMs] = useState<number | null>(null);
  const [unauthorized, setUnauthorized] = useState(false);

  const fetchFn = useCallback(
    () => resolvedClient.fetchMemoryState(target),
    [resolvedClient, target],
  );

  const handleSuccess = useCallback((snapshot: MemorySnapshot) => {
    setData(snapshot);
    setError(null);
    setUnauthorized(false);
    setConnectionStatus("connected");
  }, []);

  const handleError = useCallback((message: string, err: unknown) => {
    setError(message);
    if (isAuthUnauthorizedError(err)) {
      setUnauthorized(true);
      setData(null);
      return;
    }
    setUnauthorized(false);
    // Keep last known data visible so the UI degrades gracefully.
  }, []);

  // Auto-retry loop: polls every 60s on success, backs off on error (60→120s).
  useRetryFetch({
    fetchFn,
    onSuccess: handleSuccess,
    onError: handleError,
    setLoading,
    baseIntervalMs: 60_000,
    retryIntervalMs: 2_000,
    maxIntervalMs: 10_000,
    onStatusChange: setConnectionStatus,
    onNextRetryMsChange: setNextRetryMs,
  });

  // Imperative refetch (e.g. after a save, or user-triggered refresh).
  const refetch = useCallback(
    async (options: { keepExisting?: boolean } = {}) => {
      if (!options.keepExisting) {
        setLoading(true);
      }
      setError(null);
      try {
        const snapshot = await resolvedClient.fetchMemoryState(target);
        setData(snapshot);
        setUnauthorized(false);
        setConnectionStatus("connected");
      } catch (err) {
        setError(toErrorMessage(err));
        setConnectionStatus(typeof navigator !== "undefined" && navigator.onLine === false ? "offline" : "reconnecting");
        if (isAuthUnauthorizedError(err)) {
          setUnauthorized(true);
          setData(null);
        } else {
          setUnauthorized(false);
          if (!options.keepExisting) {
            setData(null);
          }
        }
      } finally {
        setLoading(false);
      }
    },
    [resolvedClient, target],
  );

  const save = useCallback(
    async (content: string) => {
      setLoading(true);
      setError(null);
      try {
        await resolvedClient.updateMemoryState(target, content);
        setUnauthorized(false);
        // Refetch to keep local state consistent with the server's parsed
        // representation (entry list, char_count, etc.).
        await refetch({ keepExisting: true });
      } catch (err) {
        setError(toErrorMessage(err));
        if (isAuthUnauthorizedError(err)) {
          setUnauthorized(true);
          setData(null);
        }
        setLoading(false);
      }
    },
    [resolvedClient, target, refetch],
  );

  return { data, loading, error, connectionStatus, nextRetryMs, unauthorized, refetch, refresh: refetch, save };
}
