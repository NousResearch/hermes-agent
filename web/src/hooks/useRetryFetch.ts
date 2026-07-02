/**
 * useRetryFetch — shared polling-with-backoff utility for dashboard data hooks.
 *
 * Drives a repeated fetch cycle:
 *   - Fetches immediately on mount (and whenever `fetchFn` identity changes).
 *   - On success: resets delay to `baseIntervalMs`, schedules the next fetch.
 *   - On error:   backs off exponentially (delay × 2) up to `maxIntervalMs`,
 *                 then retries.
 *   - Set `enabled: false` to pause all fetching (e.g. while offline).
 *
 * The hook is intentionally transport-agnostic — pass any async function that
 * resolves to T on success and throws on error.
 *
 * @example
 *   useRetryFetch({
 *     fetchFn: () => api.getProfiles(),
 *     onSuccess: (data) => setProfiles(data.profiles),
 *     onError: (msg) => setError(msg),
 *     setLoading,
 *     baseIntervalMs: 30_000,
 *     maxIntervalMs:  120_000,
 *   });
 */

import { useCallback, useEffect, useRef } from "react";

export type RetryFetchStatus = "idle" | "loading" | "connected" | "reconnecting" | "offline";

export interface UseRetryFetchOptions<T> {
  /** The async function to call on every fetch cycle. */
  fetchFn: () => Promise<T>;
  /** Called with the resolved value on a successful fetch. */
  onSuccess: (data: T) => void;
  /** Called with an error message and the original thrown value when the fetch throws. */
  onError: (message: string, error: unknown) => void;
  /** Called with true at the start of the first fetch and false after any
   *  resolution (success or error). On retries, loading is NOT re-set to true
   *  so the UI doesn't flicker during background recovery. */
  setLoading: (loading: boolean) => void;
  /**
   * Base poll interval on success (ms). The hook also uses this as the
   * starting retry delay after a first failure.
   * @default 30_000
   */
  baseIntervalMs?: number;
  /**
   * Maximum backoff delay (ms).
   * @default 120_000
   */
  maxIntervalMs?: number;
  /**
   * First retry delay after a failed request. Defaults to 2 seconds so
   * backend kill/restore and browser offline/online tests recover within the
   * dashboard's 12–15 second verification window while successful polling can
   * remain much quieter.
   * @default 2_000
   */
  retryIntervalMs?: number;
  /** Called whenever the retry loop enters loading/connected/reconnecting/offline. */
  onStatusChange?: (status: RetryFetchStatus) => void;
  /** Called with the next retry delay in milliseconds, or null when idle/successful. */
  onNextRetryMsChange?: (delayMs: number | null) => void;
  /**
   * When false, no fetches are issued. Useful for pausing while the user
   * has taken the UI offline intentionally.
   * @default true
   */
  enabled?: boolean;
}

function toErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

export function useRetryFetch<T>({
  fetchFn,
  onSuccess,
  onError,
  setLoading,
  baseIntervalMs = 30_000,
  maxIntervalMs = 120_000,
  retryIntervalMs = 2_000,
  onStatusChange,
  onNextRetryMsChange,
  enabled = true,
}: UseRetryFetchOptions<T>): void {
  // Generation counter — incremented whenever the effect re-runs so stale
  // callbacks from a previous interval never write state.
  const genRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const delayRef = useRef(baseIntervalMs);
  const isFirstFetch = useRef(true);
  const inFailureRef = useRef(false);
  // Monotonic request id for the current effect generation. Online events,
  // timers, and manual refreshes can overlap; only the newest background
  // request is allowed to publish state.
  const requestSeqRef = useRef(0);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!enabled) return;

    const gen = ++genRef.current;
    isFirstFetch.current = true;
    delayRef.current = retryIntervalMs;

    let cancelled = false;

    function browserIsOffline(): boolean {
      return typeof navigator !== "undefined" && navigator.onLine === false;
    }

    function scheduleNext(delayMs: number) {
      if (cancelled || genRef.current !== gen) return;
      onNextRetryMsChange?.(delayMs);
      clearTimer();
      timerRef.current = setTimeout(() => {
        timerRef.current = null;
        onNextRetryMsChange?.(null);
        void runCycle();
      }, delayMs);
    }

    async function runCycle() {
      if (cancelled || genRef.current !== gen) return;

      if (browserIsOffline()) {
        onStatusChange?.("offline");
        setLoading(false);
        isFirstFetch.current = false;
        scheduleNext(retryIntervalMs);
        return;
      }

      // Show loading spinner only on the very first fetch, not on retries/polls
      // so the UI doesn't flicker on background refreshes.
      if (isFirstFetch.current) {
        onStatusChange?.("loading");
        setLoading(true);
      } else if (inFailureRef.current) {
        onStatusChange?.("reconnecting");
      } else {
        onStatusChange?.("connected");
      }

      const requestId = ++requestSeqRef.current;

      try {
        const data = await fetchFn();
        if (cancelled || genRef.current !== gen || requestId !== requestSeqRef.current) return;

        inFailureRef.current = false;
        onSuccess(data);
        onStatusChange?.("connected");
        onNextRetryMsChange?.(null);
        setLoading(false);
        isFirstFetch.current = false;

        // Quiet poll cadence after a successful request.
        delayRef.current = retryIntervalMs;
        scheduleNext(baseIntervalMs);
      } catch (err) {
        if (cancelled || genRef.current !== gen || requestId !== requestSeqRef.current) return;

        inFailureRef.current = true;
        onError(toErrorMessage(err), err);
        onStatusChange?.(browserIsOffline() ? "offline" : "reconnecting");
        setLoading(false);
        isFirstFetch.current = false;

        const retryDelay = Math.min(delayRef.current, maxIntervalMs);
        delayRef.current = Math.min(retryDelay * 2, maxIntervalMs);
        scheduleNext(retryDelay);
      }
    }

    function handleOnline() {
      inFailureRef.current = true;
      delayRef.current = retryIntervalMs;
      clearTimer();
      onNextRetryMsChange?.(null);
      void runCycle();
    }

    function handleOffline() {
      inFailureRef.current = true;
      onStatusChange?.("offline");
      setLoading(false);
      if (!timerRef.current) scheduleNext(retryIntervalMs);
    }

    if (typeof window !== "undefined") {
      window.addEventListener("online", handleOnline);
      window.addEventListener("offline", handleOffline);
    }

    void runCycle();

    return () => {
      cancelled = true;
      if (typeof window !== "undefined") {
        window.removeEventListener("online", handleOnline);
        window.removeEventListener("offline", handleOffline);
      }
      onNextRetryMsChange?.(null);
      clearTimer();
    };
  }, [
    fetchFn,
    onSuccess,
    onError,
    setLoading,
    baseIntervalMs,
    maxIntervalMs,
    retryIntervalMs,
    onStatusChange,
    onNextRetryMsChange,
    enabled,
    clearTimer,
  ]);
}
