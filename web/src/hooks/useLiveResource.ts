import { useCallback, useEffect, useRef, useState } from "react";
import type { Dispatch, SetStateAction } from "react";

export interface UseLiveResourceOptions<T> {
  load: (signal?: AbortSignal) => Promise<T>;
  intervalMs?: number;
  enabled?: boolean;
  initialData?: T;
  refreshOnWindowFocus?: boolean;
  refreshWhenVisible?: boolean;
}

export interface UseLiveResourceResult<T> {
  data: T | undefined;
  setData: Dispatch<SetStateAction<T | undefined>>;
  loading: boolean;
  isRefreshing: boolean;
  error: Error | null;
  lastUpdated: Date | null;
  refresh: () => Promise<void>;
}

function isDocumentHidden(): boolean {
  return typeof document !== "undefined" && document.hidden;
}

function toError(value: unknown): Error {
  return value instanceof Error ? value : new Error(String(value));
}

/**
 * Poll a dashboard resource without wiping the last good state on transient
 * errors. The hook pauses interval refreshes while the tab is hidden and
 * refreshes immediately when the tab becomes visible again.
 */
export function useLiveResource<T>({
  load,
  intervalMs = 0,
  enabled = true,
  initialData,
  refreshOnWindowFocus = true,
  refreshWhenVisible = true,
}: UseLiveResourceOptions<T>): UseLiveResourceResult<T> {
  const [data, setData] = useState<T | undefined>(initialData);
  const [loading, setLoading] = useState(enabled && initialData === undefined);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);
  const requestSeq = useRef(0);
  const mounted = useRef(false);

  const refresh = useCallback(async () => {
    if (!enabled) return;

    const seq = requestSeq.current + 1;
    requestSeq.current = seq;
    const controller = new AbortController();

    setIsRefreshing(true);

    try {
      const next = await load(controller.signal);
      if (requestSeq.current !== seq) return;
      setData(next);
      setError(null);
      setLastUpdated(new Date());
    } catch (err) {
      if (requestSeq.current !== seq) return;
      if (!controller.signal.aborted) {
        setError(toError(err));
      }
    } finally {
      if (requestSeq.current === seq) {
        setLoading(false);
        setIsRefreshing(false);
      }
    }
  }, [enabled, load]);

  useEffect(() => {
    mounted.current = true;
    const id = window.setTimeout(() => {
      if (mounted.current) {
        void refresh();
      }
    }, 0);
    return () => {
      window.clearTimeout(id);
      mounted.current = false;
      requestSeq.current += 1;
    };
  }, [refresh]);

  useEffect(() => {
    if (!enabled || !intervalMs) return;
    const id = window.setInterval(() => {
      if (refreshWhenVisible && isDocumentHidden()) return;
      void refresh();
    }, intervalMs);
    return () => window.clearInterval(id);
  }, [enabled, intervalMs, refresh, refreshWhenVisible]);

  useEffect(() => {
    if (!enabled || !refreshWhenVisible || typeof document === "undefined") return;
    const onVisibilityChange = () => {
      if (!document.hidden && mounted.current) {
        void refresh();
      }
    };
    document.addEventListener("visibilitychange", onVisibilityChange);
    return () => document.removeEventListener("visibilitychange", onVisibilityChange);
  }, [enabled, refresh, refreshWhenVisible]);

  useEffect(() => {
    if (!enabled || !refreshOnWindowFocus || typeof window === "undefined") return;
    const onFocus = () => {
      if (!isDocumentHidden() && mounted.current) {
        void refresh();
      }
    };
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [enabled, refresh, refreshOnWindowFocus]);

  return { data, setData, loading, isRefreshing, error, lastUpdated, refresh };
}
