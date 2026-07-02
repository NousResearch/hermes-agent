import { useCallback, useState } from "react";
import { isAuthUnauthorizedError } from "@/lib/api";
import { listAgentProfiles } from "@/lib/clients/agentProfiles";
import type { AgentProfileInfo } from "@/lib/clients/agentProfiles";
import { useRetryFetch } from "@/hooks/useRetryFetch";
import type { RetryFetchStatus } from "@/hooks/useRetryFetch";

/**
 * Snapshot type returned by the data-source adapter.
 * Keeping it in one place makes it easy to extend later (e.g. cursor, etag).
 */
export type AgentProfilesSnapshot = {
  profiles: AgentProfileInfo[];
};

/**
 * Thin client/adapter interface.
 * Pass a custom implementation to swap transport (REST → WebSocket/polling)
 * without touching this hook's internals.
 */
export type AgentProfilesClient = {
  listProfiles: () => Promise<AgentProfilesSnapshot>;
};

/**
 * State shape exposed to consumers.
 * `refreshProfiles` is callable on-demand (e.g. after a CRUD operation).
 * `keepExisting: true` is useful for background refreshes where you don't
 * want the list to flicker back to empty while the request is in-flight.
 */
export type AgentProfilesState = {
  profiles: AgentProfileInfo[];
  loading: boolean;
  error: string | null;
  connectionStatus: RetryFetchStatus;
  nextRetryMs: number | null;
  unauthorized: boolean;
  refreshProfiles: (options?: { keepExisting?: boolean }) => Promise<void>;
};

// Default adapter wired to the live REST endpoint.
const defaultClient: AgentProfilesClient = {
  listProfiles: () => listAgentProfiles(),
};

function toErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  return String(err);
}

/**
 * useAgentProfiles — fetches agent profiles from the live REST API with
 * exponential-backoff retry so the UI recovers automatically after a backend
 * kill/restore or transient network drop.
 *
 * - Polls every 30s on success.
 * - On error: retries with 30s → 60s → 120s backoff (capped at 2 minutes).
 * - Shows `loading=true` only on the initial fetch; background polls and
 *   retries never flicker the spinner.
 *
 * Accepts an optional `client` adapter so callers can inject a mocked
 * transport in tests, or swap to WebSocket/polling later without internal
 * refactoring.
 *
 * @example
 *   const { profiles, loading, error, refreshProfiles } = useAgentProfiles();
 *
 * @example (custom adapter for tests / WS)
 *   const client = { listProfiles: myWebSocketAdapter };
 *   const state = useAgentProfiles(client);
 */
export function useAgentProfiles(
  client: AgentProfilesClient = defaultClient,
): AgentProfilesState {
  const [profiles, setProfiles] = useState<AgentProfileInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<RetryFetchStatus>("idle");
  const [nextRetryMs, setNextRetryMs] = useState<number | null>(null);
  const [unauthorized, setUnauthorized] = useState(false);

  const handleSuccess = useCallback((snapshot: AgentProfilesSnapshot) => {
    setProfiles(snapshot.profiles);
    setError(null);
    setUnauthorized(false);
  }, []);

  const handleError = useCallback((message: string, err: unknown) => {
    setError(message);
    if (isAuthUnauthorizedError(err)) {
      setUnauthorized(true);
      setProfiles([]);
      return;
    }
    setUnauthorized(false);
    // Keep existing profiles visible so the UI degrades gracefully rather
    // than collapsing to an empty list during a transient backend restart.
  }, []);

  // Auto-retry loop: polls every 30s on success, backs off on error (30→60→120s).
  useRetryFetch({
    fetchFn: client.listProfiles,
    onSuccess: handleSuccess,
    onError: handleError,
    setLoading,
    baseIntervalMs: 30_000,
    retryIntervalMs: 2_000,
    maxIntervalMs: 10_000,
    onStatusChange: setConnectionStatus,
    onNextRetryMsChange: setNextRetryMs,
  });

  // Manual imperative refresh (e.g. after creating a profile).
  const refreshProfiles = useCallback(
    async (options: { keepExisting?: boolean } = {}) => {
      if (!options.keepExisting) {
        setLoading(true);
      }
      setError(null);
      try {
        const snapshot = await client.listProfiles();
        setProfiles(snapshot.profiles);
        setUnauthorized(false);
        setConnectionStatus("connected");
      } catch (err) {
        setError(toErrorMessage(err));
        setConnectionStatus(typeof navigator !== "undefined" && navigator.onLine === false ? "offline" : "reconnecting");
        if (isAuthUnauthorizedError(err)) {
          setUnauthorized(true);
          setProfiles([]);
        } else {
          setUnauthorized(false);
          if (!options.keepExisting) {
            setProfiles([]);
          }
        }
      } finally {
        setLoading(false);
      }
    },
    [client],
  );

  return { profiles, loading, error, connectionStatus, nextRetryMs, unauthorized, refreshProfiles };
}
