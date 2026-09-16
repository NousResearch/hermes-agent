import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { useProfileScope } from "@/contexts/useProfileScope";
import { GatewayClient } from "@/lib/gatewayClient";
import {
  isRemoteApprovalExpired,
  normalizePendingApprovals,
  type PendingApprovalsResponse,
  type RemoteApproval,
} from "@/lib/remote-approvals";

const POLL_INTERVAL_MS = 5_000;

type ApprovalChoice = RemoteApproval["choices"][number];

export interface RemoteApprovalsContextValue {
  approvals: RemoteApproval[];
  error: string | null;
  lastUpdatedAt: number | null;
  loading: boolean;
  refresh: () => Promise<void>;
  refreshing: boolean;
  respond: (approval: RemoteApproval, choice: ApprovalChoice) => Promise<void>;
}

const RemoteApprovalsContext = createContext<RemoteApprovalsContextValue | null>(null);

function errorText(reason: unknown): string {
  return reason instanceof Error ? reason.message : String(reason);
}

export function RemoteApprovalsProvider({ children }: { children: ReactNode }) {
  const { currentProfile, profile } = useProfileScope();
  const scope = profile || currentProfile || "default";
  const clientRef = useRef<GatewayClient | null>(null);
  const scopeGenerationRef = useRef(0);
  const requestSequenceRef = useRef(0);
  const [approvals, setApprovals] = useState<RemoteApproval[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    const client = clientRef.current;
    if (!client) return;
    const requestGeneration = scopeGenerationRef.current;
    const requestSequence = ++requestSequenceRef.current;
    setRefreshing(true);
    try {
      if (client.connectionState !== "open") {
        await client.connect();
      }
      const response = await client.request<PendingApprovalsResponse>(
        "approval.pending.all",
        { profile: scope },
      );
      if (
        scopeGenerationRef.current !== requestGeneration
        || requestSequenceRef.current !== requestSequence
        || clientRef.current !== client
      ) return;
      setApprovals(normalizePendingApprovals(response, scope));
      setError(null);
      setLastUpdatedAt(Date.now());
    } catch (reason: unknown) {
      if (
        scopeGenerationRef.current !== requestGeneration
        || requestSequenceRef.current !== requestSequence
        || clientRef.current !== client
      ) return;
      setError(errorText(reason));
    } finally {
      if (
        scopeGenerationRef.current === requestGeneration
        && requestSequenceRef.current === requestSequence
        && clientRef.current === client
      ) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [profile, scope]);

  useEffect(() => {
    scopeGenerationRef.current += 1;
    const generation = scopeGenerationRef.current;
    const client = new GatewayClient();
    clientRef.current = client;
    setApprovals([]);
    setError(null);
    setLastUpdatedAt(null);
    setLoading(true);

    void refresh();
    const interval = setInterval(() => {
      if (scopeGenerationRef.current === generation) void refresh();
    }, POLL_INTERVAL_MS);

    return () => {
      clearInterval(interval);
      if (clientRef.current === client) {
        clientRef.current = null;
        client.close();
      }
    };
  }, [refresh]);

  const respond = useCallback(
    async (approval: RemoteApproval, choice: ApprovalChoice) => {
      if (approval.profile !== scope || !approval.choices.includes(choice) || isRemoteApprovalExpired(approval)) {
        await refresh();
        throw new Error("This approval is no longer available");
      }
      const client = clientRef.current;
      if (!client) throw new Error("Approval gateway is not connected");
      try {
        if (client.connectionState !== "open") await client.connect();
        const result = await client.request<{ resolved?: unknown }>("approval.respond", {
          choice,
          request_id: approval.request_id,
          session_id: approval.session_id,
          profile: scope,
        });
        if (result?.resolved !== 1) {
          throw new Error("Approval was stale, expired, or already resolved");
        }
        await refresh();
      } catch (reason: unknown) {
        await refresh();
        throw reason;
      }
    },
    [refresh, scope],
  );

  const value = useMemo<RemoteApprovalsContextValue>(
    () => ({ approvals, error, lastUpdatedAt, loading, refresh, refreshing, respond }),
    [approvals, error, lastUpdatedAt, loading, refresh, refreshing, respond],
  );

  return <RemoteApprovalsContext.Provider value={value}>{children}</RemoteApprovalsContext.Provider>;
}

export function useRemoteApprovals(): RemoteApprovalsContextValue {
  const value = useContext(RemoteApprovalsContext);
  if (!value) throw new Error("useRemoteApprovals must be used inside RemoteApprovalsProvider");
  return value;
}
