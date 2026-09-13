import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ActionStatusResponse } from "@/lib/api";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useI18n } from "@/i18n";
import {
  SystemActionsContext,
  type SystemAction,
} from "./system-actions-context";

const ACTION_NAMES: Record<SystemAction, string> = {
  restart: "gateway-restart",
  update: "hermes-update",
};

export function SystemActionsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [pendingAction, setPendingAction] = useState<SystemAction | null>(null);
  const [activeAction, setActiveAction] = useState<SystemAction | null>(null);
  const [activeInvocationId, setActiveInvocationId] = useState<string | null>(null);
  const [actionGeneration, setActionGeneration] = useState(0);
  const [actionStatus, setActionStatus] = useState<ActionStatusResponse | null>(
    null,
  );
  const [toast, setToast] = useState<ToastState | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!activeAction || !activeInvocationId) return;
    const name = ACTION_NAMES[activeAction];
    let cancelled = false;

    const poll = async () => {
      try {
        const resp = await api.getActionStatus(name, 200, activeInvocationId);
        if (cancelled) return;
        setActionStatus(resp);
        if (!resp.running) {
          const ok = resp.exit_code === 0;
          setToast({
            type: ok ? "success" : "error",
            message: ok
              ? t.status.actionFinished
              : `${t.status.actionFailed} (exit ${resp.exit_code ?? "?"})`,
          });
          return;
        }
      } catch {
        // transient fetch error; keep polling
      }
      if (!cancelled) setTimeout(poll, 1500);
    };

    poll();
    return () => {
      cancelled = true;
    };
  }, [
    activeAction,
    activeInvocationId,
    actionGeneration,
    t.status.actionFinished,
    t.status.actionFailed,
  ]);

  const runAction = useCallback(
    async (action: SystemAction) => {
      const label = action === "restart"
        ? t.status.restartGateway
        : t.status.updateHermes;
      if (typeof window === "undefined" || !window.confirm(`${label}?`)) {
        return;
      }
      setPendingAction(action);
      setActionStatus(null);
      try {
        const launch = action === "restart"
          ? await api.restartGateway()
          : await api.updateHermes();
        setActionGeneration((generation) => generation + 1);
        setActiveInvocationId(launch.invocation_id);
        setActiveAction(action);
      } catch {
        setActiveAction(null);
        setActiveInvocationId(null);
        setActionStatus(null);
        setToast({
          type: "error",
          message: t.status.actionFailed,
        });
      } finally {
        setPendingAction(null);
      }
    },
    [t.status.actionFailed, t.status.restartGateway, t.status.updateHermes],
  );

  const dismissLog = useCallback(() => {
    setActiveAction(null);
    setActiveInvocationId(null);
    setActionStatus(null);
  }, []);

  const isRunning = activeAction !== null && actionStatus?.running !== false;
  const isBusy = pendingAction !== null || isRunning;

  return (
    <SystemActionsContext.Provider
      value={{
        actionStatus,
        activeAction,
        dismissLog,
        isBusy,
        isRunning,
        pendingAction,
        runAction,
      }}
    >
      {children}
      <Toast toast={toast} />
    </SystemActionsContext.Provider>
  );
}

interface ToastState {
  message: string;
  type: "success" | "error";
}
