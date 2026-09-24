import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ActionStatusResponse } from "@/lib/api";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { sharedGatewayProfiles, sharedGatewayRestartedMessage } from "@/lib/shared-gateway";
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
    if (!activeAction) return;
    const name = ACTION_NAMES[activeAction];
    let cancelled = false;

    const poll = async () => {
      try {
        const resp = await api.getActionStatus(name);
        if (cancelled) return;
        setActionStatus(resp);
        if (!resp.running) {
          const ok = resp.exit_code === 0;
          // A restart of the shared multiplexer reconnected every bot on the device: name the count.
          const shared =
            ok && activeAction === "restart"
              ? sharedGatewayProfiles(await api.getStatus().catch(() => null))
              : null;
          if (cancelled) return;
          setToast({
            type: ok ? "success" : "error",
            message: ok
              ? shared
                ? sharedGatewayRestartedMessage(shared.length)
                : t.status.actionFinished
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
  }, [activeAction, t.status.actionFinished, t.status.actionFailed]);

  const runAction = useCallback(
    async (action: SystemAction) => {
      setPendingAction(action);
      setActionStatus(null);
      try {
        if (action === "restart") {
          await api.restartGateway();
          setActiveAction(action);
        } else {
          const resp = await api.updateHermes();
          // Some installs cannot apply updates from inside the dashboard. The
          // endpoint returns a structured {ok:false, message, update_command}
          // envelope instead of spawning the action; surface that guidance
          // rather than polling a synthetic failed action.
          if (!resp.ok) {
            const cmd = resp.update_command ? `  ${resp.update_command}` : "";
            setToast({
              type: "success",
              message:
                (resp.message ??
                  "Updates don't apply from this dashboard.") +
                cmd,
            });
            return;
          }
          setActiveAction(action);
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err);
        setToast({
          type: "error",
          message: `${t.status.actionFailed}: ${detail}`,
        });
      } finally {
        setPendingAction(null);
      }
    },
    [t.status.actionFailed],
  );

  const dismissLog = useCallback(() => {
    setActiveAction(null);
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
