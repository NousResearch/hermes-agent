import { Link } from "react-router-dom";
import type { StatusResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";
import { useKanbanState } from "@/hooks/useKanbanState";
import { useMemoryData } from "@/hooks/useMemoryData";

/** Gateway + session summary for the System sidebar block (no separate strip chrome). */
export function SidebarStatusStrip({ status }: SidebarStatusStripProps) {
  const { t } = useI18n();

  // Live Kanban board state (poll every 30s)
  const {
    data: kanban,
    loading: kanbanLoading,
    error: kanbanError,
    connectionStatus: kanbanConnectionStatus,
    unauthorized: kanbanUnauthorized,
  } = useKanbanState({ pollingIntervalMs: 30_000 });

  // Live memory content state. Future real-time transports can subscribe here
  // and call useMemoryData's refetch() from a WS/SSE callback without changing
  // the rendering branch below.
  const {
    data: memoryData,
    loading: memoryLoading,
    error: memoryError,
    connectionStatus: memoryConnectionStatus,
    unauthorized: memoryUnauthorized,
  } = useMemoryData("memory");

  if (status === null) {
    return (
      <div className="px-5 py-1.5" aria-hidden>
        <div className="h-2 w-[80%] max-w-full animate-pulse rounded-sm bg-midground/10" />
      </div>
    );
  }

  const gw = gatewayLine(status, t);
  const { activeSessionsLabel, gatewayStatusLabel } = t.app;

  // Derive running task count from kanban columns
  const runningTasks =
    kanban?.columns.find((c) => c.status === "running")?.tasks.length ?? null;
  const blockedTasks =
    kanban?.columns.find((c) => c.status === "blocked")?.tasks.length ?? 0;

  // Memory utilisation percentage
  const memPct =
    memoryData && memoryData.char_limit > 0
      ? Math.round((memoryData.char_count / memoryData.char_limit) * 100)
      : null;

  return (
    <Link
      to="/sessions"
      title={t.app.statusOverview}
      className={cn(
        "block text-left",
        "px-5 pb-2 pt-0.5",
        "text-text-secondary",
        "transition-colors hover:text-midground",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/40",
        "focus-visible:ring-inset",
      )}
    >
      <div className="flex flex-col gap-1 font-mondwest text-xs leading-snug tracking-[0.08em]">
        <p className="break-words">
          <span className="text-text-tertiary">{gatewayStatusLabel}</span>{" "}
          <span className={cn("font-medium", gw.tone)}>{gw.label}</span>
        </p>

        <p className="break-words">
          <span className="text-text-tertiary">{activeSessionsLabel}</span>{" "}
          <span className="tabular-nums text-text-secondary">
            {status.active_sessions}
          </span>
        </p>

        {/* Kanban task summary */}
        {kanbanLoading && !kanban && (
          <p className="break-words">
            <span className="text-text-tertiary">tasks</span>{" "}
            <span className="h-2 w-6 animate-pulse rounded-sm bg-midground/10 inline-block align-middle" />
          </p>
        )}
        {kanbanError && (
          <p className="break-words text-warning text-[10px]">
            kanban {kanbanUnauthorized ? "unauthorized" : kanbanConnectionStatus === "offline" ? "disconnected" : "reconnecting"}
          </p>
        )}
        {!kanbanUnauthorized && !kanbanLoading && runningTasks !== null && (
          <p className="break-words">
            <span className="text-text-tertiary">tasks</span>{" "}
            <span className="tabular-nums text-text-secondary">
              {runningTasks} running
              {blockedTasks > 0 && (
                <span className="text-warning"> · {blockedTasks} blocked</span>
              )}
            </span>
          </p>
        )}

        {/* Memory utilisation */}
        {memoryLoading && !memoryData && (
          <p className="break-words">
            <span className="text-text-tertiary">memory</span>{" "}
            <span className="h-2 w-8 animate-pulse rounded-sm bg-midground/10 inline-block align-middle" />
          </p>
        )}
        {memoryError && (
          <p className="break-words text-warning text-[10px]">
            memory {memoryUnauthorized ? "unauthorized" : memoryConnectionStatus === "offline" ? "disconnected" : "reconnecting"}
          </p>
        )}
        {!memoryUnauthorized && !memoryLoading && memPct !== null && (
          <p className="break-words">
            <span className="text-text-tertiary">memory</span>{" "}
            <span
              className={cn(
                "tabular-nums",
                memPct >= 90
                  ? "text-destructive"
                  : memPct >= 70
                    ? "text-warning"
                    : "text-text-secondary",
              )}
            >
              {memPct}%
            </span>
          </p>
        )}
      </div>
    </Link>
  );
}

// Shared with the legacy App sidebar summary until that call site moves to its own helper file.
// eslint-disable-next-line react-refresh/only-export-components
export function gatewayLine(
  status: StatusResponse,
  t: ReturnType<typeof useI18n>["t"],
): { label: string; tone: string } {
  const g = t.app.gatewayStrip;
  const byState: Record<string, { label: string; tone: string }> = {
    running: { label: g.running, tone: "text-success" },
    starting: { label: g.starting, tone: "text-warning" },
    startup_failed: { label: g.failed, tone: "text-destructive" },
    stopped: { label: g.stopped, tone: "text-muted-foreground" },
  };
  if (status.gateway_state && byState[status.gateway_state]) {
    return byState[status.gateway_state];
  }
  return status.gateway_running
    ? { label: g.running, tone: "text-success" }
    : { label: g.off, tone: "text-muted-foreground" };
}

interface SidebarStatusStripProps {
  status: StatusResponse | null;
}
