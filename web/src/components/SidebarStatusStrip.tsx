import { Link } from "react-router-dom";
import type { StatusResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";
import type { SidebarStatus } from "@/hooks/useSidebarStatus";

/** Gateway + session summary for the System sidebar block (no separate strip chrome). */
export function SidebarStatusStrip({ status }: SidebarStatusStripProps) {
  const { t } = useI18n();

  // First-poll loading state — show the same skeleton pulse the old
  // `status === null` branch did, so we don't visually regress.
  if (status.kind === "loading") {
    return (
      <div className="px-5 py-1.5" aria-hidden>
        <div className="h-2 w-[80%] max-w-full animate-pulse rounded-sm bg-midground/10" />
      </div>
    );
  }

  // Resolve to a flat StatusResponse|null + unreachable flag. Doing this
  // once here keeps the render block below simple.
  let data: StatusResponse | null;
  if (status.kind === "live") {
    data = status.data;
  } else {
    data = status.lastData;
  }
  const unreachable = status.kind === "unreachable";

  const { activeSessionsLabel, gatewayStatusLabel } = t.app;

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
      <div className="flex flex-col gap-1 font-sans text-xs leading-snug tracking-[0.08em]">
        <p className="break-words">
          <span className="text-text-tertiary">{gatewayStatusLabel}</span>{" "}
          <span className={cn("font-medium", resolveTone(status, t))}>
            {resolveLabel(status, t)}
          </span>
        </p>

        <p className="break-words">
          <span className="text-text-tertiary">{activeSessionsLabel}</span>{" "}
          <span
            className={cn(
              "tabular-nums",
              unreachable ? "text-text-tertiary" : "text-text-secondary",
            )}
            title={
              unreachable
                ? t.app.statusUnreachableHint ?? t.app.statusUnreachable
                : undefined
            }
          >
            {data?.active_sessions ?? "—"}
          </span>
        </p>
      </div>
    </Link>
  );
}

/**
 * Pick the label/tone for the "Gateway Status:" line in the sidebar.
 *
 * Returns destructive-tone "Unreachable" whenever the hook has crossed the
 * consecutive-failure threshold, regardless of what the last good response
 * said. This is the user-visible fix for #50270 — without this, the sidebar
 * would still show "Running" in green on a frozen snapshot.
 */
function resolveLabel(
  status: Exclude<SidebarStatus, { kind: "loading" }>,
  t: ReturnType<typeof useI18n>["t"],
): string {
  if (status.kind === "unreachable") {
    return t.app.statusUnreachable ?? t.app.gatewayStrip.failed;
  }
  return gatewayLine(status.data, t).label;
}

function resolveTone(
  status: Exclude<SidebarStatus, { kind: "loading" }>,
  t: ReturnType<typeof useI18n>["t"],
): string {
  if (status.kind === "unreachable") {
    return "text-destructive";
  }
  return gatewayLine(status.data, t).tone;
}

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
  status: SidebarStatus;
}