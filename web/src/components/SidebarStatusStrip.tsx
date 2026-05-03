import { Link } from "react-router-dom";
import type { CSSProperties } from "react";
import type { StatusResponse } from "@/lib/api";
import { useSidebarStatus } from "@/hooks/useSidebarStatus";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";

/**
 * Inline-style map for the status dot — avoids the bg-* class generation
 * pitfall with color-mix values (semi-transparent muted-foreground is
 * invisible on a dark sidebar), and guarantees correct colors regardless
 * of JIT scanning.
 */
const DOT_STYLE: Record<string, CSSProperties> = {
  "text-success": { background: "var(--color-success)" },
  "text-warning": { background: "var(--color-warning)" },
  "text-destructive": { background: "var(--color-destructive)" },
  "text-muted-foreground": {
    background: "color-mix(in srgb, var(--midground-base) 45%, transparent)",
    outline: "1px solid color-mix(in srgb, var(--midground-base) 25%, transparent)",
  },
};

const DOT_FALLBACK: CSSProperties = {
  background: "color-mix(in srgb, var(--midground-base) 45%, transparent)",
  outline: "1px solid color-mix(in srgb, var(--midground-base) 25%, transparent)",
};

/** Gateway + session summary for the System sidebar block (no separate strip chrome). */
export function SidebarStatusStrip({ collapsed = false }: { collapsed?: boolean }) {
  const status = useSidebarStatus();
  const { t } = useI18n();

  if (status === null) {
    return collapsed ? (
      <div className="flex justify-center px-2 py-2" aria-hidden>
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-full animate-pulse"
          style={{ background: "color-mix(in srgb, var(--midground-base) 20%, transparent)" }}
        />
      </div>
    ) : (
      <div className="px-5 py-1.5" aria-hidden>
        <div className="h-2 w-[80%] max-w-full animate-pulse rounded-sm bg-midground/10" />
      </div>
    );
  }

  const gw = gatewayLine(status, t);
  const { activeSessionsLabel, gatewayStatusLabel } = t.app;
  const dotStyle = DOT_STYLE[gw.tone] ?? DOT_FALLBACK;

  if (collapsed) {
    return (
      <Link
        to="/sessions"
        title={`${gw.label} · ${status.active_sessions} ${activeSessionsLabel}`}
        className={cn(
          "flex justify-center",
          "px-2 py-2",
          "transition-opacity hover:opacity-80",
          "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/40",
          "focus-visible:ring-inset",
        )}
      >
        <span
          className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
          style={dotStyle}
        />
      </Link>
    );
  }

  return (
    <Link
      to="/sessions"
      title={t.app.statusOverview}
      className={cn(
        "block text-left",
        "px-5 pb-2 pt-0.5",
        "text-muted-foreground/70",
        "transition-colors hover:text-muted-foreground/90",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/40",
        "focus-visible:ring-inset",
      )}
    >
      <div className="flex flex-col gap-1 font-mondwest text-[0.55rem] leading-snug tracking-[0.12em]">
        <p className="break-words">
          <span className="text-muted-foreground/50">{gatewayStatusLabel}</span>{" "}
          <span className={cn("font-medium", gw.tone)}>{gw.label}</span>
        </p>

        <p className="break-words">
          <span className="text-muted-foreground/50">{activeSessionsLabel}</span>{" "}
          <span className="tabular-nums text-muted-foreground/70">
            {status.active_sessions}
          </span>
        </p>
      </div>
    </Link>
  );
}

function gatewayLine(
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
