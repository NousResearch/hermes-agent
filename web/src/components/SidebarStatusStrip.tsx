import { Link } from "react-router-dom";
import type { StatusResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { useI18n } from "@/i18n";
import { gatewayLine } from "@/components/sidebarStatusHelpers";

/**
 * Agent Runner-style gateway rail for the app shell.
 *
 * The old sidebar block hid the important operator question — “is the gateway
 * alive and where should I look?” — in two low-contrast text rows. This rail
 * keeps the same click target (/sessions) but surfaces hot counters and a
 * platform flow strip, matching the dense-at-a-glance visual language used by
 * Agent Runner and the Kanban overview.
 */
export function SidebarStatusStrip({ status }: SidebarStatusStripProps) {
  const { t } = useI18n();

  if (status === null) {
    return (
      <div className="px-3 pb-2 pt-1" aria-hidden>
        <div className="grid gap-2 rounded-md border border-current/10 bg-card/40 p-2">
          <div className="h-2 w-[72%] max-w-full animate-pulse rounded-sm bg-midground/10" />
          <div className="grid grid-cols-3 gap-1.5">
            <div className="h-10 animate-pulse rounded-sm bg-midground/5" />
            <div className="h-10 animate-pulse rounded-sm bg-midground/5" />
            <div className="h-10 animate-pulse rounded-sm bg-midground/5" />
          </div>
        </div>
      </div>
    );
  }

  const gw = gatewayLine(status, t);
  const platformEntries = Object.entries(status.gateway_platforms ?? {});
  const platformCounts = summarizePlatforms(platformEntries);
  const attentionCount = platformCounts.disconnected + platformCounts.fatal;
  const { activeSessionsLabel } = t.app;

  return (
    <Link
      to="/sessions"
      title={t.app.statusOverview}
      className={cn(
        "group/status block text-left",
        "px-3 pb-2 pt-1",
        "text-text-secondary transition-colors hover:text-midground",
        "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-midground/40",
        "focus-visible:ring-inset",
      )}
    >
      <div
        className={cn(
          "relative overflow-hidden rounded-md border border-current/15",
          "bg-card/45 p-2 shadow-[inset_0_1px_0_color-mix(in_srgb,var(--color-foreground)_6%,transparent)]",
          "transition-colors group-hover/status:border-current/30",
        )}
      >
        <span
          aria-hidden
          className={cn("absolute inset-y-2 left-0 w-px", gatewayToneBar(gw.tone))}
        />

        <div className="flex min-w-0 items-start justify-between gap-2 pl-1.5">
          <div className="min-w-0">
            <p className="font-mondwest text-display text-[0.62rem] uppercase leading-none tracking-[0.14em] text-text-tertiary">
              Gateway
            </p>
            <p className="mt-1 truncate font-mono-ui text-[0.7rem] uppercase tracking-[0.08em] text-text-secondary">
              {gw.label}
            </p>
          </div>

          <span
            className={cn(
              "mt-0.5 inline-flex h-2 w-2 shrink-0 rounded-full",
              gatewayToneDot(gw.tone),
              status.gateway_running && "animate-pulse",
            )}
            aria-hidden
          />
        </div>

        <div className="mt-2 grid grid-cols-3 gap-1.5">
          <GatewayMetric
            label={activeSessionsLabel.replace(/:$/, "")}
            value={status.active_sessions}
          />
          <GatewayMetric
            label="Platforms"
            value={platformCounts.connected}
            suffix={platformCounts.total > 0 ? `/${platformCounts.total}` : undefined}
          />
          <GatewayMetric
            label="Attention"
            value={attentionCount}
            hot={attentionCount > 0}
          />
        </div>

        {platformEntries.length > 0 && (
          <div
            aria-label="Gateway platform states"
            className="mt-2 flex gap-1 overflow-hidden pl-1.5"
          >
            {platformEntries.slice(0, 5).map(([name, info]) => (
              <span
                key={name}
                title={`${name}: ${info.state}`}
                className={cn(
                  "inline-flex min-w-0 items-center gap-1 rounded-full border border-current/10 px-1.5 py-0.5",
                  "font-mono-ui text-[0.58rem] uppercase tracking-[0.06em]",
                  platformChipTone(info.state),
                )}
              >
                <span className="h-1 w-1 shrink-0 rounded-full bg-current" aria-hidden />
                <span className="truncate">{name}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </Link>
  );
}

function GatewayMetric({ label, value, suffix, hot = false }: GatewayMetricProps) {
  return (
    <div
      className={cn(
        "min-w-0 rounded-sm border border-current/10 bg-background-base/20 px-1.5 py-1",
        hot && "border-destructive/50 text-destructive",
      )}
    >
      <p className="font-mono-ui text-[1rem] font-bold leading-none tabular-nums text-text-primary">
        {value}
        {suffix && (
          <span className="text-[0.62rem] font-medium text-text-tertiary">
            {suffix}
          </span>
        )}
      </p>
      <p className="mt-0.5 truncate font-mondwest text-[0.56rem] uppercase leading-none tracking-[0.08em] text-text-tertiary">
        {label}
      </p>
    </div>
  );
}

function summarizePlatforms(platforms: [string, { state: string }][]) {
  return platforms.reduce(
    (acc, [, info]) => {
      acc.total += 1;
      if (info.state === "connected") acc.connected += 1;
      else if (info.state === "fatal") acc.fatal += 1;
      else if (info.state === "disconnected") acc.disconnected += 1;
      else acc.other += 1;
      return acc;
    },
    { connected: 0, disconnected: 0, fatal: 0, other: 0, total: 0 },
  );
}

function platformChipTone(state: string): string {
  if (state === "connected") return "text-success bg-success/5";
  if (state === "fatal") return "text-destructive bg-destructive/5";
  if (state === "disconnected") return "text-warning bg-warning/5";
  return "text-text-tertiary bg-card/30";
}

function gatewayToneDot(tone: string): string {
  const byTone: Record<string, string> = {
    "text-success": "bg-success",
    "text-warning": "bg-warning",
    "text-destructive": "bg-destructive",
    "text-muted-foreground": "bg-muted-foreground",
  };
  return byTone[tone] ?? "bg-muted-foreground";
}

function gatewayToneBar(tone: string): string {
  const byTone: Record<string, string> = {
    "text-success": "bg-success",
    "text-warning": "bg-warning",
    "text-destructive": "bg-destructive",
    "text-muted-foreground": "bg-muted-foreground",
  };
  return byTone[tone] ?? "bg-muted-foreground";
}

interface GatewayMetricProps {
  hot?: boolean;
  label: string;
  suffix?: string;
  value: number;
}

interface SidebarStatusStripProps {
  status: StatusResponse | null;
}
