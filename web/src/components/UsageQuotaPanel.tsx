/**
 * UsageQuotaPanel — per-provider rate-limit gauges for the Chat side panel.
 *
 * Polls the same `system.usage_meter` JSON-RPC method the Ink TUI's
 * `useUsageMeterPoll` hook drives (see ui-tui/src/app/useUsageMeterPoll.ts),
 * over the dashboard's own `/api/ws` sidecar — `system.usage_meter` is a
 * session-independent method (its handler takes no session id), so this
 * panel opens its own lightweight GatewayClient rather than riding
 * ChatSidebar's sidecar session.
 *
 * Each provider ("Claude", "Codex") reports one window per rolling quota
 * (a short session window plus a longer weekly window); each window carries
 * `used_percent` and a `reset_at` timestamp the panel counts down to.
 */

import { Card } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Progress } from "@nous-research/ui/ui/components/progress";

import { GatewayClient } from "@/lib/gatewayClient";
import { cn } from "@/lib/utils";
import { useEffect, useMemo, useRef, useState } from "react";

interface UsageWindow {
  label: string;
  used_percent: number;
  remaining_percent?: number;
  reset_at?: string | null;
}

interface ProviderUsage {
  available: boolean;
  unavailable_reason?: string;
  windows: UsageWindow[];
}

interface SystemUsageMeterResponse {
  claude?: ProviderUsage;
  codex?: ProviderUsage;
}

const POLL_MS = 30_000;

const PROVIDERS: { key: keyof SystemUsageMeterResponse; label: string }[] = [
  { key: "claude", label: "Claude" },
  { key: "codex", label: "Codex" },
];

/** "3d 4h" past a day out, "2h 26m" under a day, "48m" under an hour, "—" if unknown. */
function formatCountdown(resetAt: string | null | undefined, now: number): string {
  if (!resetAt) return "—";
  const target = Date.parse(resetAt);
  if (Number.isNaN(target)) return "—";
  const ms = target - now;
  if (ms <= 0) return "resetting…";
  const totalMinutes = Math.round(ms / 60_000);
  const days = Math.floor(totalMinutes / 1440);
  const hours = Math.floor((totalMinutes % 1440) / 60);
  const minutes = totalMinutes % 60;
  if (days > 0) return `${days}d ${hours}h left`;
  if (hours > 0) return `${hours}h ${minutes}m left`;
  return `${minutes}m left`;
}

/** Severity tint for the gauge fill — a quiet nudge before a window is exhausted. */
function gaugeTone(usedPercent: number): string {
  if (usedPercent >= 90) return "bg-destructive/35";
  if (usedPercent >= 70) return "bg-warning/35";
  return "bg-success/25";
}

function ProviderGauges({ label, usage }: { label: string; usage: ProviderUsage | undefined }) {
  const [now, setNow] = useState(() => Date.now());

  // Countdown-only tick; the data itself refreshes on the shared poll below.
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 30_000);
    return () => clearInterval(id);
  }, []);

  const available = usage?.available && usage.windows.length > 0;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-text-secondary">{label}</span>
        {!available && (
          <Badge tone="secondary" className="shrink-0">
            n/a
          </Badge>
        )}
      </div>

      {available ? (
        usage!.windows.map((w) => {
          // Longer-horizon windows (weekly, etc.) change slowly and matter
          // less turn-to-turn than the short session window — render them
          // smaller so the session gauge stays the visual anchor.
          const isLongHorizon = /week/i.test(w.label);

          return (
            <div key={w.label} className="flex flex-col gap-0.5">
              <div
                className={cn(
                  "flex items-center justify-between tracking-wide text-text-tertiary",
                  isLongHorizon ? "text-[9px] opacity-70" : "text-[10px]",
                )}
              >
                <span className="truncate">{w.label}</span>
                <span className="shrink-0 tabular-nums">{formatCountdown(w.reset_at, now)}</span>
              </div>
              <Progress
                value={Math.max(0, Math.min(100, w.used_percent))}
                barProps={{
                  className: cn(gaugeTone(w.used_percent), isLongHorizon && "py-0.5"),
                }}
                className={isLongHorizon ? "min-h-[0.9rem] opacity-70" : "min-h-[1.4rem]"}
              >
                {!isLongHorizon && <span className="text-[10px]">{Math.round(w.used_percent)}%</span>}
              </Progress>
            </div>
          );
        })
      ) : (
        <div className="wrap-break-word text-[10px] text-text-tertiary">
          {usage?.unavailable_reason ?? "unavailable"}
        </div>
      )}
    </div>
  );
}

export function UsageQuotaPanel({ className }: { className?: string }) {
  const [data, setData] = useState<SystemUsageMeterResponse | null>(null);
  const gwRef = useRef<GatewayClient | null>(null);

  useEffect(() => {
    let cancelled = false;
    const gw = new GatewayClient();
    gwRef.current = gw;

    const poll = async () => {
      try {
        await gw.connect();
        const res = await gw.request<SystemUsageMeterResponse>("system.usage_meter", {});
        if (!cancelled) setData(res);
      } catch {
        // Best-effort: keep the last good dual-provider reading during
        // transient failures (mirrors ui-tui's useUsageMeterPoll).
      }
    };

    void poll();
    const id = setInterval(() => void poll(), POLL_MS);

    return () => {
      cancelled = true;
      clearInterval(id);
      gw.close();
    };
  }, []);

  const anyAvailable = useMemo(
    () => PROVIDERS.some(({ key }) => data?.[key]?.available && (data[key]?.windows.length ?? 0) > 0),
    [data],
  );

  if (data === null) {
    // Nothing to show yet; avoid flashing an empty card before the first poll lands.
    return null;
  }

  return (
    <Card className={cn("flex flex-col gap-3 px-3 py-2", className)}>
      <div className="text-display text-xs tracking-wider text-text-tertiary">usage</div>

      {PROVIDERS.map(({ key, label }) => (
        <ProviderGauges key={key} label={label} usage={data[key]} />
      ))}

      {!anyAvailable && (
        <div className="text-[10px] text-text-tertiary">
          No provider usage available right now — retrying every 30s.
        </div>
      )}
    </Card>
  );
}
