import { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Clock,
  Cpu,
  Database,
  Radio,
  Wifi,
  WifiOff,
} from "lucide-react";
import { api } from "@/lib/api";
import type { PlatformStatus, SessionInfo, StatusResponse } from "@/lib/api";
import { timeAgo, isoTimeAgo } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const PLATFORM_STATE_BADGE: Record<string, { variant: "success" | "warning" | "destructive"; label: string }> = {
  connected: { variant: "success", label: "Connected" },
  disconnected: { variant: "warning", label: "Disconnected" },
  fatal: { variant: "destructive", label: "Error" },
};

const GATEWAY_STATE_DISPLAY: Record<string, { badge: "success" | "warning" | "destructive" | "outline"; label: string }> = {
  running: { badge: "success", label: "Running" },
  starting: { badge: "warning", label: "Starting" },
  startup_failed: { badge: "destructive", label: "Failed" },
  stopped: { badge: "outline", label: "Stopped" },
};

function gatewayValue(status: StatusResponse): string {
  if (status.gateway_running) return `PID ${status.gateway_pid}`;
  if (status.gateway_state === "startup_failed") return "Start failed";
  return "Not running";
}

function gatewayBadge(status: StatusResponse) {
  const info = status.gateway_state ? GATEWAY_STATE_DISPLAY[status.gateway_state] : null;
  if (info) return info;
  return status.gateway_running
    ? { badge: "success" as const, label: "Running" }
    : { badge: "outline" as const, label: "Off" };
}

export default function StatusPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);

  useEffect(() => {
    const load = () => {
      api.getStatus().then(setStatus).catch(() => {});
      api.getSessions(50).then((resp) => setSessions(resp.sessions)).catch(() => {});
    };
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!status) {
    return (
      <div className="flex flex-col gap-7" aria-busy="true">
        <section className="overflow-hidden rounded-[32px] bg-[#1d1d1f] px-6 py-8 text-white shadow-[0_30px_80px_rgba(0,0,0,0.16)] sm:px-10 sm:py-12">
          <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr] lg:items-end">
            <div className="max-w-2xl">
              <div className="mb-5 h-4 w-36 animate-pulse rounded-full bg-white/14" />
              <div className="h-12 w-full max-w-xl animate-pulse rounded-2xl bg-white/12 sm:h-16" />
              <div className="mt-4 h-5 w-full max-w-lg animate-pulse rounded-full bg-white/10" />
            </div>
            <div className="grid gap-3 rounded-[24px] bg-white/[0.08] p-4 ring-1 ring-white/10">
              {[0, 1, 2].map((item) => (
                <div key={item} className="flex items-center justify-between border-b border-white/10 pb-3 last:border-0 last:pb-0">
                  <div className="h-4 w-24 animate-pulse rounded-full bg-white/12" />
                  <div className="h-6 w-20 animate-pulse rounded-full bg-white/16" />
                </div>
              ))}
            </div>
          </div>
        </section>
        <div className="grid gap-4 sm:grid-cols-3">
          {["Agent", "Gateway", "Sessions"].map((label) => (
            <Card key={label}>
              <CardHeader>
                <CardTitle>{label}</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-8 w-32 animate-pulse rounded-full bg-muted" />
                <div className="mt-3 h-6 w-20 animate-pulse rounded-full bg-muted" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  const gwBadge = gatewayBadge(status);

  const items = [
    {
      icon: Cpu,
      label: "Agent",
      value: `v${status.version}`,
      badgeText: "Live",
      badgeVariant: "success" as const,
    },
    {
      icon: Radio,
      label: "Gateway",
      value: gatewayValue(status),
      badgeText: gwBadge.label,
      badgeVariant: gwBadge.badge,
    },
    {
      icon: Activity,
      label: "Active Sessions",
      value: status.active_sessions > 0 ? `${status.active_sessions} running` : "None",
      badgeText: status.active_sessions > 0 ? "Live" : "Off",
      badgeVariant: (status.active_sessions > 0 ? "success" : "outline") as "success" | "outline",
    },
  ];

  const platforms = Object.entries(status.gateway_platforms ?? {});
  const activeSessions = sessions.filter((s) => s.is_active);
  const recentSessions = sessions.filter((s) => !s.is_active).slice(0, 5);

  // Collect alerts that need attention
  const alerts: { message: string; detail?: string }[] = [];
  if (status.gateway_state === "startup_failed") {
    alerts.push({
      message: "Gateway failed to start",
      detail: status.gateway_exit_reason ?? undefined,
    });
  }
  const failedPlatforms = platforms.filter(([, info]) => info.state === "fatal" || info.state === "disconnected");
  for (const [name, info] of failedPlatforms) {
    alerts.push({
      message: `${name.charAt(0).toUpperCase() + name.slice(1)} ${info.state === "fatal" ? "error" : "disconnected"}`,
      detail: info.error_message ?? undefined,
    });
  }


  return (
    <div className="flex flex-col gap-7">
      <section className="overflow-hidden rounded-[32px] bg-[#1d1d1f] px-6 py-8 text-white shadow-[0_30px_80px_rgba(0,0,0,0.16)] sm:px-10 sm:py-12">
        <div className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr] lg:items-end">
          <div className="max-w-2xl">
            <p className="mb-3 text-sm font-medium text-white/56">Hermes Control Center</p>
            <h1 className="text-4xl font-semibold tracking-[-0.045em] sm:text-6xl">
              Everything important, in one calm view.
            </h1>
            <p className="mt-4 max-w-xl text-base leading-7 text-white/68 sm:text-lg">
              Monitor the agent, gateway, platforms, and recent work without the noisy generic AI-dashboard chrome.
            </p>
          </div>
          <div className="grid gap-3 rounded-[24px] bg-white/[0.08] p-4 ring-1 ring-white/10 backdrop-blur-xl">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="text-sm text-white/56">Gateway</span>
              <Badge variant={gwBadge.badge}>{gwBadge.label}</Badge>
            </div>
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="text-sm text-white/56">Version</span>
              <span className="font-mono-ui text-sm">v{status.version}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-white/56">Active sessions</span>
              <span className="font-semibold">{status.active_sessions}</span>
            </div>
          </div>
        </div>
      </section>

      {alerts.length > 0 && (
        <div className="rounded-[24px] border border-red-100 bg-red-50 p-5 shadow-sm">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
            <div className="flex flex-col gap-2 min-w-0">
              {alerts.map((alert, i) => (
                <div key={i}>
                  <p className="text-sm font-medium text-destructive">{alert.message}</p>
                  {alert.detail && (
                    <p className="text-xs text-destructive/70 mt-0.5">{alert.detail}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-3">
        {items.map(({ icon: Icon, label, value, badgeText, badgeVariant }) => (
          <Card key={label}>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>

            <CardContent>
              <div className="text-2xl font-bold font-display">{value}</div>

              {badgeText && (
                <Badge variant={badgeVariant} className="mt-2">
                  {badgeVariant === "success" && (
                    <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                  )}
                  {badgeText}
                </Badge>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {platforms.length > 0 && (
        <PlatformsCard platforms={platforms} />
      )}

      {activeSessions.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-success" />
              <CardTitle className="text-base">Active Sessions</CardTitle>
            </div>
          </CardHeader>

          <CardContent className="grid gap-3">
            {activeSessions.map((s) => (
              <div
                key={s.id}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-2xl bg-muted/70 p-4 w-full"
              >
                <div className="flex flex-col gap-1 min-w-0 w-full">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm truncate">{s.title ?? "Untitled"}</span>

                    <Badge variant="success" className="text-[10px] shrink-0">
                      <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                      Live
                    </Badge>
                  </div>

                  <span className="text-xs text-muted-foreground truncate">
                    <span className="font-mono-ui">{(s.model ?? "unknown").split("/").pop()}</span> · {s.message_count} msgs · {timeAgo(s.last_active)}
                  </span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {recentSessions.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Clock className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-base">Recent Sessions</CardTitle>
            </div>
          </CardHeader>

          <CardContent className="grid gap-3">
            {recentSessions.map((s) => (
              <div
                key={s.id}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-2xl bg-muted/70 p-4 w-full"
              >
                <div className="flex flex-col gap-1 min-w-0 w-full">
                  <span className="font-medium text-sm truncate">{s.title ?? "Untitled"}</span>

                  <span className="text-xs text-muted-foreground truncate">
                    <span className="font-mono-ui">{(s.model ?? "unknown").split("/").pop()}</span> · {s.message_count} msgs · {timeAgo(s.last_active)}
                  </span>

                  {s.preview && (
                    <span className="text-xs text-muted-foreground/70 truncate">
                      {s.preview}
                    </span>
                  )}
                </div>

                <Badge variant="outline" className="text-[10px] shrink-0 self-start sm:self-center">
                  <Database className="mr-1 h-3 w-3" />
                  {s.source ?? "local"}
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function PlatformsCard({ platforms }: PlatformsCardProps) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Radio className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-base">Connected Platforms</CardTitle>
        </div>
      </CardHeader>

      <CardContent className="grid gap-3">
        {platforms.map(([name, info]) => {
          const display = PLATFORM_STATE_BADGE[info.state] ?? {
            variant: "outline" as const,
            label: info.state,
          };
          const IconComponent = info.state === "connected" ? Wifi : info.state === "fatal" ? AlertTriangle : WifiOff;

          return (
            <div
              key={name}
              className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 rounded-2xl bg-muted/70 p-4 w-full"
            >
              <div className="flex items-center gap-3 min-w-0 w-full">
                <IconComponent className={`h-4 w-4 shrink-0 ${
                  info.state === "connected"
                    ? "text-success"
                    : info.state === "fatal"
                      ? "text-destructive"
                      : "text-warning"
                }`} />

                <div className="flex flex-col gap-0.5 min-w-0">
                  <span className="text-sm font-medium capitalize truncate">{name}</span>

                  {info.error_message && (
                    <span className="text-xs text-destructive">{info.error_message}</span>
                  )}

                  {info.updated_at && (
                    <span className="text-xs text-muted-foreground">
                      Last update: {isoTimeAgo(info.updated_at)}
                    </span>
                  )}
                </div>
              </div>

              <Badge variant={display.variant} className="shrink-0 self-start sm:self-center">
                {display.variant === "success" && (
                  <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                )}
                {display.label}
              </Badge>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

interface PlatformsCardProps {
  platforms: [string, PlatformStatus][];
}
