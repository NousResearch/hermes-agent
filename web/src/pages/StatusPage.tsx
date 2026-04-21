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
import { Cell, Grid } from "@nous-research/ui";
import { api } from "@/lib/api";
import type { PlatformStatus, SessionInfo, StatusResponse } from "@/lib/api";
import { timeAgo, isoTimeAgo } from "@/lib/utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useI18n } from "@/i18n";

export default function StatusPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const { t } = useI18n();

  useEffect(() => {
    const load = () => {
      api
        .getStatus()
        .then(setStatus)
        .catch(() => {});
      api
        .getSessions(50)
        .then((resp) => setSessions(resp.sessions))
        .catch(() => {});
    };
    load();
    const interval = setInterval(load, 5000);
    return () => clearInterval(interval);
  }, []);

  if (!status) {
    return (
      <div className="flex items-center justify-center py-24">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  const PLATFORM_STATE_BADGE: Record<
    string,
    { variant: "success" | "warning" | "destructive"; label: string }
  > = {
    connected: { variant: "success", label: t.status.connected },
    disconnected: { variant: "warning", label: t.status.disconnected },
    fatal: { variant: "destructive", label: t.status.error },
  };

  const GATEWAY_STATE_DISPLAY: Record<
    string,
    { badge: "success" | "warning" | "destructive" | "outline"; label: string }
  > = {
    running: { badge: "success", label: t.status.running },
    starting: { badge: "warning", label: t.status.starting },
    startup_failed: { badge: "destructive", label: t.status.failed },
    stopped: { badge: "outline", label: t.status.stopped },
  };

  function gatewayValue(): string {
    if (status!.gateway_running && status!.gateway_health_url)
      return status!.gateway_health_url;
    if (status!.gateway_running && status!.gateway_pid)
      return `${t.status.pid} ${status!.gateway_pid}`;
    if (status!.gateway_running) return t.status.runningRemote;
    if (status!.gateway_state === "startup_failed") return t.status.startFailed;
    return t.status.notRunning;
  }

  function gatewayBadge() {
    const info = status!.gateway_state
      ? GATEWAY_STATE_DISPLAY[status!.gateway_state]
      : null;
    if (info) return info;
    return status!.gateway_running
      ? { badge: "success" as const, label: t.status.running }
      : { badge: "outline" as const, label: t.common.off };
  }

  const gwBadge = gatewayBadge();

  const items = [
    {
      icon: Cpu,
      label: t.status.agent,
      value: `v${status.version}`,
      badgeText: t.common.live,
      badgeVariant: "success" as const,
    },
    {
      icon: Radio,
      label: t.status.gateway,
      value: gatewayValue(),
      badgeText: gwBadge.label,
      badgeVariant: gwBadge.badge,
    },
    {
      icon: Activity,
      label: t.status.activeSessions,
      value:
        status.active_sessions > 0
          ? `${status.active_sessions} ${t.status.running.toLowerCase()}`
          : t.status.noneRunning,
      badgeText: status.active_sessions > 0 ? t.common.live : t.common.off,
      badgeVariant: (status.active_sessions > 0 ? "success" : "outline") as
        | "success"
        | "outline",
    },
  ];

  const platforms = Object.entries(status.gateway_platforms ?? {});
  const activeSessions = sessions.filter((s) => s.is_active);
  const recentSessions = sessions.filter((s) => !s.is_active).slice(0, 5);

  // Collect alerts that need attention
  const alerts: { message: string; detail?: string }[] = [];
  if (status.gateway_state === "startup_failed") {
    alerts.push({
      message: t.status.gatewayFailedToStart,
      detail: status.gateway_exit_reason ?? undefined,
    });
  }
  const failedPlatforms = platforms.filter(
    ([, info]) => info.state === "fatal" || info.state === "disconnected",
  );
  for (const [name, info] of failedPlatforms) {
    const stateLabel =
      info.state === "fatal"
        ? t.status.platformError
        : t.status.platformDisconnected;
    alerts.push({
      message: `${name.charAt(0).toUpperCase() + name.slice(1)} ${stateLabel}`,
      detail: info.error_message ?? undefined,
    });
  }

  return (
    <div className="flex flex-col gap-6">
      {alerts.length > 0 && (
        <div className="border border-destructive/30 bg-destructive/[0.06] p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
            <div className="flex flex-col gap-2 min-w-0">
              {alerts.map((alert, i) => (
                <div key={i}>
                  <p className="text-sm font-medium text-destructive">
                    {alert.message}
                  </p>
                  {alert.detail && (
                    <p className="text-xs text-destructive/70 mt-0.5">
                      {alert.detail}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <Grid className="border-b lg:!grid-cols-3">
        {items.map(({ icon: Icon, label, value, badgeText, badgeVariant }) => (
          <Cell
            key={label}
            className="flex min-w-0 flex-col gap-2 overflow-hidden"
          >
            <div className="flex items-center justify-between">
              <CardTitle className="text-sm font-medium">{label}</CardTitle>
              <Icon className="h-4 w-4 text-muted-foreground" />
            </div>

            <div
              className="truncate text-2xl font-bold font-mondwest"
              title={value}
            >
              {value}
            </div>

            {badgeText && (
              <Badge variant={badgeVariant} className="self-start">
                {badgeVariant === "success" && (
                  <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                )}
                {badgeText}
              </Badge>
            )}
          </Cell>
        ))}
      </Grid>

      {platforms.length > 0 && (
        <PlatformsCard
          platforms={platforms}
          platformStateBadge={PLATFORM_STATE_BADGE}
        />
      )}

      {activeSessions.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-success" />
              <CardTitle className="text-base">
                {t.status.activeSessions}
              </CardTitle>
            </div>
          </CardHeader>

          <CardContent className="grid gap-3">
            {activeSessions.map((s) => (
              <div
                key={s.id}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border border-border p-3 w-full"
              >
                <div className="flex flex-col gap-1 min-w-0 w-full">
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-sm truncate">
                      {s.title ?? t.common.untitled}
                    </span>

                    <Badge variant="success" className="text-[10px] shrink-0">
                      <span className="mr-1 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-current" />
                      {t.common.live}
                    </Badge>
                  </div>

                  <span className="text-xs text-muted-foreground truncate">
                    <span className="font-mono-ui">
                      {(s.model ?? t.common.unknown).split("/").pop()}
                    </span>{" "}
                    · {s.message_count} {t.common.msgs} ·{" "}
                    {timeAgo(s.last_active)}
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
              <CardTitle className="text-base">
                {t.status.recentSessions}
              </CardTitle>
            </div>
          </CardHeader>

          <CardContent className="grid gap-3">
            {recentSessions.map((s) => (
              <div
                key={s.id}
                className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border border-border p-3 w-full"
              >
                <div className="flex flex-col gap-1 min-w-0 w-full">
                  <span className="font-medium text-sm truncate">
                    {s.title ?? t.common.untitled}
                  </span>

                  <span className="text-xs text-muted-foreground truncate">
                    <span className="font-mono-ui">
                      {(s.model ?? t.common.unknown).split("/").pop()}
                    </span>{" "}
                    · {s.message_count} {t.common.msgs} ·{" "}
                    {timeAgo(s.last_active)}
                  </span>

                  {s.preview && (
                    <span className="text-xs text-muted-foreground/70 truncate">
                      {s.preview}
                    </span>
                  )}
                </div>

                <Badge
                  variant="outline"
                  className="text-[10px] shrink-0 self-start sm:self-center"
                >
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

function PlatformsCard({ platforms, platformStateBadge }: PlatformsCardProps) {
  const { t } = useI18n();

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Radio className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-base">
            {t.status.connectedPlatforms}
          </CardTitle>
        </div>
      </CardHeader>

      <CardContent className="grid gap-3">
        {platforms.map(([name, info]) => {
          const display = platformStateBadge[info.state] ?? {
            variant: "outline" as const,
            label: info.state,
          };
          const IconComponent =
            info.state === "connected"
              ? Wifi
              : info.state === "fatal"
                ? AlertTriangle
                : WifiOff;

          return (
            <div
              key={name}
              className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border border-border p-3 w-full"
            >
              <div className="flex items-center gap-3 min-w-0 w-full">
                <IconComponent
                  className={`h-4 w-4 shrink-0 ${
                    info.state === "connected"
                      ? "text-success"
                      : info.state === "fatal"
                        ? "text-destructive"
                        : "text-warning"
                  }`}
                />

                <div className="flex flex-col gap-0.5 min-w-0">
                  <span className="text-sm font-medium capitalize truncate">
                    {name}
                  </span>

                  {info.error_message && (
                    <span className="text-xs text-destructive">
                      {info.error_message}
                    </span>
                  )}

                  {info.updated_at && (
                    <span className="text-xs text-muted-foreground">
                      {t.status.lastUpdate}: {isoTimeAgo(info.updated_at)}
                    </span>
                  )}
                </div>
              </div>

              <Badge
                variant={display.variant}
                className="shrink-0 self-start sm:self-center"
              >
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
  platformStateBadge: Record<
    string,
    { variant: "success" | "warning" | "destructive"; label: string }
  >;
}
