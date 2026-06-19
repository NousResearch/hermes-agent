import { useEffect, useMemo, useState, type ComponentType, type ReactNode } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  Clock,
  Cpu,
  MessageSquare,
  PlugZap,
  Radio,
  Settings,
  ShieldCheck,
  Sparkles,
  Terminal,
  Workflow,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type {
  MessagingPlatform,
  ModelInfoResponse,
  PaginatedSessions,
  SessionStoreStats,
  StatusResponse,
} from "@/lib/api";

interface DashboardData {
  channels: MessagingPlatform[];
  model: ModelInfoResponse | null;
  sessions: PaginatedSessions | null;
  stats: SessionStoreStats | null;
  status: StatusResponse | null;
}

const emptyData: DashboardData = {
  channels: [],
  model: null,
  sessions: null,
  stats: null,
  status: null,
};

const gatewayTone: Record<string, "success" | "warning" | "destructive" | "outline"> = {
  running: "success",
  starting: "warning",
  startup_failed: "destructive",
  stopped: "outline",
};

function gatewayLabel(status: StatusResponse | null): string {
  if (!status) return "Unknown";
  if (status.gateway_state) return status.gateway_state.replace(/_/g, " ");
  return status.gateway_running ? "running" : "off";
}

function formatTime(seconds: number | null | undefined): string {
  if (!seconds) return "—";
  const date = new Date(seconds * 1000);
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function DashboardPage() {
  const [data, setData] = useState<DashboardData>(emptyData);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Promise.allSettled([
      api.getStatus(),
      api.getSessions(5),
      api.getMessagingPlatforms(),
      api.getModelInfo(),
      api.getSessionStats(),
    ])
      .then(([status, sessions, messaging, model, stats]) => {
        if (cancelled) return;
        setData({
          status: status.status === "fulfilled" ? status.value : null,
          sessions: sessions.status === "fulfilled" ? sessions.value : null,
          channels: messaging.status === "fulfilled" ? messaging.value.platforms : [],
          model: model.status === "fulfilled" ? model.value : null,
          stats: stats.status === "fulfilled" ? stats.value : null,
        });
        const firstError = [status, sessions, messaging, model, stats].find(
          (result) => result.status === "rejected",
        );
        setError(firstError?.status === "rejected" ? String(firstError.reason) : null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const connectedChannels = useMemo(
    () => data.channels.filter((channel) => channel.state === "connected"),
    [data.channels],
  );
  const homeChannels = useMemo(
    () => data.channels.filter((channel) => channel.home_channel !== null),
    [data.channels],
  );
  const recentSessions = data.sessions?.sessions ?? [];
  const statusTone = gatewayTone[data.status?.gateway_state ?? ""] ?? (
    data.status?.gateway_running ? "success" : "outline"
  );

  if (loading) {
    return (
      <div className="flex min-h-[18rem] items-center justify-center">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5 pb-6">
      {error ? (
        <Card className="border-warning/40">
          <CardContent className="p-4 text-sm text-muted-foreground">
            Some dashboard data could not be loaded: {error}
          </CardContent>
        </Card>
      ) : null}

      <section className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(20rem,0.75fr)]">
        <Card className="overflow-hidden border-midground/30 bg-card/70">
          <CardContent className="relative p-5 sm:p-6">
            <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_18%_20%,rgba(69,229,214,0.16),transparent_32%),radial-gradient(circle_at_82%_0%,rgba(255,255,255,0.06),transparent_28%)]" />
            <div className="relative flex flex-col gap-5">
              <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                <div className="space-y-2">
                  <Badge tone="outline" className="w-fit uppercase tracking-[0.16em]">
                    contextual command deck
                  </Badge>
                  <h2 className="font-expanded text-2xl font-bold tracking-[0.06em] text-midground sm:text-3xl">
                    One screen for the work; deeper rooms one click away.
                  </h2>
                  <p className="max-w-2xl text-sm leading-6 text-muted-foreground">
                    A mixed layout: status, recent context, and the next likely actions stay visible here. Room Hubs moved into their own page so the main dashboard does not become a railway timetable in a waistcoat.
                  </p>
                </div>
                <Link
                  to="/chat"
                  className="inline-flex shrink-0 items-center justify-center gap-2 rounded-sm border border-midground/40 bg-midground px-4 py-2 text-sm font-medium uppercase tracking-[0.08em] text-background transition-colors hover:bg-midground/90"
                >
                  <Terminal className="h-4 w-4" />
                  Open chat
                </Link>
              </div>

              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
                <MetricCard
                  icon={Activity}
                  label="Gateway"
                  value={gatewayLabel(data.status)}
                  badge={<Badge tone={statusTone}>{data.status?.gateway_running ? "live" : "idle"}</Badge>}
                />
                <MetricCard
                  icon={MessageSquare}
                  label="Active sessions"
                  value={String(data.status?.active_sessions ?? 0)}
                  detail={`${data.stats?.total ?? data.sessions?.total ?? 0} total`}
                />
                <MetricCard
                  icon={Radio}
                  label="Channels"
                  value={`${connectedChannels.length}/${data.channels.length}`}
                  detail="connected"
                />
                <MetricCard
                  icon={Cpu}
                  label="Model"
                  value={data.model?.provider || "—"}
                  detail={data.model?.model || "not configured"}
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4 text-midground" />
              Context rail
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <ContextAction
              icon={Clock}
              title="Resume"
              detail={recentSessions[0]?.title || recentSessions[0]?.preview || "No recent session yet"}
              to="/sessions"
            />
            <ContextAction
              icon={Workflow}
              title="Automate"
              detail="Cron, webhooks, and gateway actions"
              to="/cron"
            />
            <ContextAction
              icon={PlugZap}
              title="Connect"
              detail={`${homeChannels.length} home hub${homeChannels.length === 1 ? "" : "s"} configured`}
              to="/hubs"
            />
            <ContextAction
              icon={Settings}
              title="Tune"
              detail="Models, skills, config, and keys"
              to="/models"
            />
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] 2xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between gap-2 text-base">
              Recent context
              <Link className="text-xs font-normal text-primary hover:underline" to="/sessions">
                All sessions
              </Link>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {recentSessions.length > 0 ? recentSessions.slice(0, 4).map((session) => (
              <Link
                key={session.id}
                to="/sessions"
                className="block rounded-sm border border-border/70 bg-background/30 p-3 transition-colors hover:border-midground/50 hover:bg-midground/5"
              >
                <div className="flex items-center justify-between gap-3">
                  <p className="truncate text-sm font-medium text-foreground">
                    {session.title || session.preview || "Untitled session"}
                  </p>
                  {session.is_active ? <Badge tone="success">active</Badge> : null}
                </div>
                <p className="mt-1 truncate text-xs text-muted-foreground">
                  {session.source || "session"} · {formatTime(session.last_active)}
                </p>
              </Link>
            )) : (
              <EmptyLine>No context yet. Admirably clean, suspiciously quiet.</EmptyLine>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center justify-between gap-2 text-base">
              Room Hubs
              <Link className="text-xs font-normal text-primary hover:underline" to="/hubs">
                Open hubs
              </Link>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {homeChannels.length > 0 ? homeChannels.slice(0, 4).map((platform) => (
              <div key={platform.id} className="rounded-sm border border-border/70 bg-background/30 p-3">
                <div className="flex items-center justify-between gap-3">
                  <p className="truncate text-sm font-medium text-foreground">
                    {platform.home_channel?.name || platform.name}
                  </p>
                  <Badge tone={platform.state === "connected" ? "success" : "outline"}>
                    {platform.state.replace(/_/g, " ")}
                  </Badge>
                </div>
                <p className="mt-1 truncate font-courier text-xs text-muted-foreground">
                  {platform.name} · {platform.home_channel?.chat_id}
                </p>
              </div>
            )) : (
              <EmptyLine>No home channels yet. Set one from a messaging room.</EmptyLine>
            )}
          </CardContent>
        </Card>

        <Card className="lg:col-span-2 2xl:col-span-1">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-base">
              <ShieldCheck className="h-4 w-4 text-midground" />
              System posture
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-1">
            <PostureLine label="Auth gate" value={data.status?.auth_required ? "enabled" : "loopback / local"} />
            <PostureLine label="Config" value={`${data.status?.config_version ?? "—"}/${data.status?.latest_config_version ?? "—"}`} />
            <PostureLine label="Hermes" value={data.status?.version ?? "—"} />
            <PostureLine label="Release" value={data.status?.release_date ?? "—"} />
          </CardContent>
        </Card>
      </section>
    </div>
  );
}

function MetricCard({
  badge,
  detail,
  icon: Icon,
  label,
  value,
}: {
  badge?: ReactNode;
  detail?: string;
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-sm border border-border/70 bg-background/40 p-3 backdrop-blur-sm">
      <div className="flex items-start justify-between gap-3">
        <Icon className="mt-0.5 h-4 w-4 shrink-0 text-midground" />
        {badge}
      </div>
      <p className="mt-3 text-xs uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p className="mt-1 truncate text-lg font-semibold text-foreground">{value}</p>
      {detail ? <p className="truncate text-xs text-muted-foreground">{detail}</p> : null}
    </div>
  );
}

function ContextAction({
  detail,
  icon: Icon,
  title,
  to,
}: {
  detail: string;
  icon: ComponentType<{ className?: string }>;
  title: string;
  to: string;
}) {
  return (
    <Link
      to={to}
      className="flex w-full items-center justify-between gap-3 rounded-sm border border-border/70 bg-background/30 px-3 py-3 text-left transition-colors hover:border-midground/50 hover:bg-midground/5"
    >
        <span className="flex min-w-0 items-center gap-3">
          <Icon className="h-4 w-4 shrink-0 text-midground" />
          <span className="min-w-0">
            <span className="block text-sm font-medium text-foreground">{title}</span>
            <span className="block truncate text-xs font-normal text-muted-foreground">{detail}</span>
          </span>
        </span>
        <ArrowRight className="h-4 w-4 shrink-0 text-muted-foreground" />
    </Link>
  );
}

function PostureLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-sm border border-border/70 bg-background/30 p-3">
      <p className="text-xs uppercase tracking-[0.14em] text-muted-foreground">{label}</p>
      <p className="mt-1 truncate text-sm text-foreground">{value}</p>
    </div>
  );
}

function EmptyLine({ children }: { children: ReactNode }) {
  return (
    <p className="rounded-sm border border-dashed border-border/70 p-4 text-sm text-muted-foreground">
      {children}
    </p>
  );
}
