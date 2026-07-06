import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Bot,
  Brain,
  CalendarClock,
  CheckCircle2,
  Clock3,
  Command,
  Cpu,
  Gauge,
  MessageCircle,
  Radio,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  TerminalSquare,
  Zap,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { usePageHeader } from "@/contexts/usePageHeader";
import { api } from "@/lib/api";
import type { CronJob, ModelInfoResponse, SessionInfo, StatusResponse, ToolsetInfo } from "@/lib/api";
import {
  buildControlRoomSnapshot,
  formatControlRoomRelativeTime,
  type ControlRoomHealth,
} from "@/lib/control-room";
import { cn, themedBody } from "@/lib/utils";

type LoadState = {
  status: StatusResponse | null;
  sessions: SessionInfo[];
  jobs: CronJob[];
  toolsets: ToolsetInfo[];
  model: ModelInfoResponse | null;
};

const EMPTY_STATE: LoadState = {
  status: null,
  sessions: [],
  jobs: [],
  toolsets: [],
  model: null,
};

const HEALTH_STYLES: Record<ControlRoomHealth, string> = {
  healthy: "border-success/50 bg-success/10 text-success",
  warning: "border-warning/50 bg-warning/10 text-warning",
  critical: "border-destructive/50 bg-destructive/10 text-destructive",
  idle: "border-border bg-muted/20 text-muted-foreground",
};

function HealthBadge({ health }: { health: ControlRoomHealth }) {
  const label = health === "healthy" ? "Live" : health === "warning" ? "Needs eyes" : health === "critical" ? "Attention" : "Idle";
  return (
    <span className={cn("inline-flex items-center gap-1.5 border px-2.5 py-1 text-xs font-medium", HEALTH_STYLES[health])}>
      <span className="h-1.5 w-1.5 rounded-full bg-current" />
      {label}
    </span>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = "neutral",
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  detail: string;
  tone?: "neutral" | "good" | "warn" | "bad";
}) {
  const toneClass =
    tone === "good"
      ? "text-success"
      : tone === "warn"
        ? "text-warning"
        : tone === "bad"
          ? "text-destructive"
          : "text-foreground";
  return (
    <Card className="overflow-hidden border-border/80 bg-card/75">
      <CardContent className="p-4">
        <div className="flex items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-muted-foreground">{label}</p>
            <p className={cn("mt-3 text-3xl font-semibold tabular-nums", toneClass)}>{value}</p>
            <p className="mt-1 text-sm text-muted-foreground">{detail}</p>
          </div>
          <div className="border border-border bg-background/70 p-2">
            <Icon className="h-5 w-5 text-muted-foreground" />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SessionRow({ session }: { session: SessionInfo }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-border/60 py-3 last:border-0">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className={cn("h-2 w-2 rounded-full", session.is_active ? "bg-success" : "bg-muted-foreground/40")} />
          <p className="truncate font-medium">{session.title || "Untitled session"}</p>
        </div>
        <p className="mt-1 line-clamp-1 text-sm text-muted-foreground">{session.preview || session.source || session.id}</p>
      </div>
      <div className="shrink-0 text-right text-xs text-muted-foreground">
        <p>{formatControlRoomRelativeTime(new Date(session.last_active * 1000).toISOString())}</p>
        <p className="mt-1 tabular-nums">{session.tool_call_count} tools</p>
      </div>
    </div>
  );
}

function CronRow({ job }: { job: CronJob }) {
  const failed = Boolean(job.last_error || job.last_delivery_error || job.last_status === "error");
  const paused = !job.enabled || job.state === "paused";
  return (
    <div className="flex items-start justify-between gap-3 border-b border-border/60 py-3 last:border-0">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          {failed ? <AlertTriangle className="h-4 w-4 text-destructive" /> : paused ? <Clock3 className="h-4 w-4 text-warning" /> : <CheckCircle2 className="h-4 w-4 text-success" />}
          <p className="truncate font-medium">{job.name || job.id}</p>
        </div>
        <p className="mt-1 line-clamp-1 text-sm text-muted-foreground">{job.last_error || job.last_delivery_error || job.schedule_display || "Scheduled job"}</p>
      </div>
      <Badge tone={failed ? "destructive" : paused ? "warning" : "success"}>{failed ? "error" : paused ? "paused" : "ready"}</Badge>
    </div>
  );
}

function ControlPanel({ onRestart, restarting }: { onRestart: () => void; restarting: boolean }) {
  return (
    <Card className="border-border/80 bg-card/75">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Command className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-base">Control surface</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="grid gap-3">
        <Button onClick={onRestart} disabled={restarting} className="justify-start gap-2">
          {restarting ? <Spinner className="h-4 w-4" /> : <RefreshCw className="h-4 w-4" />}
          Restart gateway
        </Button>
        <div className="grid grid-cols-2 gap-2 text-xs text-muted-foreground">
          <a className="border border-border bg-background/50 p-3 transition hover:bg-muted/30" href="/cron">Cron jobs</a>
          <a className="border border-border bg-background/50 p-3 transition hover:bg-muted/30" href="/skills">Skills</a>
          <a className="border border-border bg-background/50 p-3 transition hover:bg-muted/30" href="/models">Models</a>
          <a className="border border-border bg-background/50 p-3 transition hover:bg-muted/30" href="/logs">Logs</a>
        </div>
        <p className="text-xs text-muted-foreground">Destructive controls stay behind the dashboard’s existing confirmation and auth layers.</p>
      </CardContent>
    </Card>
  );
}

export default function ControlRoomPage() {
  const { setEnd, setTitle } = usePageHeader();
  const { toast, showToast } = useToast();
  const [state, setState] = useState<LoadState>(EMPTY_STATE);
  const [loading, setLoading] = useState(true);
  const [restarting, setRestarting] = useState(false);

  useLayoutEffect(() => {
    setTitle("Control Room");
    setEnd(null);
    return () => {
      setTitle(null);
      setEnd(null);
    };
  }, [setEnd, setTitle]);

  const load = useCallback(async () => {
    setLoading(true);
    const [status, sessions, jobs, toolsets, model] = await Promise.allSettled([
      api.getStatus(),
      api.getSessions(12, 0, undefined, "recent"),
      api.getCronJobs("all"),
      api.getToolsets(),
      api.getModelInfo(),
    ]);
    setState({
      status: status.status === "fulfilled" ? status.value : null,
      sessions: sessions.status === "fulfilled" ? sessions.value.sessions : [],
      jobs: jobs.status === "fulfilled" ? jobs.value : [],
      toolsets: toolsets.status === "fulfilled" ? toolsets.value : [],
      model: model.status === "fulfilled" ? model.value : null,
    });
    setLoading(false);
    if ([status, sessions, jobs, toolsets, model].some((item) => item.status === "rejected")) {
      showToast("Some control-room data could not be loaded", "error");
    }
  }, [showToast]);

  useEffect(() => {
    load();
    const id = window.setInterval(load, 15000);
    return () => window.clearInterval(id);
  }, [load]);

  const snapshot = useMemo(() => {
    if (!state.status) return null;
    return buildControlRoomSnapshot({
      status: state.status,
      sessions: state.sessions,
      jobs: state.jobs,
      toolsets: state.toolsets,
    });
  }, [state]);

  const restartGateway = useCallback(async () => {
    setRestarting(true);
    try {
      await api.restartGateway();
      showToast("Gateway restart started", "success");
      await load();
    } catch (err) {
      showToast((err as Error).message || "Gateway restart failed", "error");
    } finally {
      setRestarting(false);
    }
  }, [load, showToast]);

  if (loading && !snapshot) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Spinner />
      </div>
    );
  }

  if (!snapshot || !state.status) {
    return (
      <div className={cn("mx-auto max-w-3xl p-6", themedBody)}>
        <Card>
          <CardContent className="p-6">
            <p className="text-lg font-medium">Control room unavailable</p>
            <p className="mt-2 text-sm text-muted-foreground">The dashboard API did not return status data.</p>
            <Button className="mt-4" onClick={load}>Retry</Button>
          </CardContent>
        </Card>
        <Toast toast={toast} />
      </div>
    );
  }

  const modelLine = state.model ? `${state.model.provider} · ${state.model.model}` : "Model status unavailable";

  return (
    <div className={cn("mx-auto grid max-w-7xl gap-5 p-4 sm:p-6", themedBody)}>
      <section className="relative overflow-hidden border border-border bg-card p-5 sm:p-7">
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top_right,color-mix(in_srgb,var(--foreground)_12%,transparent),transparent_36%)]" />
        <div className="relative grid gap-6 lg:grid-cols-[1.35fr_0.65fr] lg:items-end">
          <div>
            <div className="mb-4 flex flex-wrap items-center gap-2">
              <HealthBadge health={snapshot.gateway.health} />
              <span className="inline-flex items-center gap-1.5 border border-border bg-background/60 px-2.5 py-1 text-xs text-muted-foreground">
                <ShieldCheck className="h-3.5 w-3.5" /> oversight mode
              </span>
            </div>
            <p className="text-xs uppercase tracking-[0.28em] text-muted-foreground">Hermes mini-app</p>
            <h1 className="mt-3 flex max-w-3xl flex-wrap gap-x-3 text-4xl font-semibold tracking-tight sm:text-5xl">
              <span>Control</span>
              <span>Room</span>
            </h1>
            <p className="mt-4 max-w-2xl text-base text-muted-foreground sm:text-lg">
              A mobile-first cockpit for watching Hermes work, checking channel health, and taking safe operational control without losing project context.
            </p>
          </div>
          <div className="border border-border bg-background/70 p-4">
            <div className="flex items-center gap-2 text-sm font-medium">
              <Cpu className="h-4 w-4 text-muted-foreground" />
              Current brain
            </div>
            <p className="mt-3 break-words text-sm text-muted-foreground">{modelLine}</p>
            <p className="mt-3 text-xs text-muted-foreground">Hermes {state.status.version} · config v{state.status.config_version}</p>
          </div>
        </div>
      </section>

      <section className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={Radio} label="Gateway" value={snapshot.gateway.primaryLine} detail={`Updated ${snapshot.gateway.updatedAgo}`} tone={snapshot.gateway.health === "healthy" ? "good" : snapshot.gateway.health === "warning" ? "warn" : "bad"} />
        <MetricCard icon={MessageCircle} label="Sessions" value={`${snapshot.sessions.active}`} detail={`${snapshot.sessions.totalRecent} recent conversations`} tone="neutral" />
        <MetricCard icon={CalendarClock} label="Cron" value={`${snapshot.cron.enabled}/${snapshot.cron.total}`} detail={`${snapshot.cron.failed} failed · ${snapshot.cron.paused} paused`} tone={snapshot.cron.failed ? "bad" : snapshot.cron.paused ? "warn" : "good"} />
        <MetricCard icon={TerminalSquare} label="Tools" value={`${snapshot.tools.enabled}`} detail={`${snapshot.tools.configured}/${snapshot.tools.total} configured`} tone="neutral" />
      </section>

      <section className="grid gap-5 lg:grid-cols-[1fr_1fr_0.8fr]">
        <Card className="border-border/80 bg-card/75">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-base">Recent work</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            {snapshot.sessions.recent.length ? snapshot.sessions.recent.map((session) => <SessionRow key={session.id} session={session} />) : <p className="text-sm text-muted-foreground">No recent sessions yet.</p>}
          </CardContent>
        </Card>

        <Card className="border-border/80 bg-card/75">
          <CardHeader>
            <div className="flex items-center gap-2">
              <Gauge className="h-5 w-5 text-muted-foreground" />
              <CardTitle className="text-base">Operational watchlist</CardTitle>
            </div>
          </CardHeader>
          <CardContent>
            {snapshot.cron.priorityJobs.length ? snapshot.cron.priorityJobs.map((job) => <CronRow key={`${job.profile_name || "default"}:${job.id}`} job={job} />) : <p className="text-sm text-muted-foreground">No scheduled work configured.</p>}
          </CardContent>
        </Card>

        <div className="grid gap-5">
          <ControlPanel onRestart={restartGateway} restarting={restarting} />
          <Card className="border-border/80 bg-card/75">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-muted-foreground" />
                <CardTitle className="text-base">Mini-app next slice</CardTitle>
              </div>
            </CardHeader>
            <CardContent className="space-y-3 text-sm text-muted-foreground">
              <p className="flex gap-2"><Zap className="mt-0.5 h-4 w-4 shrink-0 text-warning" /> Photon launch card gated behind PHOTON_MINI_APPS.</p>
              <p className="flex gap-2"><Brain className="mt-0.5 h-4 w-4 shrink-0" /> Live run stream and approval queue once the backend exposes a narrow read API.</p>
              <p className="flex gap-2"><Bot className="mt-0.5 h-4 w-4 shrink-0" /> Project context panel for the business channel.</p>
            </CardContent>
          </Card>
        </div>
      </section>
      <Toast toast={toast} />
    </div>
  );
}
