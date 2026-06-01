import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  BarChart3,
  Clock,
  Cpu,
  FileText,
  GitBranch,
  KeyRound,
  MessageSquare,
  Package,
  Puzzle,
  Radar,
  RefreshCw,
  Search,
  Settings,
  Shield,
  Terminal,
  Users,
  Zap,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { api } from "@/lib/api";
import type {
  AnalyticsResponse,
  CronJob,
  EnvVarInfo,
  ModelInfoResponse,
  PaginatedSessions,
  ProfileInfo,
  SkillInfo,
  StatusResponse,
  SystemHealthResponse,
  KanbanStatsResponse,
} from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { usePageHeader } from "@/contexts/usePageHeader";

type LoadState = "loading" | "ready" | "error";

interface OverviewData {
  status: StatusResponse | null;
  system: SystemHealthResponse | null;
  kanban: KanbanStatsResponse | null;
  sessions: PaginatedSessions | null;
  analytics: AnalyticsResponse | null;
  model: ModelInfoResponse | null;
  cron: CronJob[] | null;
  skills: SkillInfo[] | null;
  profiles: ProfileInfo[] | null;
  env: Record<string, EnvVarInfo> | null;
  logs: string[];
}

const emptyData: OverviewData = {
  status: null,
  system: null,
  kanban: null,
  sessions: null,
  analytics: null,
  model: null,
  cron: null,
  skills: null,
  profiles: null,
  env: null,
  logs: [],
};

function fmtNum(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "Unavailable";
  return new Intl.NumberFormat().format(value);
}

function fmtCost(value: number | null | undefined): string {
  if (!value) return "Unavailable";
  return `$${value.toFixed(4)}`;
}

function StatusDot({ ok, warn = false }: { ok: boolean; warn?: boolean }) {
  const color = ok ? "bg-success shadow-success/40" : warn ? "bg-warning shadow-warning/40" : "bg-destructive shadow-destructive/40";
  return <span className={`inline-block h-2.5 w-2.5 rounded-full shadow-[0_0_14px] ${color}`} />;
}

function StatCard({
  label,
  value,
  hint,
  icon: Icon,
  tone = "neutral",
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon: typeof Activity;
  tone?: "neutral" | "good" | "warn";
}) {
  const color = tone === "good" ? "text-success" : tone === "warn" ? "text-warning" : "text-primary";
  return (
    <Card className="group overflow-hidden border-primary/15 bg-[linear-gradient(145deg,color-mix(in_srgb,var(--midground-base)_8%,transparent),color-mix(in_srgb,var(--background-base)_88%,transparent))] shadow-[0_0_0_1px_rgba(255,230,203,0.03),0_18px_60px_rgba(0,0,0,0.22)] backdrop-blur">
      <CardContent className="relative p-4">
        <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
        <div className="flex items-start justify-between gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.22em] text-text-tertiary">{label}</div>
            <div className="mt-2 text-2xl font-semibold text-text-primary">{value}</div>
            {hint ? <div className="mt-1 text-xs text-text-secondary">{hint}</div> : null}
          </div>
          <div className="rounded border border-primary/20 bg-background/40 p-2">
            <Icon className={`h-5 w-5 ${color}`} />
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

function SectionLink({ to, label, children, className = "" }: { to: string; label: string; children: React.ReactNode; className?: string }) {
  return (
    <Card className={`border-primary/15 bg-surface/70 shadow-[inset_0_1px_0_rgba(255,230,203,0.06)] ${className}`}>
      <CardHeader className="flex flex-row items-center justify-between gap-3 pb-2">
        <CardTitle className="font-mono text-xs uppercase tracking-[0.18em] text-text-secondary">{label}</CardTitle>
        <Link className="font-mono text-[11px] uppercase tracking-[0.14em] text-primary hover:underline" to={to}>Open</Link>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function QuickLink({ to, icon: Icon, title, subtitle }: { to: string; icon: typeof Activity; title: string; subtitle: string }) {
  return (
    <Link className="group rounded-lg border border-primary/15 bg-surface/55 p-4 text-sm transition hover:border-primary/50 hover:bg-primary/5" to={to}>
      <Icon className="mb-3 h-4 w-4 text-primary transition group-hover:scale-110" />
      <div className="font-medium text-text-primary">{title}</div>
      <div className="mt-1 text-xs text-text-tertiary">{subtitle}</div>
    </Link>
  );
}

export default function OverviewPage() {
  const { setTitle, setAfterTitle } = usePageHeader();
  const [data, setData] = useState<OverviewData>(emptyData);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  useEffect(() => {
    setTitle("Mission Control");
    setAfterTitle(<span className="text-xs text-text-tertiary">Live local Hermes observability — real data, read-only cockpit.</span>);
    return () => {
      setTitle(null);
      setAfterTitle(null);
    };
  }, [setTitle, setAfterTitle]);

  async function load() {
    setState((prev) => (prev === "ready" ? "ready" : "loading"));
    setError(null);
    try {
      const [status, system, kanban, sessions, analytics, model, cron, skills, profiles, env, logs] = await Promise.all([
        api.getStatus().catch(() => null),
        api.getSystemHealth().catch(() => null),
        api.getKanbanStats().catch(() => null),
        api.getSessions(8, 0).catch(() => null),
        api.getAnalytics(30).catch(() => null),
        api.getModelInfo().catch(() => null),
        api.getCronJobs("all").catch(() => null),
        api.getSkills().catch(() => null),
        api.getProfiles().catch(() => null),
        api.getEnvVars().catch(() => null),
        api.getLogs({ file: "errors", lines: 8 }).then((r) => r.lines).catch(() => []),
      ]);
      setData({ status, system, kanban, sessions, analytics, model, cron, skills, profiles: profiles?.profiles ?? null, env, logs });
      setUpdatedAt(new Date());
      setState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }

  useEffect(() => {
    load();
    const id = window.setInterval(load, 30000);
    return () => window.clearInterval(id);
  }, []);

  const envSummary = useMemo(() => {
    const values = Object.values(data.env ?? {});
    return { total: values.length, set: values.filter((v) => v.is_set).length, missing: values.filter((v) => !v.is_set).length };
  }, [data.env]);

  const lastSession = data.sessions?.sessions?.[0];
  const enabledCron = data.cron?.filter((j) => j.enabled).length ?? null;
  const gatewayOk = Boolean(data.status?.gateway_running);
  const dirtyGit = Boolean(data.system?.git.dirty);
  const activeCron = enabledCron ?? 0;
  const kanbanStatus = data.kanban?.by_status ?? null;
  const kanbanTotal = kanbanStatus ? Object.values(kanbanStatus).reduce((sum, value) => sum + value, 0) : null;
  const kanbanOpen = kanbanStatus
    ? ["triage", "todo", "scheduled", "ready", "running", "blocked", "review"].reduce((sum, key) => sum + (kanbanStatus[key] ?? 0), 0)
    : null;
  const kanbanBlocked = kanbanStatus?.blocked ?? 0;
  const kanbanRunning = kanbanStatus?.running ?? 0;
  const assigneeCount = data.kanban ? Object.keys(data.kanban.by_assignee ?? {}).length : null;

  return (
    <div className="relative space-y-6 overflow-hidden rounded-xl border border-primary/10 bg-[radial-gradient(circle_at_top_left,color-mix(in_srgb,var(--midground-base)_10%,transparent),transparent_30%),linear-gradient(rgba(255,230,203,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,230,203,0.035)_1px,transparent_1px)] bg-[size:auto,28px_28px,28px_28px] p-1">
      <div className="space-y-6 rounded-lg bg-background/40 p-3 md:p-5">
        <div className="relative overflow-hidden rounded-xl border border-primary/20 bg-[linear-gradient(135deg,rgba(255,230,203,0.10),rgba(4,28,28,0.76)_42%,rgba(74,222,128,0.06))] p-5 shadow-[0_24px_90px_rgba(0,0,0,0.35)]">
          <div className="absolute right-6 top-6 hidden h-28 w-28 rounded-full border border-primary/20 md:block" />
          <div className="absolute right-10 top-10 hidden h-20 w-20 rounded-full border border-primary/10 md:block" />
          <div className="relative flex flex-wrap items-start justify-between gap-5">
            <div className="max-w-3xl">
              <div className="mb-3 flex flex-wrap items-center gap-2 font-mono text-[11px] uppercase tracking-[0.22em] text-text-tertiary">
                <span className="rounded-full border border-primary/20 bg-background/40 px-3 py-1">Localhost Control Plane</span>
                <span className="rounded-full border border-success/30 bg-success/10 px-3 py-1 text-success">Real Data</span>
                <span className="rounded-full border border-primary/20 bg-background/40 px-3 py-1">Read-only Overview</span>
              </div>
              <div className="flex items-center gap-3">
                <Radar className="h-8 w-8 text-primary" />
                <h2 className="text-3xl font-semibold tracking-tight text-text-primary md:text-5xl">Hermes Mission Control</h2>
              </div>
              <p className="mt-3 max-w-2xl text-sm text-text-secondary md:text-base">
                Tactical local cockpit for your active Hermes setup. No fake metrics, no secrets displayed, and missing integrations are labelled instead of invented. Proper little command deck, this.
              </p>
            </div>
            <div className="min-w-56 rounded-lg border border-primary/15 bg-background/45 p-4 font-mono text-xs text-text-secondary">
              <div className="mb-3 flex items-center justify-between text-text-primary"><span>LIVE STATUS</span><StatusDot ok={gatewayOk} warn={!gatewayOk} /></div>
              <div className="space-y-2">
                <div className="flex justify-between gap-4"><span>Gateway</span><span>{data.status?.gateway_state ?? (gatewayOk ? "running" : "stopped")}</span></div>
                <div className="flex justify-between gap-4"><span>Sessions</span><span>{fmtNum(data.sessions?.total)}</span></div>
                <div className="flex justify-between gap-4"><span>Cron</span><span>{data.cron ? `${activeCron}/${data.cron.length}` : "Unavailable"}</span></div>
                <div className="flex justify-between gap-4"><span>Tasks</span><span>{kanbanOpen === null ? "Unavailable" : `${kanbanOpen} open`}</span></div>
                <div className="flex justify-between gap-4"><span>Updated</span><span>{updatedAt ? updatedAt.toLocaleTimeString() : "—"}</span></div>
              </div>
              <Button className="mt-4 w-full" size="sm" onClick={load} disabled={state === "loading"}>
                <RefreshCw className="mr-2 h-4 w-4" /> Refresh Scan
              </Button>
            </div>
          </div>
        </div>

        {error ? <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{error}</div> : null}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
          <StatCard label="Gateway" value={data.status?.gateway_state ?? (gatewayOk ? "running" : "stopped")} hint={data.status?.gateway_pid ? `PID ${data.status.gateway_pid}` : "Local/runtime status"} icon={Shield} tone={gatewayOk ? "good" : "warn"} />
          <StatCard label="Sessions" value={fmtNum(data.sessions?.total)} hint={lastSession ? `Latest ${timeAgo(lastSession.last_active || lastSession.started_at)}` : "No recent session found"} icon={MessageSquare} />
          <StatCard label="Tasks" value={kanbanOpen === null ? "Unavailable" : kanbanOpen} hint={kanbanTotal === null ? "Kanban plugin not wired" : `${kanbanRunning} running · ${kanbanBlocked} blocked · ${kanbanTotal} total`} icon={Puzzle} tone={kanbanBlocked > 0 ? "warn" : kanbanOpen ? "good" : "neutral"} />
          <StatCard label="Cron jobs" value={data.cron ? `${enabledCron}/${data.cron.length}` : "Unavailable"} hint="enabled / total across profiles" icon={Clock} tone={activeCron > 0 ? "good" : "neutral"} />
          <StatCard label="Secrets" value={data.env ? `${envSummary.set}/${envSummary.total}` : "Unavailable"} hint={data.env ? `${envSummary.missing} missing, values redacted` : "Presence only"} icon={KeyRound} tone={envSummary.missing ? "warn" : "good"} />
        </div>

        <div className="grid gap-4 xl:grid-cols-3">
          <SectionLink to="/models" label="Model uplink">
            <div className="space-y-3 text-sm">
              <div className="flex items-center gap-2 text-text-primary"><Cpu className="h-4 w-4 text-primary" /> {data.model?.model || "Not configured"}</div>
              <div className="grid grid-cols-2 gap-2 text-xs text-text-secondary">
                <div className="rounded border border-border/50 bg-background/30 p-2">Provider<br /><span className="text-text-primary">{data.model?.provider || "Unavailable"}</span></div>
                <div className="rounded border border-border/50 bg-background/30 p-2">Context<br /><span className="text-text-primary">{data.model?.effective_context_length ? fmtNum(data.model.effective_context_length) : "Unavailable"}</span></div>
              </div>
            </div>
          </SectionLink>

          <SectionLink to="/analytics" label="30-day telemetry">
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-primary" /> Sessions: {fmtNum(data.analytics?.totals.total_sessions)}</div>
              <div>API calls: {fmtNum(data.analytics?.totals.total_api_calls)}</div>
              <div>Estimated cost: {fmtCost(data.analytics?.totals.total_estimated_cost)}</div>
            </div>
          </SectionLink>

          <SectionLink to="/system" label="System health">
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2"><Terminal className="h-4 w-4 text-primary" /> Python {data.system?.runtime.python ?? "Unavailable"}</div>
              <div className="flex items-center gap-2"><GitBranch className="h-4 w-4 text-primary" /> {data.system?.git.branch ?? "git unavailable"} · {data.system?.git.commit ?? "—"}</div>
              <div className="flex items-center gap-2 text-text-secondary"><StatusDot ok={!dirtyGit} warn={dirtyGit} /> {dirtyGit ? `${data.system?.git.dirty_count ?? 0} changed files` : "git clean"}</div>
            </div>
          </SectionLink>

          <SectionLink to="/kanban" label="Kanban lanes">
            <div className="space-y-3 text-sm">
              {data.kanban ? (
                <>
                  <div className="grid grid-cols-3 gap-2 text-center font-mono text-xs">
                    <div className="rounded border border-border/50 bg-background/30 p-2"><div className="text-lg text-text-primary">{kanbanOpen}</div><div className="text-text-tertiary">open</div></div>
                    <div className="rounded border border-border/50 bg-background/30 p-2"><div className="text-lg text-text-primary">{kanbanRunning}</div><div className="text-text-tertiary">running</div></div>
                    <div className="rounded border border-border/50 bg-background/30 p-2"><div className="text-lg text-text-primary">{kanbanBlocked}</div><div className="text-text-tertiary">blocked</div></div>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs text-text-secondary">
                    {Object.entries(kanbanStatus ?? {}).slice(0, 8).map(([status, count]) => (
                      <span key={status} className="rounded-full border border-primary/15 bg-background/30 px-2 py-1">{status}: {count}</span>
                    ))}
                  </div>
                  <div className="text-xs text-text-tertiary">Assignees with tasks: {assigneeCount ?? 0}</div>
                </>
              ) : (
                <div className="rounded border border-warning/20 bg-warning/5 p-3 text-sm text-text-secondary">Kanban plugin stats unavailable. Feature not yet wired or plugin route not mounted.</div>
              )}
            </div>
          </SectionLink>
        </div>

        <div className="grid gap-4 xl:grid-cols-[1.25fr_0.75fr]">
          <SectionLink to="/sessions" label="Mission feed — recent sessions">
            <div className="space-y-3">
              {(data.sessions?.sessions ?? []).slice(0, 6).map((s) => (
                <Link to={`/sessions?session=${encodeURIComponent(s.id)}`} key={s.id} className="block rounded border border-border/60 bg-background/30 p-3 transition hover:border-primary/40 hover:bg-primary/5">
                  <div className="flex items-center justify-between gap-3">
                    <div className="truncate text-sm font-medium text-text-primary">{s.title || s.id}</div>
                    <Badge tone={s.is_active ? "success" : "secondary"}>{s.source || "unknown"}</Badge>
                  </div>
                  <div className="mt-1 truncate text-xs text-text-secondary">{s.preview || "No preview"}</div>
                  <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 font-mono text-[11px] uppercase tracking-[0.08em] text-text-tertiary">
                    <span>{s.message_count} msgs</span><span>{s.tool_call_count} tools</span><span>{s.model || "model n/a"}</span><span>{timeAgo(s.last_active || s.started_at)}</span>
                  </div>
                </Link>
              ))}
              {data.sessions && data.sessions.sessions.length === 0 ? <div className="text-sm text-text-secondary">No sessions found.</div> : null}
            </div>
          </SectionLink>

          <div className="space-y-4">
            <SectionLink to="/skills" label="Crew loadout">
              <div className="space-y-2 text-sm">
                <div className="flex items-center gap-2"><Package className="h-4 w-4 text-primary" /> Skills: {data.skills ? fmtNum(data.skills.length) : "Unavailable"}</div>
                <div className="flex items-center gap-2"><Users className="h-4 w-4 text-primary" /> Profiles: {data.profiles ? fmtNum(data.profiles.length) : "Unavailable"}</div>
                <div className="text-text-secondary">Default: {data.profiles?.find((p) => p.is_default)?.name ?? "default"}</div>
              </div>
            </SectionLink>

            <SectionLink to="/logs" label="Fault monitor">
              <div className="space-y-2">
                {data.logs.length ? data.logs.slice(-4).map((line, idx) => (
                  <pre key={idx} className="overflow-hidden text-ellipsis rounded border border-warning/20 bg-warning/5 p-2 text-xs text-text-secondary">{line}</pre>
                )) : <div className="text-sm text-text-secondary">No recent error lines found.</div>}
              </div>
            </SectionLink>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
          <QuickLink to="/sessions" icon={Search} title="Search sessions" subtitle="Inspect real session store" />
          <QuickLink to="/config" icon={Settings} title="Config" subtitle="Sanitized config view" />
          <QuickLink to="/env" icon={KeyRound} title="Keys" subtitle="Presence only, redacted" />
          <QuickLink to="/plugins" icon={Puzzle} title="Plugins / Kanban" subtitle="Installed dashboard tabs" />
          <QuickLink to="/docs" icon={FileText} title="Docs" subtitle="Command references" />
        </div>

        <div className="rounded-lg border border-primary/15 bg-background/35 p-3 font-mono text-[11px] uppercase tracking-[0.16em] text-text-tertiary">
          <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
            <span className="flex items-center gap-2"><Zap className="h-3.5 w-3.5 text-primary" /> Local-first dashboard</span>
            <span>Hermes home: {data.status?.hermes_home ?? data.system?.hermes.home ?? "Unavailable"}</span>
            <span>Config: {data.status?.config_path ?? data.system?.hermes.config_path ?? "Unavailable"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
