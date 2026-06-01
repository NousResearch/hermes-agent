import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { ExternalLink, Gauge, GitPullRequest, PlayCircle, Radar, RefreshCw, ShieldAlert, Users, Workflow } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { api } from "@/lib/api";
import type {
  KanbanActiveWorkersResponse,
  KanbanBoardResponse,
  KanbanDiagnosticsResponse,
  KanbanStatsResponse,
  KanbanTask,
} from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";

const OPEN_STATUSES = ["triage", "todo", "scheduled", "ready", "running", "blocked", "review"];
const FOCUS_STATUSES = ["blocked", "running", "ready", "review", "scheduled"];

type LoadState = "loading" | "ready" | "error";

interface TaskMissionData {
  stats: KanbanStatsResponse | null;
  board: KanbanBoardResponse | null;
  diagnostics: KanbanDiagnosticsResponse | null;
  workers: KanbanActiveWorkersResponse | null;
}

const emptyData: TaskMissionData = {
  stats: null,
  board: null,
  diagnostics: null,
  workers: null,
};

function fmtAge(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${Math.max(0, Math.floor(seconds))}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  return `${Math.floor(seconds / 86400)}d`;
}

function StatTile({ label, value, hint, tone = "neutral" }: { label: string; value: string | number; hint?: string; tone?: "neutral" | "good" | "warn" }) {
  const toneClass = tone === "good" ? "text-success" : tone === "warn" ? "text-warning" : "text-primary";
  return (
    <Card className="border-primary/15 bg-surface/65 shadow-[inset_0_1px_0_rgba(255,230,203,0.06)]">
      <CardContent className="p-4">
        <div className="font-mono text-[11px] uppercase tracking-[0.2em] text-text-tertiary">{label}</div>
        <div className={`mt-2 text-3xl font-semibold ${toneClass}`}>{value}</div>
        {hint ? <div className="mt-1 text-xs text-text-secondary">{hint}</div> : null}
      </CardContent>
    </Card>
  );
}

function TaskRow({ task }: { task: KanbanTask }) {
  const summary = task.latest_summary || task.body || "No summary yet.";
  return (
    <div className="rounded-lg border border-primary/10 bg-background/35 p-3 transition hover:border-primary/35">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <div className="font-medium text-text-primary">{task.title}</div>
          <div className="mt-1 font-mono text-[11px] uppercase tracking-[0.12em] text-text-tertiary">{task.id}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge tone="outline">{task.status}</Badge>
          {task.assignee ? <Badge tone="secondary">{task.assignee}</Badge> : null}
          {task.priority ? <Badge tone="outline">P{task.priority}</Badge> : null}
        </div>
      </div>
      <p className="mt-2 line-clamp-2 text-sm text-text-secondary">{summary}</p>
      <div className="mt-3 flex flex-wrap gap-3 text-xs text-text-tertiary">
        <span>age {fmtAge(task.age?.created_age_seconds)}</span>
        {task.comment_count ? <span>{task.comment_count} comments</span> : null}
        {task.progress ? <span>{task.progress.done}/{task.progress.total} children done</span> : null}
        {task.diagnostics?.length ? <span className="text-warning">{task.diagnostics.length} diagnostics</span> : null}
      </div>
    </div>
  );
}

function Panel({ title, icon: Icon, children }: { title: string; icon: typeof Radar; children: React.ReactNode }) {
  return (
    <Card className="border-primary/15 bg-surface/70">
      <CardHeader className="flex flex-row items-center gap-2 pb-2">
        <Icon className="h-4 w-4 text-primary" />
        <CardTitle className="font-mono text-xs uppercase tracking-[0.18em] text-text-secondary">{title}</CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

export default function TasksPage() {
  const { setTitle, setAfterTitle } = usePageHeader();
  const [data, setData] = useState<TaskMissionData>(emptyData);
  const [state, setState] = useState<LoadState>("loading");
  const [error, setError] = useState<string | null>(null);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  useEffect(() => {
    setTitle("Tasks Mission");
    setAfterTitle(<span className="text-xs text-text-tertiary">Kanban command view — live board, workers, diagnostics, read-only controls.</span>);
    return () => {
      setTitle(null);
      setAfterTitle(null);
    };
  }, [setTitle, setAfterTitle]);

  async function load() {
    setState((prev) => (prev === "ready" ? "ready" : "loading"));
    setError(null);
    try {
      const [stats, board, diagnostics, workers] = await Promise.all([
        api.getKanbanStats().catch(() => null),
        api.getKanbanBoard().catch(() => null),
        api.getKanbanDiagnostics().catch(() => null),
        api.getKanbanActiveWorkers().catch(() => null),
      ]);
      setData({ stats, board, diagnostics, workers });
      setUpdatedAt(new Date());
      setState("ready");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setState("error");
    }
  }

  useEffect(() => {
    load();
    const id = window.setInterval(load, 10000);
    return () => window.clearInterval(id);
  }, []);

  const byStatus = data.stats?.by_status ?? {};
  const total = Object.values(byStatus).reduce((sum, value) => sum + value, 0);
  const open = OPEN_STATUSES.reduce((sum, key) => sum + (byStatus[key] ?? 0), 0);
  const blocked = byStatus.blocked ?? 0;
  const running = byStatus.running ?? 0;
  const ready = byStatus.ready ?? 0;
  const diagnosticsCount = data.diagnostics?.count ?? 0;
  const workerCount = data.workers?.count ?? 0;
  const allTasks = useMemo(() => data.board?.columns.flatMap((column) => column.tasks) ?? [], [data.board]);
  const focusTasks = useMemo(
    () => allTasks.filter((task) => FOCUS_STATUSES.includes(task.status)).slice(0, 12),
    [allTasks],
  );
  const assignees = data.stats ? Object.keys(data.stats.by_assignee ?? {}) : [];

  return (
    <div className="relative space-y-6 overflow-hidden rounded-xl border border-primary/10 bg-[radial-gradient(circle_at_top_left,color-mix(in_srgb,var(--midground-base)_10%,transparent),transparent_30%),linear-gradient(rgba(255,230,203,0.035)_1px,transparent_1px),linear-gradient(90deg,rgba(255,230,203,0.035)_1px,transparent_1px)] bg-[size:auto,28px_28px,28px_28px] p-1">
      <div className="space-y-6 rounded-lg bg-background/40 p-3 md:p-5">
        <div className="relative overflow-hidden rounded-xl border border-primary/20 bg-[linear-gradient(135deg,rgba(255,230,203,0.09),rgba(4,28,28,0.78)_45%,rgba(74,222,128,0.06))] p-5 shadow-[0_24px_90px_rgba(0,0,0,0.35)]">
          <div className="absolute right-6 top-6 hidden h-28 w-28 rounded-full border border-primary/20 md:block" />
          <div className="absolute right-12 top-12 hidden h-16 w-16 rounded-full border border-success/20 md:block" />
          <div className="relative flex flex-wrap items-start justify-between gap-4">
            <div>
              <div className="mb-3 flex flex-wrap gap-2 font-mono text-[11px] uppercase tracking-[0.22em] text-text-tertiary">
                <span className="rounded-full border border-primary/20 bg-background/40 px-3 py-1">Kanban Ops</span>
                <span className="rounded-full border border-success/30 bg-success/10 px-3 py-1 text-success">Real Board Data</span>
                <span className="rounded-full border border-warning/30 bg-warning/10 px-3 py-1 text-warning">Read-only</span>
              </div>
              <h2 className="text-3xl font-semibold tracking-tight text-text-primary">Tasks Mission Control</h2>
              <p className="mt-2 max-w-2xl text-sm text-text-secondary">
                Live cockpit for the Hermes Kanban work queue: lane pressure, active workers, diagnostics, and high-attention cards.
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Link className="inline-flex items-center gap-2 rounded-md border border-primary/20 bg-background/45 px-3 py-2 text-sm text-primary hover:bg-primary/10" to="/kanban">
                <ExternalLink className="h-4 w-4" /> Open full Kanban
              </Link>
              <Button size="sm" onClick={load} disabled={state === "loading"}>
                <RefreshCw className={`mr-2 h-4 w-4 ${state === "loading" ? "animate-spin" : ""}`} /> Refresh
              </Button>
            </div>
          </div>
        </div>

        {error ? <div className="rounded-md border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">{error}</div> : null}

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-6">
          <StatTile label="Open" value={data.stats ? open : "Unavailable"} hint={`${total} non-archived total`} tone={open ? "good" : "neutral"} />
          <StatTile label="Running" value={data.stats ? running : "Unavailable"} hint={`${workerCount} active workers`} tone={running ? "good" : "neutral"} />
          <StatTile label="Ready" value={data.stats ? ready : "Unavailable"} hint={`oldest ${fmtAge(data.stats?.oldest_ready_age_seconds)}`} tone={ready ? "good" : "neutral"} />
          <StatTile label="Blocked" value={data.stats ? blocked : "Unavailable"} hint="needs operator attention" tone={blocked ? "warn" : "neutral"} />
          <StatTile label="Diagnostics" value={data.diagnostics ? diagnosticsCount : "Unavailable"} hint="active warnings/errors" tone={diagnosticsCount ? "warn" : "neutral"} />
          <StatTile label="Assignees" value={data.stats ? assignees.length : "Unavailable"} hint="profiles with tasks" />
        </div>

        <div className="grid gap-4 xl:grid-cols-[1.35fr_0.65fr]">
          <Panel title="Lane pressure" icon={Gauge}>
            {data.stats ? (
              <div className="space-y-3">
                {OPEN_STATUSES.map((status) => {
                  const count = byStatus[status] ?? 0;
                  const pct = open ? Math.round((count / open) * 100) : 0;
                  return (
                    <div key={status}>
                      <div className="mb-1 flex justify-between text-xs text-text-secondary"><span>{status}</span><span>{count}</span></div>
                      <div className="h-2 overflow-hidden rounded-full bg-background/60">
                        <div className="h-full rounded-full bg-primary/70" style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="rounded border border-warning/20 bg-warning/5 p-3 text-sm text-text-secondary">Kanban stats unavailable. Plugin API may not be mounted.</div>
            )}
          </Panel>

          <Panel title="Active workers" icon={PlayCircle}>
            {data.workers?.workers.length ? (
              <div className="space-y-3">
                {data.workers.workers.map((worker) => (
                  <div key={worker.run_id} className="rounded border border-primary/10 bg-background/35 p-3 text-sm">
                    <div className="font-medium text-text-primary">{worker.task_title}</div>
                    <div className="mt-1 text-xs text-text-tertiary">PID {worker.worker_pid ?? "—"} · {worker.profile ?? "default"}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-text-secondary">No active workers right now.</div>
            )}
          </Panel>
        </div>

        <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
          <Panel title="Attention queue" icon={ShieldAlert}>
            {focusTasks.length ? (
              <div className="space-y-3">
                {focusTasks.map((task) => <TaskRow key={task.id} task={task} />)}
              </div>
            ) : (
              <div className="rounded border border-primary/10 bg-background/35 p-4 text-sm text-text-secondary">No blocked/running/ready/review/scheduled tasks found.</div>
            )}
          </Panel>

          <div className="space-y-4">
            <Panel title="Diagnostics" icon={GitPullRequest}>
              {data.diagnostics?.diagnostics.length ? (
                <div className="space-y-3">
                  {data.diagnostics.diagnostics.slice(0, 8).map((item) => (
                    <div key={item.task_id} className="rounded border border-warning/20 bg-warning/5 p-3 text-sm">
                      <div className="font-medium text-text-primary">{item.task_title ?? item.task_id}</div>
                      <div className="mt-1 text-xs text-text-secondary">{item.task_status ?? "unknown"} · {item.task_assignee ?? "unassigned"} · {item.diagnostics.length} diagnostics</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-sm text-text-secondary">No active diagnostics.</div>
              )}
            </Panel>

            <Panel title="Assignee loadout" icon={Users}>
              {assignees.length ? (
                <div className="space-y-2">
                  {assignees.slice(0, 10).map((assignee) => {
                    const counts = data.stats?.by_assignee[assignee] ?? {};
                    const assignedTotal = Object.values(counts).reduce((sum, value) => sum + value, 0);
                    return (
                      <div key={assignee} className="flex items-center justify-between rounded border border-primary/10 bg-background/35 px-3 py-2 text-sm">
                        <span className="text-text-primary">{assignee}</span>
                        <span className="text-text-tertiary">{assignedTotal} tasks</span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-sm text-text-secondary">No assignee load yet.</div>
              )}
            </Panel>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-primary/10 bg-surface/50 p-4 text-xs text-text-tertiary">
          <div className="flex items-center gap-2"><Workflow className="h-4 w-4 text-primary" /> Full write controls stay inside the Kanban plugin page; this surface is intentionally read-only.</div>
          <div>Updated: {updatedAt ? updatedAt.toLocaleTimeString() : "—"}</div>
        </div>
      </div>
    </div>
  );
}
