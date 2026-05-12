import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Archive,
  ArrowRight,
  Brain,
  Building2,
  CheckCircle2,
  Circle,
  Clock3,
  FileText,
  GitBranch,
  Inbox,
  LayoutGrid,
  ListTree,
  Loader2,
  MessageSquare,
  MessageSquarePlus,
  Network,
  RefreshCw,
  Table2,
  RotateCcw,
  Search,
  Send,
  Shield,
  Sparkles,
  UserRound,
  Users,
  Wifi,
  WifiOff,
  Wrench,
  X,
  Zap,
} from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { usePageHeader } from "@/contexts/usePageHeader";
import { fetchJSON } from "@/lib/api";
import { cn } from "@/lib/utils";

type OfficeProfileName =
  | "triage"
  | "supervisor"
  | "chief"
  | "pm"
  | "architect"
  | "research"
  | "coder"
  | "reviewer"
  | "qa"
  | "security"
  | "memory"
  | "tooling"
  | "approval"
  | "observability"
  | "devops"
  | "docs"
  | "demo";

type TaskDiagnostic = { severity?: string; message?: string; kind?: string; last_seen_at?: number; count?: number };
type TaskWarningSummary = { count?: number; highest_severity?: string; kinds?: Record<string, number>; latest_at?: number };

type KanbanTask = {
  id: string;
  title: string;
  body?: string | null;
  status: string;
  assignee?: string | null;
  priority?: number | null;
  created_at?: number | null;
  started_at?: number | null;
  completed_at?: number | null;
  latest_summary?: string | null;
  office_role?: string | null;
  comment_count?: number;
  warnings?: TaskWarningSummary | null;
  age?: { created_age_seconds?: number | null; started_age_seconds?: number | null; time_to_complete_seconds?: number | null };
  link_counts?: { parents: number; children: number };
  progress?: { done: number; total: number } | null;
  diagnostics?: TaskDiagnostic[];
};

type BoardColumn = { name: string; tasks: KanbanTask[] };
type BoardLink = { parent_id: string; child_id: string };
type BoardMeta = { slug: string; name?: string | null; description?: string | null; is_current?: boolean; counts?: Record<string, number>; total?: number };
type BoardListResponse = { boards: BoardMeta[]; current: string };
type LiveEvent = { id: number; task_id?: string | null; kind: string; created_at?: number | null; payload?: unknown };
type OfficeStatus = {
  enabled: boolean;
  board: string;
  preferred_board: string;
  board_exists: boolean;
  gateway_running: boolean;
  gateway_pid?: number | null;
  profiles?: {
    expected: number;
    present: string[];
    missing: string[];
  };
};

type BoardResponse = {
  columns: BoardColumn[];
  assignees: string[];
  links?: BoardLink[];
  latest_event_id: number;
  now: number;
  office?: OfficeStatus;
};

type TaskComment = { id: number; author?: string | null; body: string; created_at?: number | null };
type TaskEvent = { id: number; kind: string; office_kind?: string; created_at?: number | null; payload?: unknown };
type TaskRun = { id: number; profile?: string | null; status?: string | null; outcome?: string | null; summary?: string | null; error?: string | null; started_at?: number | null; ended_at?: number | null; last_heartbeat_at?: number | null };
type TaskDetailResponse = { task: KanbanTask; comments: TaskComment[]; events: TaskEvent[]; links: { parents: string[]; children: string[] }; runs: TaskRun[] };

type OfficeRoom = "front" | "strategy" | "build" | "quality" | "ops";

type RoleMeta = {
  name: OfficeProfileName;
  title: string;
  room: OfficeRoom;
  seat: string;
  description: string;
  icon: typeof UserRound;
};

const OFFICE_ROLES: RoleMeta[] = [
  { name: "triage", title: "Triage Desk", room: "front", seat: "reception", description: "Receives ideas and turns intake into clear tasks.", icon: MessageSquarePlus },
  { name: "chief", title: "Chief", room: "strategy", seat: "corner office", description: "Owns ambiguous decisions and routing fallbacks.", icon: Sparkles },
  { name: "supervisor", title: "Supervisor", room: "strategy", seat: "control desk", description: "Watches stuck, stale, or blocked work.", icon: Activity },
  { name: "pm", title: "Product Manager", room: "strategy", seat: "planning wall", description: "Turns intent into requirements, milestones, and acceptance criteria.", icon: Users },
  { name: "architect", title: "Architect", room: "strategy", seat: "blueprint table", description: "Designs systems, boundaries, data flow, and integration shape.", icon: Network },
  { name: "research", title: "Research", room: "strategy", seat: "library", description: "Gathers external evidence, references, and options.", icon: Brain },
  { name: "coder", title: "Builder", room: "build", seat: "dev pod", description: "Implements scoped product changes.", icon: Wrench },
  { name: "tooling", title: "Tooling", room: "build", seat: "integration bench", description: "Connects MCP, APIs, tools, and runtime affordances.", icon: Zap },
  { name: "memory", title: "Memory/RAG", room: "build", seat: "archive", description: "Builds retrieval, durable context, and scoped memory.", icon: GitBranch },
  { name: "reviewer", title: "Reviewer", room: "quality", seat: "review desk", description: "Inspects diffs, risks, and implementation quality.", icon: CheckCircle2 },
  { name: "qa", title: "QA/Evals", room: "quality", seat: "test lab", description: "Runs regression checks, evals, and acceptance tests.", icon: Circle },
  { name: "security", title: "Security", room: "quality", seat: "threat room", description: "Threat models risky actions, policies, secrets, and permissions.", icon: Shield },
  { name: "approval", title: "Human Approval", room: "quality", seat: "approval gate", description: "Escalates risky steps for human-in-the-loop decisions.", icon: UserRound },
  { name: "observability", title: "Observability", room: "ops", seat: "telemetry wall", description: "Adds traces, metrics, logs, dashboards, and signals.", icon: Activity },
  { name: "devops", title: "DevOps", room: "ops", seat: "deploy station", description: "Handles deploy, migrations, infra, CI, and release readiness.", icon: Building2 },
  { name: "docs", title: "Docs", room: "ops", seat: "docs desk", description: "Writes runbooks, guides, and architecture documentation.", icon: MessageSquarePlus },
  { name: "demo", title: "Founder Demo", room: "ops", seat: "showcase wall", description: "Builds the story and demo path for founder-grade presentation.", icon: Sparkles },
];

const ROOMS: Array<{ id: OfficeRoom; label: string; className: string }> = [
  { id: "front", label: "Front Desk", className: "lg:col-span-3" },
  { id: "strategy", label: "Strategy Room", className: "lg:col-span-6" },
  { id: "build", label: "Build Floor", className: "lg:col-span-5" },
  { id: "quality", label: "Quality / Security", className: "lg:col-span-4" },
  { id: "ops", label: "Ops + Showcase", className: "lg:col-span-7" },
];

const STATUS_LABELS: Record<string, string> = {
  triage: "intake",
  todo: "waiting",
  ready: "queued",
  running: "working",
  blocked: "blocked",
  done: "done",
};

const STATUS_ORDER = ["all", "triage", "todo", "ready", "running", "blocked", "done"];

const ROOM_GLOW: Record<OfficeRoom, string> = {
  front: "from-cyan-400/15",
  strategy: "from-violet-400/15",
  build: "from-emerald-400/15",
  quality: "from-amber-400/15",
  ops: "from-sky-400/15",
};

function roleMap(tasks: KanbanTask[]) {
  const map = new Map<string, KanbanTask[]>();
  for (const task of tasks) {
    const key = task.assignee || task.office_role || "unassigned";
    map.set(key, [...(map.get(key) ?? []), task]);
  }
  return map;
}

function activeTask(tasks: KanbanTask[]): KanbanTask | undefined {
  return tasks.find((t) => t.status === "running") ?? tasks.find((t) => t.status === "blocked") ?? tasks.find((t) => t.status === "ready") ?? tasks[0];
}

function taskStatusTone(status: string): string {
  if (status === "running") return "border-emerald-300/70 bg-emerald-300/10 text-emerald-100";
  if (status === "blocked") return "border-red-300/70 bg-red-300/10 text-red-100";
  if (status === "ready") return "border-sky-300/70 bg-sky-300/10 text-sky-100";
  if (status === "triage") return "border-violet-300/60 bg-violet-300/10 text-violet-100";
  if (status === "done") return "border-muted-foreground/40 bg-muted/20 text-muted-foreground";
  return "border-border bg-card/50 text-muted-foreground";
}

function diagnosticTone(severity?: string): string {
  if (severity === "critical") return "border-red-300/70 bg-red-300/15 text-red-100";
  if (severity === "error") return "border-orange-300/70 bg-orange-300/15 text-orange-100";
  return "border-amber-300/70 bg-amber-300/10 text-amber-100";
}

function miniTaskLabel(task: KanbanTask): string {
  const label = STATUS_LABELS[task.status] ?? task.status;
  return `${label}: ${task.title}`;
}

function timeAgo(now: number, seconds?: number | null): string {
  if (!seconds) return "unknown";
  const delta = Math.max(0, now - seconds);
  if (delta < 60) return `${delta}s ago`;
  if (delta < 3600) return `${Math.floor(delta / 60)}m ago`;
  if (delta < 86400) return `${Math.floor(delta / 3600)}h ago`;
  return `${Math.floor(delta / 86400)}d ago`;
}

function taskAgeSeconds(now: number, task: KanbanTask): number {
  return Math.max(0, now - (task.started_at || task.created_at || now));
}

function StatusDot({ status }: { status: string }) {
  return (
    <span
      className={cn(
        "inline-block h-2.5 w-2.5 rounded-full border",
        status === "running" && "border-emerald-200 bg-emerald-300 shadow-[0_0_16px_rgba(110,231,183,0.75)]",
        status === "blocked" && "border-red-200 bg-red-300 shadow-[0_0_16px_rgba(252,165,165,0.75)]",
        status === "ready" && "border-sky-200 bg-sky-300 shadow-[0_0_12px_rgba(125,211,252,0.65)]",
        status !== "running" && status !== "blocked" && status !== "ready" && "border-muted-foreground/60 bg-muted-foreground/30",
      )}
    />
  );
}

function TaskSummaryCard({ task, now, onOpen, compact = false }: { task: KanbanTask; now: number; onOpen: (id: string) => void; compact?: boolean }) {
  const highestSeverity = task.warnings?.highest_severity ?? task.diagnostics?.[0]?.severity;
  return (
    <button
      type="button"
      onClick={() => onOpen(task.id)}
      className="w-full border border-border bg-card/40 p-3 text-left transition hover:border-midground/60 hover:bg-card/75 focus:border-midground focus:outline-none"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="line-clamp-2 text-sm normal-case text-foreground">{task.title}</div>
          <div className="mt-1 text-[10px] normal-case text-muted-foreground">{task.id} · {task.assignee ? `@${task.assignee}` : "unassigned"} · {timeAgo(now, task.started_at || task.created_at)}</div>
        </div>
        <Badge className={cn("shrink-0 border px-2 py-0 text-[10px]", taskStatusTone(task.status))}>{STATUS_LABELS[task.status] ?? task.status}</Badge>
      </div>
      {!compact && task.latest_summary && <p className="mt-2 line-clamp-2 text-xs normal-case leading-snug text-muted-foreground">{task.latest_summary}</p>}
      <div className="mt-2 flex flex-wrap items-center gap-1.5">
        {typeof task.priority === "number" && task.priority > 0 && <Badge className="border border-border bg-muted/20 px-2 py-0 text-[10px] text-muted-foreground">P{task.priority}</Badge>}
        {task.progress && <Badge className="border border-border bg-muted/20 px-2 py-0 text-[10px] text-muted-foreground">{task.progress.done}/{task.progress.total} children</Badge>}
        {(task.comment_count ?? 0) > 0 && <Badge className="border border-border bg-muted/20 px-2 py-0 text-[10px] text-muted-foreground">{task.comment_count} comments</Badge>}
        {highestSeverity && <Badge className={cn("border px-2 py-0 text-[10px]", diagnosticTone(highestSeverity))}>{highestSeverity}</Badge>}
      </div>
    </button>
  );
}

function PersonCard({ role, tasks, now, office, onOpenTask }: { role: RoleMeta; tasks: KanbanTask[]; now: number; office?: OfficeStatus; onOpenTask: (id: string) => void }) {
  const current = activeTask(tasks);
  const busy = Boolean(current && current.status !== "done");
  const status = current?.status ?? "idle";
  const installed = office?.profiles?.present?.includes(role.name) ?? true;
  const stale = current?.status === "running" && taskAgeSeconds(now, current) > 30 * 60;
  const Icon = role.icon;
  const diagnostics = tasks.reduce((sum, task) => sum + (task.diagnostics?.length ?? task.warnings?.count ?? 0), 0);

  return (
    <div className="group relative overflow-hidden border border-border/70 bg-black/35 p-3 shadow-[0_0_0_1px_rgba(255,255,255,0.03)_inset] transition hover:border-midground/50 hover:bg-card/70">
      <div className="absolute inset-x-4 top-0 h-px bg-gradient-to-r from-transparent via-midground/40 to-transparent" />
      <div className="flex items-start gap-3">
        <div className={cn("relative flex h-12 w-10 shrink-0 items-center justify-center border", busy ? "border-emerald-200/60 bg-emerald-300/10" : "border-border bg-muted/10")}>
          <Icon className={cn("h-5 w-5", busy ? "text-emerald-100" : "text-muted-foreground")} />
          <span className="absolute -bottom-1 -right-1"><StatusDot status={status} /></span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <div>
              <div className="text-sm font-bold tracking-[0.05em] text-foreground">{role.title}</div>
              <div className="text-[10px] text-muted-foreground">@{role.name} · {role.seat}</div>
            </div>
            <Badge className={cn("border px-2 py-0 text-[10px] uppercase", !installed ? "border-red-300/60 bg-red-300/10 text-red-100" : stale ? "border-amber-300/60 bg-amber-300/10 text-amber-100" : taskStatusTone(status))}>
              {!installed ? "missing" : stale ? "stale" : busy ? STATUS_LABELS[status] ?? status : "idle"}
            </Badge>
          </div>
          <p className="mt-2 line-clamp-2 text-[11px] normal-case leading-snug text-muted-foreground">{role.description}</p>
          {current ? (
            <button type="button" onClick={() => onOpenTask(current.id)} className="mt-3 w-full rounded-sm border border-border/70 bg-background/50 p-2 text-left transition hover:border-midground/60 hover:bg-card/70 focus:border-midground focus:outline-none">
              <div className="line-clamp-1 text-xs normal-case text-foreground">{miniTaskLabel(current)}</div>
              {current.latest_summary && <div className="mt-1 line-clamp-1 text-[10px] normal-case text-muted-foreground">{current.latest_summary}</div>}
              <div className="mt-1 flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
                <span>{tasks.length} task{tasks.length === 1 ? "" : "s"}{diagnostics ? ` · ${diagnostics} alerts` : ""}</span>
                <span>{timeAgo(now, current.started_at || current.created_at)}</span>
              </div>
            </button>
          ) : !installed ? (
            <div className="mt-3 rounded-sm border border-red-300/40 bg-red-300/10 p-2 text-[11px] text-red-100">
              Profile missing on disk. Create or restore this Office profile before dispatching work here.
            </div>
          ) : (
            <div className="mt-3 rounded-sm border border-dashed border-border/70 bg-muted/5 p-2 text-[11px] text-muted-foreground">
              No assigned work. Available for dispatch.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function OfficeRoomCard({ room, roles, assignments, now, office, onOpenTask }: { room: (typeof ROOMS)[number]; roles: RoleMeta[]; assignments: Map<string, KanbanTask[]>; now: number; office?: OfficeStatus; onOpenTask: (id: string) => void }) {
  const roomTasks = roles.flatMap((role) => assignments.get(role.name) ?? []);
  const active = roomTasks.filter((task) => task.status === "running").length;
  const blocked = roomTasks.filter((task) => task.status === "blocked").length;
  return (
    <Card className={cn("relative overflow-hidden bg-gradient-to-br to-transparent", ROOM_GLOW[room.id], room.className)}>
      <div className="absolute inset-0 opacity-[0.08]" style={{ backgroundImage: "linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)", backgroundSize: "24px 24px" }} />
      <CardHeader className="relative">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>{room.label}</CardTitle>
            <CardDescription>{roles.length} people · {roomTasks.length} assigned · {active} active{blocked ? ` · ${blocked} blocked` : ""}</CardDescription>
          </div>
          <Building2 className="h-5 w-5 text-muted-foreground" />
        </div>
      </CardHeader>
      <CardContent className="relative grid gap-3 md:grid-cols-2">
        {roles.map((role) => (
          <PersonCard key={role.name} role={role} tasks={assignments.get(role.name) ?? []} now={now} office={office} onOpenTask={onOpenTask} />
        ))}
      </CardContent>
    </Card>
  );
}

function FlowMap({ tasks, selectedStatus, onSelectStatus }: { tasks: KanbanTask[]; selectedStatus: string; onSelectStatus: (status: string) => void }) {
  const stages = ["triage", "todo", "ready", "running", "blocked", "done"];
  const counts = Object.fromEntries(stages.map((stage) => [stage, tasks.filter((task) => task.status === stage).length]));
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Network className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle>Information Flow</CardTitle>
            <CardDescription>Click a lifecycle stage to filter the office.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-2 md:grid-cols-6">
          {stages.map((stage, index) => (
            <div key={stage} className="relative">
              <button
                type="button"
                onClick={() => onSelectStatus(selectedStatus === stage ? "all" : stage)}
                className={cn("min-h-24 w-full border bg-card/45 p-3 text-left transition hover:border-midground/60 hover:bg-card/80 focus:border-midground focus:outline-none", selectedStatus === stage ? "border-midground/70" : "border-border")}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-xs font-bold uppercase tracking-[0.08em] text-foreground">{STATUS_LABELS[stage]}</span>
                  <Badge className="border border-border bg-muted/20 px-2 py-0 text-[10px] text-muted-foreground">{counts[stage]}</Badge>
                </div>
                <div className="mt-3 h-2 overflow-hidden bg-muted/30">
                  <div className={cn("h-full", stage === "running" ? "bg-emerald-300" : stage === "blocked" ? "bg-red-300" : "bg-midground/60")} style={{ width: `${Math.min(100, Number(counts[stage]) * 18)}%` }} />
                </div>
                <div className="mt-2 text-[10px] normal-case text-muted-foreground">
                  {stage === "triage" && "Reception captures the idea."}
                  {stage === "todo" && "Task waits for specification."}
                  {stage === "ready" && "Routed to a role and ready."}
                  {stage === "running" && "A person/agent is actively doing it."}
                  {stage === "blocked" && "Supervisor attention needed."}
                  {stage === "done" && "Result handed back."}
                </div>
              </button>
              {index < stages.length - 1 && (
                <ArrowRight className="absolute -right-4 top-10 z-10 hidden h-5 w-5 text-muted-foreground md:block" />
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function AttentionQueue({ tasks, now, office, onOpenTask }: { tasks: KanbanTask[]; now: number; office?: OfficeStatus; onOpenTask: (id: string) => void }) {
  const diagnosticTasks = tasks.filter((task) => (task.diagnostics?.length ?? task.warnings?.count ?? 0) > 0);
  const blockedTasks = tasks.filter((task) => task.status === "blocked");
  const staleRunning = tasks.filter((task) => task.status === "running" && taskAgeSeconds(now, task) > 30 * 60);
  const readyTasks = tasks.filter((task) => task.status === "ready");
  const items = [
    ...diagnosticTasks.slice(0, 3).map((task) => ({ task, label: "diagnostic", tone: diagnosticTone(task.warnings?.highest_severity ?? task.diagnostics?.[0]?.severity), icon: AlertTriangle })),
    ...blockedTasks.slice(0, 3).map((task) => ({ task, label: "blocked", tone: taskStatusTone("blocked"), icon: AlertTriangle })),
    ...staleRunning.slice(0, 3).map((task) => ({ task, label: `running ${timeAgo(now, task.started_at || task.created_at)}`, tone: "border-amber-300/70 bg-amber-300/10 text-amber-100", icon: Clock3 })),
  ].slice(0, 5);

  return (
    <Card className={cn(!office?.gateway_running && "border-amber-300/40 bg-amber-300/5")}>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className={cn("h-5 w-5", office?.gateway_running ? "text-muted-foreground" : "text-amber-200")} />
            <div>
              <CardTitle>Attention Required</CardTitle>
              <CardDescription>Operator-first view of risk, stuck work, and dispatch readiness.</CardDescription>
            </div>
          </div>
          <Badge className={cn("border px-2 py-0 text-[10px]", items.length || !office?.gateway_running ? "border-amber-300/60 bg-amber-300/10 text-amber-100" : "border-emerald-300/60 bg-emerald-300/10 text-emerald-100")}>{items.length || !office?.gateway_running ? "needs review" : "clear"}</Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {!office?.gateway_running && (
          <div className="rounded-sm border border-amber-300/40 bg-amber-300/10 p-3 text-sm normal-case text-amber-100">
            Gateway is not running. New ready tasks may sit idle until the gateway dispatcher is started.
          </div>
        )}
        {items.length === 0 ? (
          <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No blocked, diagnostic, or stale running work detected. {readyTasks.length ? `${readyTasks.length} task${readyTasks.length === 1 ? " is" : "s are"} ready for dispatch.` : ""}</div>
        ) : (
          <div className="grid gap-2 xl:grid-cols-2">
            {items.map(({ task, label, tone, icon: Icon }) => (
              <button key={`${label}-${task.id}`} type="button" onClick={() => onOpenTask(task.id)} className="flex items-start gap-3 border border-border bg-card/40 p-3 text-left transition hover:border-midground/60 hover:bg-card/75 focus:border-midground focus:outline-none">
                <Icon className="mt-0.5 h-4 w-4 shrink-0 text-amber-100" />
                <div className="min-w-0 flex-1">
                  <div className="line-clamp-1 text-sm normal-case text-foreground">{task.title}</div>
                  <div className="mt-1 text-[10px] text-muted-foreground">{task.id} · {task.assignee ? `@${task.assignee}` : "unassigned"}</div>
                </div>
                <Badge className={cn("shrink-0 border px-2 py-0 text-[10px]", tone)}>{label}</Badge>
              </button>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function TaskExplorer({ tasks, now, query, setQuery, status, setStatus, assignee, setAssignee, assignees, onOpenTask }: { tasks: KanbanTask[]; now: number; query: string; setQuery: (value: string) => void; status: string; setStatus: (value: string) => void; assignee: string; setAssignee: (value: string) => void; assignees: string[]; onOpenTask: (id: string) => void }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Search className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle>Find Work</CardTitle>
            <CardDescription>Search, filter, and open task details without leaving the office.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search title, id, assignee, body, summary..."
          className="w-full border border-border bg-background px-3 py-2 text-sm normal-case text-foreground outline-none placeholder:text-muted-foreground focus:border-midground"
        />
        <div className="grid gap-2 sm:grid-cols-2">
          <select value={status} onChange={(e) => setStatus(e.target.value)} className="border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-midground">
            {STATUS_ORDER.map((item) => <option key={item} value={item}>{item === "all" ? "All statuses" : STATUS_LABELS[item] ?? item}</option>)}
          </select>
          <select value={assignee} onChange={(e) => setAssignee(e.target.value)} className="border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-midground">
            <option value="all">All assignees</option>
            <option value="unassigned">Unassigned</option>
            {assignees.map((person) => <option key={person} value={person}>@{person}</option>)}
          </select>
        </div>
        <div className="max-h-[460px] space-y-2 overflow-auto pr-1">
          {tasks.length === 0 ? (
            <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No tasks match the current filters.</div>
          ) : (
            tasks.slice(0, 20).map((task) => <TaskSummaryCard key={task.id} task={task} now={now} onOpen={onOpenTask} />)
          )}
        </div>
        {tasks.length > 20 && <div className="text-xs normal-case text-muted-foreground">Showing first 20 of {tasks.length}. Refine search to narrow results.</div>}
      </CardContent>
    </Card>
  );
}

function BoardPicker({ boards, selectedBoard, onSelectBoard, onSwitchCurrent }: { boards: BoardMeta[]; selectedBoard: string; onSelectBoard: (slug: string) => void; onSwitchCurrent: (slug: string) => void }) {
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <GitBranch className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle>Board Selector</CardTitle>
            <CardDescription>Switch between Office boards without leaving the command center.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <select value={selectedBoard || boards[0]?.slug || ""} onChange={(e) => onSelectBoard(e.target.value)} className="w-full border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-midground">
          {boards.map((board) => <option key={board.slug} value={board.slug}>{board.name || board.slug} · {board.total ?? 0} tasks{board.is_current ? " · current" : ""}</option>)}
        </select>
        <div className="grid max-h-40 gap-2 overflow-auto pr-1">
          {boards.map((board) => (
            <button key={board.slug} type="button" onClick={() => onSelectBoard(board.slug)} className={cn("border p-2 text-left transition hover:border-midground/60", selectedBoard === board.slug ? "border-midground/70 bg-card/80" : "border-border bg-card/40")}>
              <div className="flex items-center justify-between gap-2 text-sm normal-case text-foreground">
                <span>{board.name || board.slug}</span>
                <Badge className="border border-border bg-muted/20 px-2 py-0 text-[10px] text-muted-foreground">{board.total ?? 0}</Badge>
              </div>
              <div className="mt-1 text-[10px] normal-case text-muted-foreground">{board.slug}{board.is_current ? " · CLI/gateway current" : ""}</div>
            </button>
          ))}
        </div>
        <Button ghost onClick={() => onSwitchCurrent(selectedBoard)} className="w-full gap-2"><GitBranch className="h-4 w-4" /> Make current board</Button>
      </CardContent>
    </Card>
  );
}

function LiveEventFeed({ events, connected, now, onRefresh }: { events: LiveEvent[]; connected: boolean; now: number; onRefresh: () => void }) {
  return (
    <Card className={cn(connected ? "border-emerald-300/25" : "border-amber-300/25")}>
      <CardHeader>
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            {connected ? <Wifi className="h-5 w-5 text-emerald-100" /> : <WifiOff className="h-5 w-5 text-amber-100" />}
            <div>
              <CardTitle>Live Event Feed</CardTitle>
              <CardDescription>{connected ? "WebSocket connected; board refreshes on new events." : "WebSocket unavailable; polling fallback is active."}</CardDescription>
            </div>
          </div>
          <Button ghost onClick={onRefresh} className="gap-2"><RefreshCw className="h-4 w-4" /> Sync</Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {events.length === 0 ? <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No live events seen in this browser session yet.</div> : events.slice(0, 8).map((event) => (
          <button key={event.id} type="button" className="w-full border border-border bg-card/40 p-2 text-left text-xs normal-case text-muted-foreground">
            <div className="flex items-center justify-between gap-2"><span className="text-foreground">{event.kind}</span><span>{timeAgo(now, event.created_at)}</span></div>
            <div className="mt-1 truncate">{event.task_id || "board"} · event #{event.id}</div>
          </button>
        ))}
      </CardContent>
    </Card>
  );
}

function TaskTableView({ tasks, now, onOpenTask }: { tasks: KanbanTask[]; now: number; onOpenTask: (id: string) => void }) {
  const [sortKey, setSortKey] = useState<"priority" | "status" | "assignee" | "age">("priority");
  const sorted = useMemo(() => [...tasks].sort((a, b) => {
    if (sortKey === "priority") return (b.priority ?? 0) - (a.priority ?? 0) || a.title.localeCompare(b.title);
    if (sortKey === "status") return a.status.localeCompare(b.status) || a.title.localeCompare(b.title);
    if (sortKey === "assignee") return (a.assignee ?? "").localeCompare(b.assignee ?? "") || a.title.localeCompare(b.title);
    return taskAgeSeconds(now, b) - taskAgeSeconds(now, a);
  }), [now, sortKey, tasks]);
  return (
    <Card>
      <CardHeader>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2"><Table2 className="h-5 w-5 text-muted-foreground" /><div><CardTitle>Task Table</CardTitle><CardDescription>Sortable operator table for scanning large boards.</CardDescription></div></div>
          <select value={sortKey} onChange={(e) => setSortKey(e.target.value as typeof sortKey)} className="border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-midground">
            <option value="priority">Sort by priority</option><option value="age">Sort by age</option><option value="status">Sort by status</option><option value="assignee">Sort by assignee</option>
          </select>
        </div>
      </CardHeader>
      <CardContent>
        <div className="overflow-auto border border-border">
          <table className="w-full min-w-[900px] border-collapse text-sm normal-case">
            <thead className="bg-muted/20 text-xs uppercase tracking-[0.12em] text-muted-foreground"><tr><th className="p-2 text-left">Task</th><th className="p-2 text-left">Status</th><th className="p-2 text-left">Assignee</th><th className="p-2 text-left">Priority</th><th className="p-2 text-left">Age</th><th className="p-2 text-left">Signals</th></tr></thead>
            <tbody>
              {sorted.map((task) => (
                <tr key={task.id} onClick={() => onOpenTask(task.id)} className="cursor-pointer border-t border-border/70 transition hover:bg-card/70">
                  <td className="p-2"><div className="line-clamp-1 text-foreground">{task.title}</div><div className="text-[10px] text-muted-foreground">{task.id}</div></td>
                  <td className="p-2"><Badge className={cn("border", taskStatusTone(task.status))}>{STATUS_LABELS[task.status] ?? task.status}</Badge></td>
                  <td className="p-2 text-muted-foreground">{task.assignee ? `@${task.assignee}` : "unassigned"}</td>
                  <td className="p-2 text-muted-foreground">{task.priority ?? 0}</td>
                  <td className="p-2 text-muted-foreground">{timeAgo(now, task.started_at || task.created_at)}</td>
                  <td className="p-2 text-muted-foreground">{task.link_counts ? `${task.link_counts.parents}↑ ${task.link_counts.children}↓` : "0↑ 0↓"}{task.warnings?.count ? ` · ${task.warnings.count} alerts` : ""}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </CardContent>
    </Card>
  );
}

function DependencyGraph({ tasks, links, onOpenTask }: { tasks: KanbanTask[]; links: BoardLink[]; onOpenTask: (id: string) => void }) {
  const byId = useMemo(() => new Map(tasks.map((task) => [task.id, task])), [tasks]);
  const childrenByParent = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const link of links) map.set(link.parent_id, [...(map.get(link.parent_id) ?? []), link.child_id]);
    return map;
  }, [links]);
  const roots = tasks.filter((task) => (task.link_counts?.parents ?? 0) === 0 && ((task.link_counts?.children ?? 0) > 0 || links.length === 0)).slice(0, 12);
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2"><ListTree className="h-5 w-5 text-muted-foreground" /><div><CardTitle>Dependency Graph</CardTitle><CardDescription>Parent → child lineage for PM → coder → QA task chains.</CardDescription></div></div>
      </CardHeader>
      <CardContent className="space-y-3">
        {links.length === 0 ? <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No task dependencies on this board yet.</div> : roots.map((root) => (
          <div key={root.id} className="border border-border bg-card/35 p-3">
            <button type="button" onClick={() => onOpenTask(root.id)} className="flex w-full items-center justify-between gap-2 text-left"><span className="line-clamp-1 text-sm text-foreground">{root.title}</span><Badge className={cn("border", taskStatusTone(root.status))}>{STATUS_LABELS[root.status] ?? root.status}</Badge></button>
            <div className="mt-3 ml-4 space-y-2 border-l border-border pl-3">
              {(childrenByParent.get(root.id) ?? []).map((childId) => {
                const child = byId.get(childId);
                return <button key={childId} type="button" onClick={() => onOpenTask(childId)} className="flex w-full items-center justify-between gap-2 border border-border bg-background/60 p-2 text-left text-xs normal-case"><span className="line-clamp-1 text-foreground">↳ {child?.title ?? childId}</span><span className="text-muted-foreground">{child?.assignee ? `@${child.assignee}` : child?.status ?? "missing"}</span></button>;
              })}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}

function TaskCreatePanel({ assignees, board, onCreated }: { assignees: string[]; board: string; onCreated: () => void }) {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [assignee, setAssignee] = useState<"office" | string>("office");
  const [priority, setPriority] = useState(1);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const people = useMemo(() => {
    const known = OFFICE_ROLES.map((role) => role.name);
    return Array.from(new Set([...known, ...assignees])).sort();
  }, [assignees]);

  async function submit() {
    if (!title.trim()) return;
    setSubmitting(true);
    setError(null);
    try {
      await fetchJSON(`/api/plugins/kanban/tasks?board=${encodeURIComponent(board)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: title.trim(),
          body: body.trim() || null,
          assignee: assignee === "office" ? null : assignee,
          triage: assignee === "office",
          priority,
        }),
      });
      setTitle("");
      setBody("");
      onCreated();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Send className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle>Assign Work</CardTitle>
            <CardDescription>Send a task to intake, a role, or any known assignee.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Task title"
          className="w-full border border-border bg-background px-3 py-2 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-midground"
        />
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          placeholder="What should they do? Add acceptance criteria, files, constraints, and definition of done."
          rows={4}
          className="w-full resize-none border border-border bg-background px-3 py-2 text-sm normal-case text-foreground outline-none placeholder:text-muted-foreground focus:border-midground"
        />
        <div className="grid gap-3 sm:grid-cols-[1fr_120px]">
          <select
            value={assignee}
            onChange={(e) => setAssignee(e.target.value)}
            className="border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-midground"
          >
            <option value="office">Office intake / auto-route</option>
            {people.map((person) => (
              <option key={person} value={person}>@{person}</option>
            ))}
          </select>
          <select value={priority} onChange={(e) => setPriority(Number(e.target.value))} className="border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-midground">
            <option value={0}>P0 normal</option>
            <option value={1}>P1 high</option>
            <option value={2}>P2 urgent</option>
            <option value={3}>P3 now</option>
          </select>
        </div>
        <Button onClick={submit} disabled={!title.trim() || submitting} className="w-full gap-2">
          {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          Assign to Office
        </Button>
        {error && <div className="border border-red-300/50 bg-red-300/10 p-2 text-xs normal-case text-red-100">{error}</div>}
      </CardContent>
    </Card>
  );
}

function TaskDetailDrawer({ taskId, assignees, board, now, onClose, onChanged }: { taskId: string | null; assignees: string[]; board: string; now: number; onClose: () => void; onChanged: () => void }) {
  const [detail, setDetail] = useState<TaskDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [comment, setComment] = useState("");
  const [newAssignee, setNewAssignee] = useState("");

  const withBoard = useCallback((path: string) => `${path}${path.includes("?") ? "&" : "?"}board=${encodeURIComponent(board)}`, [board]);

  const loadDetail = useCallback(async () => {
    if (!taskId) return;
    setLoading(true);
    setError(null);
    try {
      const next = await fetchJSON<TaskDetailResponse>(withBoard(`/api/plugins/kanban/tasks/${encodeURIComponent(taskId)}`));
      setDetail(next);
      setNewAssignee(next.task.assignee ?? "");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [taskId, withBoard]);

  useEffect(() => {
    const id = window.setTimeout(() => {
      setDetail(null);
      setMessage(null);
      setComment("");
      void loadDetail();
    }, 0);
    return () => window.clearTimeout(id);
  }, [loadDetail]);

  async function runAction(label: string, fn: () => Promise<unknown>) {
    setActionBusy(label);
    setMessage(null);
    setError(null);
    try {
      await fn();
      setMessage(`${label} succeeded`);
      await loadDetail();
      onChanged();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setActionBusy(null);
    }
  }

  if (!taskId) return null;
  const task = detail?.task;
  const people = Array.from(new Set([...OFFICE_ROLES.map((role) => role.name), ...assignees])).sort();

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/55 backdrop-blur-sm" onClick={onClose}>
      <aside className="h-full w-full max-w-3xl overflow-auto border-l border-border bg-background shadow-2xl" onClick={(e) => e.stopPropagation()}>
        <div className="sticky top-0 z-10 flex items-start justify-between gap-3 border-b border-border bg-background/95 p-4 backdrop-blur">
          <div className="min-w-0">
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-muted-foreground"><FileText className="h-4 w-4" /> Task Detail</div>
            <h2 className="mt-1 line-clamp-2 text-xl font-bold tracking-[0.04em] text-foreground">{task?.title ?? taskId}</h2>
            <div className="mt-1 text-xs normal-case text-muted-foreground">{taskId}{task?.assignee ? ` · @${task.assignee}` : " · unassigned"}</div>
          </div>
          <Button ghost onClick={onClose} className="gap-2"><X className="h-4 w-4" /> Close</Button>
        </div>
        <div className="space-y-4 p-4">
          {loading && <div className="flex items-center gap-2 text-sm normal-case text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> Loading task detail...</div>}
          {error && <div className="border border-red-300/50 bg-red-300/10 p-3 text-sm normal-case text-red-100">{error}</div>}
          {message && <div className="border border-emerald-300/50 bg-emerald-300/10 p-3 text-sm normal-case text-emerald-100">{message}</div>}
          {task && (
            <>
              <div className="grid gap-3 md:grid-cols-4">
                <Card><CardContent className="p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Status</div><Badge className={cn("mt-2 border", taskStatusTone(task.status))}>{STATUS_LABELS[task.status] ?? task.status}</Badge></CardContent></Card>
                <Card><CardContent className="p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Priority</div><div className="mt-2 text-2xl text-foreground">{task.priority ?? 0}</div></CardContent></Card>
                <Card><CardContent className="p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Age</div><div className="mt-2 text-sm normal-case text-foreground">{timeAgo(now, task.started_at || task.created_at)}</div></CardContent></Card>
                <Card><CardContent className="p-3"><div className="text-[10px] uppercase tracking-[0.12em] text-muted-foreground">Links</div><div className="mt-2 text-sm normal-case text-foreground">{detail?.links.parents.length ?? 0} parents · {detail?.links.children.length ?? 0} children</div></CardContent></Card>
              </div>

              <Card>
                <CardHeader><CardTitle>Operator Actions</CardTitle><CardDescription>Use dashboard-native actions instead of dropping to CLI for routine recovery.</CardDescription></CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    {task.status === "triage" && <Button ghost disabled={Boolean(actionBusy)} onClick={() => runAction("Specify", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}/specify`), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ author: "dashboard" }) }))} className="gap-2"><Sparkles className="h-4 w-4" /> Specify</Button>}
                    {(task.status === "blocked" || task.status === "todo" || task.status === "triage") && <Button ghost disabled={Boolean(actionBusy)} onClick={() => runAction("Mark ready", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}`), { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "ready" }) }))} className="gap-2"><RotateCcw className="h-4 w-4" /> Ready</Button>}
                    {task.status === "running" && <Button ghost disabled={Boolean(actionBusy)} onClick={() => runAction("Reclaim", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}/reclaim`), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reason: "dashboard operator reclaim" }) }))} className="gap-2"><RotateCcw className="h-4 w-4" /> Reclaim</Button>}
                    {task.status !== "done" && <Button ghost disabled={Boolean(actionBusy)} onClick={() => runAction("Block", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}`), { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "blocked", block_reason: "blocked from dashboard" }) }))} className="gap-2"><AlertTriangle className="h-4 w-4" /> Block</Button>}
                    {task.status !== "done" && <Button ghost disabled={Boolean(actionBusy)} onClick={() => runAction("Complete", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}`), { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "done", summary: "Completed from Office dashboard" }) }))} className="gap-2"><CheckCircle2 className="h-4 w-4" /> Done</Button>}
                    <Button ghost disabled={Boolean(actionBusy)} onClick={() => runAction("Archive", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}`), { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: "archived" }) }))} className="gap-2"><Archive className="h-4 w-4" /> Archive</Button>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-[1fr_auto]">
                    <select value={newAssignee} onChange={(e) => setNewAssignee(e.target.value)} className="border border-border bg-background px-3 py-2 text-sm text-foreground outline-none focus:border-midground">
                      <option value="">Unassigned / intake</option>
                      {people.map((person) => <option key={person} value={person}>@{person}</option>)}
                    </select>
                    <Button disabled={Boolean(actionBusy)} onClick={() => runAction("Reassign", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}/reassign`), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ profile: newAssignee || null, reclaim_first: task.status === "running", reason: "dashboard operator reassign" }) }))}>Reassign</Button>
                  </div>
                  {actionBusy && <div className="flex items-center gap-2 text-xs normal-case text-muted-foreground"><Loader2 className="h-4 w-4 animate-spin" /> {actionBusy} in progress...</div>}
                </CardContent>
              </Card>

              <Card>
                <CardHeader><CardTitle>Brief</CardTitle><CardDescription>Original request and latest worker handoff.</CardDescription></CardHeader>
                <CardContent className="space-y-3 text-sm normal-case text-muted-foreground">
                  <div className="whitespace-pre-wrap border border-border bg-card/40 p-3">{task.body || "No body provided."}</div>
                  {task.latest_summary && <div className="whitespace-pre-wrap border border-border bg-card/40 p-3 text-foreground">{task.latest_summary}</div>}
                </CardContent>
              </Card>

              {(task.diagnostics?.length ?? 0) > 0 && (
                <Card className="border-amber-300/40">
                  <CardHeader><CardTitle>Diagnostics</CardTitle><CardDescription>Signals that need operator attention.</CardDescription></CardHeader>
                  <CardContent className="space-y-2">
                    {task.diagnostics?.map((diag, index) => (
                      <div key={`${diag.kind}-${index}`} className="border border-amber-300/40 bg-amber-300/10 p-3 text-sm normal-case text-amber-100">
                        <div className="font-semibold uppercase tracking-[0.08em]">{diag.severity ?? "warning"} · {diag.kind ?? "diagnostic"}</div>
                        <div className="mt-1">{diag.message ?? "No diagnostic message."}</div>
                      </div>
                    ))}
                  </CardContent>
                </Card>
              )}

              <Card>
                <CardHeader><CardTitle>Comments</CardTitle><CardDescription>Add operator notes that future workers can read.</CardDescription></CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2">
                    {detail?.comments.length ? detail.comments.map((entry) => (
                      <div key={entry.id} className="border border-border bg-card/40 p-3 text-sm normal-case text-muted-foreground">
                        <div className="mb-1 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">{entry.author || "unknown"} · {timeAgo(now, entry.created_at)}</div>
                        <div className="whitespace-pre-wrap text-foreground">{entry.body}</div>
                      </div>
                    )) : <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No comments yet.</div>}
                  </div>
                  <textarea value={comment} onChange={(e) => setComment(e.target.value)} rows={3} placeholder="Add a note for the next worker..." className="w-full resize-none border border-border bg-background px-3 py-2 text-sm normal-case text-foreground outline-none placeholder:text-muted-foreground focus:border-midground" />
                  <Button disabled={!comment.trim() || Boolean(actionBusy)} onClick={() => runAction("Comment", () => fetchJSON(withBoard(`/api/plugins/kanban/tasks/${task.id}/comments`), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ body: comment.trim(), author: "dashboard" }) }).then(() => setComment("")))} className="gap-2"><MessageSquare className="h-4 w-4" /> Add Comment</Button>
                </CardContent>
              </Card>

              <div className="grid gap-4 lg:grid-cols-2">
                <Card>
                  <CardHeader><CardTitle>Run History</CardTitle><CardDescription>Worker attempts and outcomes.</CardDescription></CardHeader>
                  <CardContent className="space-y-2">
                    {detail?.runs.length ? detail.runs.slice().reverse().map((run) => (
                      <div key={run.id} className="border border-border bg-card/40 p-3 text-xs normal-case text-muted-foreground">
                        <div className="flex items-center justify-between gap-2"><span className="text-foreground">@{run.profile || "unknown"}</span><Badge className="border border-border bg-muted/20 px-2 py-0 text-[10px] text-muted-foreground">{run.outcome || run.status || "running"}</Badge></div>
                        {run.summary && <div className="mt-2 line-clamp-3 text-foreground">{run.summary}</div>}
                        {run.error && <div className="mt-2 text-red-100">{run.error}</div>}
                      </div>
                    )) : <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No run history.</div>}
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader><CardTitle>Event Timeline</CardTitle><CardDescription>Recent state changes.</CardDescription></CardHeader>
                  <CardContent className="space-y-2">
                    {detail?.events.length ? detail.events.slice().reverse().slice(0, 12).map((event) => (
                      <div key={event.id} className="border border-border bg-card/40 p-2 text-xs normal-case text-muted-foreground">
                        <div className="flex items-center justify-between gap-2"><span className="text-foreground">{event.office_kind || event.kind}</span><span>{timeAgo(now, event.created_at)}</span></div>
                      </div>
                    )) : <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No events.</div>}
                  </CardContent>
                </Card>
              </div>
            </>
          )}
        </div>
      </aside>
    </div>
  );
}

export default function OfficePage() {
  const { setTitle } = usePageHeader();
  const [data, setData] = useState<BoardResponse | null>(null);
  const [boards, setBoards] = useState<BoardMeta[]>([]);
  const [selectedBoard, setSelectedBoardState] = useState(() => window.localStorage.getItem("hermes.office.selectedBoard") || "");
  const [liveEvents, setLiveEvents] = useState<LiveEvent[]>([]);
  const [liveConnected, setLiveConnected] = useState(false);
  const [viewMode, setViewMode] = useState<"floor" | "table" | "graph">("floor");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [assigneeFilter, setAssigneeFilter] = useState("all");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [fallbackNow, setFallbackNow] = useState(0);

  useEffect(() => {
    setTitle("Office");
    return () => setTitle(null);
  }, [setTitle]);

  const setSelectedBoard = useCallback((slug: string) => {
    setSelectedBoardState(slug);
    window.localStorage.setItem("hermes.office.selectedBoard", slug);
    setLiveEvents([]);
  }, []);

  const loadBoards = useCallback(async () => {
    try {
      const response = await fetchJSON<BoardListResponse>("/api/plugins/kanban/boards");
      const nextBoards = response.boards.length ? response.boards : [{ slug: response.current || "inbox", name: response.current || "inbox", total: 0 }];
      setBoards(nextBoards);
      if ((!selectedBoard || !nextBoards.some((board) => board.slug === selectedBoard)) && response.current) setSelectedBoard(response.current);
    } catch {
      setBoards((prev) => prev.length ? prev : [{ slug: selectedBoard, name: selectedBoard, total: 0 }]);
    }
  }, [selectedBoard, setSelectedBoard]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const board = await fetchJSON<BoardResponse>(selectedBoard ? `/api/plugins/kanban/board?board=${encodeURIComponent(selectedBoard)}` : "/api/plugins/kanban/board");
      setData(board);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, [selectedBoard]);

  useEffect(() => {
    const id = window.setTimeout(() => void loadBoards(), 0);
    return () => window.clearTimeout(id);
  }, [loadBoards]);

  useEffect(() => {
    const first = window.setTimeout(() => void load(), 0);
    const id = window.setInterval(() => void load(), 7000);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(id);
    };
  }, [load]);

  useEffect(() => {
    const token = (window as unknown as { __HERMES_SESSION_TOKEN__?: string }).__HERMES_SESSION_TOKEN__;
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const params = new URLSearchParams({ since: String(data?.latest_event_id ?? 0), board: selectedBoard });
    if (token) params.set("token", token);
    const ws = new WebSocket(`${proto}://${window.location.host}/api/plugins/kanban/events?${params.toString()}`);
    ws.onopen = () => setLiveConnected(true);
    ws.onclose = () => setLiveConnected(false);
    ws.onerror = () => setLiveConnected(false);
    ws.onmessage = (message) => {
      try {
        const payload = JSON.parse(message.data) as { events?: LiveEvent[]; cursor?: number };
        if (payload.events?.length) {
          setLiveEvents((prev) => [...payload.events!.slice().reverse(), ...prev].slice(0, 50));
          void load();
          void loadBoards();
        }
      } catch {
        // Ignore malformed event frames; polling remains active.
      }
    };
    return () => ws.close();
  }, [data?.latest_event_id, load, loadBoards, selectedBoard]);

  useEffect(() => {
    const tick = () => setFallbackNow(Math.floor(Date.now() / 1000));
    const first = window.setTimeout(tick, 0);
    const id = window.setInterval(tick, 30000);
    return () => {
      window.clearTimeout(first);
      window.clearInterval(id);
    };
  }, []);

  const tasks = useMemo(() => data?.columns.flatMap((column) => column.tasks) ?? [], [data]);
  const links = data?.links ?? [];
  const now = data?.now ?? fallbackNow;
  const assignees = useMemo(() => Array.from(new Set([...(data?.assignees ?? []), ...OFFICE_ROLES.map((role) => role.name)])).sort(), [data?.assignees]);
  const filteredTasks = useMemo(() => {
    const q = query.trim().toLowerCase();
    return tasks.filter((task) => {
      if (statusFilter !== "all" && task.status !== statusFilter) return false;
      if (assigneeFilter === "unassigned" && task.assignee) return false;
      if (assigneeFilter !== "all" && assigneeFilter !== "unassigned" && task.assignee !== assigneeFilter) return false;
      if (!q) return true;
      const haystack = [task.id, task.title, task.body, task.latest_summary, task.assignee, task.status].filter(Boolean).join(" ").toLowerCase();
      return haystack.includes(q);
    });
  }, [assigneeFilter, query, statusFilter, tasks]);
  const assignments = useMemo(() => roleMap(filteredTasks), [filteredTasks]);
  const unassigned = assignments.get("unassigned") ?? [];
  const activeCount = tasks.filter((task) => task.status === "running").length;
  const idleCount = OFFICE_ROLES.filter((role) => (roleMap(tasks).get(role.name) ?? []).filter((task) => task.status !== "done").length === 0).length;
  const blockedCount = tasks.filter((task) => task.status === "blocked").length;
  const diagnosticCount = tasks.filter((task) => (task.diagnostics?.length ?? task.warnings?.count ?? 0) > 0).length;

  async function dispatchNow() {
    setError(null);
    try {
      await fetchJSON(`/api/plugins/kanban/dispatch?board=${encodeURIComponent(selectedBoard)}`, { method: "POST" });
      await load();
      await loadBoards();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  async function switchCurrentBoard(slug: string) {
    setError(null);
    try {
      await fetchJSON(`/api/plugins/kanban/boards/${encodeURIComponent(slug)}/switch`, { method: "POST" });
      await loadBoards();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  return (
    <div className="h-full overflow-auto p-4 lg:p-6">
      <div className="mx-auto flex max-w-[1800px] flex-col gap-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-muted-foreground">
              <Building2 className="h-4 w-4" /> Nous Hermes Office
            </div>
            <h1 className="mt-1 text-2xl font-bold tracking-[0.06em] text-foreground">Agent Office Command Center</h1>
            <p className="mt-1 max-w-3xl text-sm normal-case text-muted-foreground">
              Operate the agent office: find work, inspect handoffs, recover stuck tasks, route assignments, and watch flow from intake to done.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge className="border border-sky-300/60 bg-sky-300/10 px-3 py-1 text-sky-100">board {selectedBoard}</Badge>
            <Badge className={cn("border px-3 py-1", liveConnected ? "border-emerald-300/60 bg-emerald-300/10 text-emerald-100" : "border-amber-300/60 bg-amber-300/10 text-amber-100")}>live {liveConnected ? "connected" : "polling"}</Badge>
            <Badge className={cn("border px-3 py-1", data?.office?.enabled ? "border-emerald-300/60 bg-emerald-300/10 text-emerald-100" : "border-red-300/60 bg-red-300/10 text-red-100")}>office {data?.office?.enabled ? "enabled" : "disabled"}</Badge>
            <Badge className={cn("border px-3 py-1", data?.office?.gateway_running ? "border-emerald-300/60 bg-emerald-300/10 text-emerald-100" : "border-amber-300/60 bg-amber-300/10 text-amber-100")}>gateway {data?.office?.gateway_running ? "running" : "not running"}</Badge>
            <Button ghost onClick={() => void dispatchNow()} className="gap-2"><Zap className="h-4 w-4" /> Dispatch now</Button>
            <Button ghost onClick={load} disabled={loading} className="gap-2">
              <RefreshCw className={cn("h-4 w-4", loading && "animate-spin")} /> Refresh
            </Button>
          </div>
        </div>

        {error && (
          <Card className="border-red-300/50 bg-red-300/10">
            <CardContent className="text-sm normal-case text-red-100">Failed to load office data: {error}</CardContent>
          </Card>
        )}

        <div className="grid gap-3 md:grid-cols-5">
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Active workers</div><div className="mt-2 text-3xl text-foreground">{activeCount}</div></CardContent></Card>
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Idle people</div><div className="mt-2 text-3xl text-foreground">{idleCount}</div></CardContent></Card>
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Blocked</div><div className="mt-2 text-3xl text-foreground">{blockedCount}</div></CardContent></Card>
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Diagnostics</div><div className="mt-2 text-3xl text-foreground">{diagnosticCount}</div></CardContent></Card>
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Shown / Total</div><div className="mt-2 text-3xl text-foreground">{filteredTasks.length}/{tasks.length}</div></CardContent></Card>
        </div>

        <div className="flex flex-wrap items-center gap-2 border border-border bg-card/30 p-2">
          <Button ghost onClick={() => setViewMode("floor")} className={cn("gap-2", viewMode === "floor" && "border-midground/70 bg-card/80")}><LayoutGrid className="h-4 w-4" /> Floor</Button>
          <Button ghost onClick={() => setViewMode("table")} className={cn("gap-2", viewMode === "table" && "border-midground/70 bg-card/80")}><Table2 className="h-4 w-4" /> Table</Button>
          <Button ghost onClick={() => setViewMode("graph")} className={cn("gap-2", viewMode === "graph" && "border-midground/70 bg-card/80")}><ListTree className="h-4 w-4" /> Dependencies</Button>
        </div>

        <AttentionQueue tasks={tasks} now={now} office={data?.office} onOpenTask={setSelectedTaskId} />

        <div className="grid gap-4 xl:grid-cols-[1fr_460px]">
          <div className="space-y-4">
            <FlowMap tasks={tasks} selectedStatus={statusFilter} onSelectStatus={setStatusFilter} />
            {viewMode === "floor" && (
              <div className="grid gap-4 lg:grid-cols-12">
                {ROOMS.map((room) => (
                  <OfficeRoomCard
                    key={room.id}
                    room={room}
                    roles={OFFICE_ROLES.filter((role) => role.room === room.id)}
                    assignments={assignments}
                    now={now}
                    office={data?.office}
                    onOpenTask={setSelectedTaskId}
                  />
                ))}
              </div>
            )}
            {viewMode === "table" && <TaskTableView tasks={filteredTasks} now={now} onOpenTask={setSelectedTaskId} />}
            {viewMode === "graph" && <DependencyGraph tasks={tasks} links={links} onOpenTask={setSelectedTaskId} />}
          </div>
          <div className="space-y-4">
            <BoardPicker boards={boards} selectedBoard={selectedBoard} onSelectBoard={setSelectedBoard} onSwitchCurrent={(slug) => void switchCurrentBoard(slug)} />
            <LiveEventFeed events={liveEvents} connected={liveConnected} now={now} onRefresh={() => { void load(); void loadBoards(); }} />
            <TaskExplorer tasks={filteredTasks} now={now} query={query} setQuery={setQuery} status={statusFilter} setStatus={setStatusFilter} assignee={assigneeFilter} setAssignee={setAssigneeFilter} assignees={assignees} onOpenTask={setSelectedTaskId} />
            <TaskCreatePanel assignees={data?.assignees ?? []} board={selectedBoard} onCreated={() => { void load(); void loadBoards(); }} />
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2"><Inbox className="h-5 w-5 text-muted-foreground" /><div><CardTitle>Unassigned / Office Intake</CardTitle><CardDescription>Work not sitting at a named desk yet.</CardDescription></div></div>
              </CardHeader>
              <CardContent className="space-y-2">
                {unassigned.length === 0 ? (
                  <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No unassigned work in the current filter.</div>
                ) : (
                  unassigned.slice(0, 8).map((task) => <TaskSummaryCard key={task.id} task={task} now={now} onOpen={setSelectedTaskId} compact />)
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Office Health</CardTitle>
                <CardDescription>Profiles, board, and dispatcher prerequisites.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm normal-case text-muted-foreground">
                <div className="flex justify-between gap-3"><span>Board</span><span className="text-foreground">{selectedBoard} / status {data?.office?.board ?? "unknown"}</span></div>
                <div className="flex justify-between gap-3"><span>Live updates</span><span className="text-foreground">{liveConnected ? "websocket" : "polling fallback"}</span></div>
                <div className="flex justify-between gap-3"><span>Board exists</span><span className="text-foreground">{data?.office?.board_exists ? "yes" : "no"}</span></div>
                <div className="flex justify-between gap-3"><span>Profiles present</span><span className="text-foreground">{data?.office?.profiles?.present?.length ?? 0}/{data?.office?.profiles?.expected ?? OFFICE_ROLES.length}</span></div>
                {(data?.office?.profiles?.missing?.length ?? 0) > 0 && (
                  <div className="rounded-sm border border-amber-300/40 bg-amber-300/10 p-2 text-xs text-amber-100">
                    Missing profiles: {data?.office?.profiles?.missing.join(", ")}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
      <TaskDetailDrawer taskId={selectedTaskId} assignees={assignees} board={selectedBoard} now={now} onClose={() => setSelectedTaskId(null)} onChanged={load} />
    </div>
  );
}
