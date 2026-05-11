import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowRight,
  Brain,
  Building2,
  CheckCircle2,
  Circle,
  GitBranch,
  Loader2,
  MessageSquarePlus,
  Network,
  RefreshCw,
  Send,
  Shield,
  Sparkles,
  UserRound,
  Users,
  Wrench,
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
  link_counts?: { parents: number; children: number };
  progress?: { done: number; total: number } | null;
  diagnostics?: Array<{ severity?: string; message?: string; kind?: string }>;
};

type BoardColumn = { name: string; tasks: KanbanTask[] };
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
  latest_event_id: number;
  now: number;
  office?: OfficeStatus;
};

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
  if (status === "done") return "border-muted-foreground/40 bg-muted/20 text-muted-foreground";
  return "border-border bg-card/50 text-muted-foreground";
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

function PersonCard({ role, tasks, now }: { role: RoleMeta; tasks: KanbanTask[]; now: number }) {
  const current = activeTask(tasks);
  const busy = Boolean(current && current.status !== "done");
  const status = current?.status ?? "idle";
  const Icon = role.icon;

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
            <Badge className={cn("border px-2 py-0 text-[10px] uppercase", taskStatusTone(status))}>
              {busy ? STATUS_LABELS[status] ?? status : "idle"}
            </Badge>
          </div>
          <p className="mt-2 line-clamp-2 text-[11px] normal-case leading-snug text-muted-foreground">{role.description}</p>
          {current ? (
            <div className="mt-3 rounded-sm border border-border/70 bg-background/50 p-2">
              <div className="line-clamp-1 text-xs normal-case text-foreground">{miniTaskLabel(current)}</div>
              <div className="mt-1 flex items-center justify-between gap-2 text-[10px] text-muted-foreground">
                <span>{tasks.length} task{tasks.length === 1 ? "" : "s"}</span>
                <span>{timeAgo(now, current.started_at || current.created_at)}</span>
              </div>
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

function OfficeRoomCard({ room, roles, assignments, now }: { room: (typeof ROOMS)[number]; roles: RoleMeta[]; assignments: Map<string, KanbanTask[]>; now: number }) {
  const roomTasks = roles.flatMap((role) => assignments.get(role.name) ?? []);
  const active = roomTasks.filter((task) => task.status === "running").length;
  return (
    <Card className={cn("relative overflow-hidden bg-gradient-to-br to-transparent", ROOM_GLOW[room.id], room.className)}>
      <div className="absolute inset-0 opacity-[0.08]" style={{ backgroundImage: "linear-gradient(currentColor 1px, transparent 1px), linear-gradient(90deg, currentColor 1px, transparent 1px)", backgroundSize: "24px 24px" }} />
      <CardHeader className="relative">
        <div className="flex items-center justify-between gap-3">
          <div>
            <CardTitle>{room.label}</CardTitle>
            <CardDescription>{roles.length} people · {roomTasks.length} assigned · {active} active</CardDescription>
          </div>
          <Building2 className="h-5 w-5 text-muted-foreground" />
        </div>
      </CardHeader>
      <CardContent className="relative grid gap-3 md:grid-cols-2">
        {roles.map((role) => (
          <PersonCard key={role.name} role={role} tasks={assignments.get(role.name) ?? []} now={now} />
        ))}
      </CardContent>
    </Card>
  );
}

function FlowMap({ tasks }: { tasks: KanbanTask[] }) {
  const stages = ["triage", "todo", "ready", "running", "blocked", "done"];
  const counts = Object.fromEntries(stages.map((stage) => [stage, tasks.filter((task) => task.status === stage).length]));
  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Network className="h-5 w-5 text-muted-foreground" />
          <div>
            <CardTitle>Information Flow</CardTitle>
            <CardDescription>How work moves through the physical office.</CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-2 md:grid-cols-6">
          {stages.map((stage, index) => (
            <div key={stage} className="relative">
              <div className="min-h-24 border border-border bg-card/45 p-3">
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
              </div>
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

function TaskCreatePanel({ assignees, onCreated }: { assignees: string[]; onCreated: () => void }) {
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [assignee, setAssignee] = useState<"office" | string>("office");
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
      await fetchJSON("/api/plugins/kanban/tasks", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          title: title.trim(),
          body: body.trim() || null,
          assignee: assignee === "office" ? null : assignee,
          triage: assignee === "office",
          priority: 1,
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
            <CardDescription>Send a task to the office intake, a role, or any known assignee.</CardDescription>
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
          placeholder="What should they do?"
          rows={4}
          className="w-full resize-none border border-border bg-background px-3 py-2 text-sm normal-case text-foreground outline-none placeholder:text-muted-foreground focus:border-midground"
        />
        <div className="grid gap-3 sm:grid-cols-[1fr_auto]">
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
          <Button onClick={submit} disabled={!title.trim() || submitting} className="gap-2">
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            Assign
          </Button>
        </div>
        {error && <div className="border border-red-300/50 bg-red-300/10 p-2 text-xs normal-case text-red-100">{error}</div>}
      </CardContent>
    </Card>
  );
}

export default function OfficePage() {
  const { setTitle } = usePageHeader();
  const [data, setData] = useState<BoardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setTitle("Office");
    return () => setTitle(null);
  }, [setTitle]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const board = await fetchJSON<BoardResponse>("/api/plugins/kanban/board");
      setData(board);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    const id = window.setInterval(() => void load(), 7000);
    return () => window.clearInterval(id);
  }, [load]);

  const tasks = useMemo(() => data?.columns.flatMap((column) => column.tasks) ?? [], [data]);
  const assignments = useMemo(() => roleMap(tasks), [tasks]);
  const unassigned = assignments.get("unassigned") ?? [];
  const activeCount = tasks.filter((task) => task.status === "running").length;
  const idleCount = OFFICE_ROLES.filter((role) => (assignments.get(role.name) ?? []).filter((task) => task.status !== "done").length === 0).length;
  const blockedCount = tasks.filter((task) => task.status === "blocked").length;
  const now = data?.now ?? Math.floor(Date.now() / 1000);

  return (
    <div className="h-full overflow-auto p-4 lg:p-6">
      <div className="mx-auto flex max-w-[1800px] flex-col gap-4">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <div className="flex items-center gap-2 text-xs uppercase tracking-[0.16em] text-muted-foreground">
              <Building2 className="h-4 w-4" /> Nous Hermes Office
            </div>
            <h1 className="mt-1 text-2xl font-bold tracking-[0.06em] text-foreground">Agent Office Floor Plan</h1>
            <p className="mt-1 max-w-3xl text-sm normal-case text-muted-foreground">
              A graphical operations room: physical role desks, live status lights, queues, blocked work, and how information flows from intake to done.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Badge className={cn("border px-3 py-1", data?.office?.enabled ? "border-emerald-300/60 bg-emerald-300/10 text-emerald-100" : "border-red-300/60 bg-red-300/10 text-red-100")}>office {data?.office?.enabled ? "enabled" : "disabled"}</Badge>
            <Badge className={cn("border px-3 py-1", data?.office?.gateway_running ? "border-emerald-300/60 bg-emerald-300/10 text-emerald-100" : "border-amber-300/60 bg-amber-300/10 text-amber-100")}>gateway {data?.office?.gateway_running ? "running" : "not running"}</Badge>
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

        <div className="grid gap-3 md:grid-cols-4">
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Active workers</div><div className="mt-2 text-3xl text-foreground">{activeCount}</div></CardContent></Card>
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Idle people</div><div className="mt-2 text-3xl text-foreground">{idleCount}</div></CardContent></Card>
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Blocked</div><div className="mt-2 text-3xl text-foreground">{blockedCount}</div></CardContent></Card>
          <Card><CardContent className="p-4"><div className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground">Total tasks</div><div className="mt-2 text-3xl text-foreground">{tasks.length}</div></CardContent></Card>
        </div>

        <div className="grid gap-4 xl:grid-cols-[1fr_420px]">
          <div className="space-y-4">
            <FlowMap tasks={tasks} />
            <div className="grid gap-4 lg:grid-cols-12">
              {ROOMS.map((room) => (
                <OfficeRoomCard
                  key={room.id}
                  room={room}
                  roles={OFFICE_ROLES.filter((role) => role.room === room.id)}
                  assignments={assignments}
                  now={now}
                />
              ))}
            </div>
          </div>
          <div className="space-y-4">
            <TaskCreatePanel assignees={data?.assignees ?? []} onCreated={load} />
            <Card>
              <CardHeader>
                <CardTitle>Unassigned / Office Intake</CardTitle>
                <CardDescription>Work not sitting at a named desk yet.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2">
                {unassigned.length === 0 ? (
                  <div className="border border-dashed border-border p-3 text-sm normal-case text-muted-foreground">No unassigned work. The routing queue is clear.</div>
                ) : (
                  unassigned.slice(0, 8).map((task) => (
                    <div key={task.id} className="border border-border bg-card/40 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <div className="line-clamp-1 text-sm normal-case text-foreground">{task.title}</div>
                        <Badge className={cn("border px-2 py-0 text-[10px]", taskStatusTone(task.status))}>{STATUS_LABELS[task.status] ?? task.status}</Badge>
                      </div>
                      <div className="mt-1 text-[10px] text-muted-foreground">{task.id}</div>
                    </div>
                  ))
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Office Health</CardTitle>
                <CardDescription>Profiles and board presence.</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm normal-case text-muted-foreground">
                <div className="flex justify-between gap-3"><span>Board</span><span className="text-foreground">{data?.office?.board ?? "unknown"}</span></div>
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
    </div>
  );
}
