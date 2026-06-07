import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  Bot,
  Clipboard,
  Database,
  ExternalLink,
  FilePlus2,
  FileText,
  GitBranch,
  MessageSquare,
  PauseCircle,
  PlayCircle,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  Workflow,
  Zap,
} from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Input } from "@nous-research/ui/ui/components/input";
import { Select, SelectOption } from "@nous-research/ui/ui/components/select";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useToast } from "@nous-research/ui/hooks/use-toast";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { api } from "@/lib/api";
import type {
  FactoryDashboardResponse,
  FactoryFinding,
  FactoryGate,
  FactoryProject,
  FactoryProjectAction,
  FactoryTask,
} from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";
import { cn } from "@/lib/utils";

type BadgeTone = "success" | "warning" | "destructive" | "outline" | "secondary";

const PROJECT_PREVIEW_LIMIT = 5;
const FACTORY_AUTO_REFRESH_MS = 15_000;
const THINKING_MESSAGES = [
  "verificando Factory DB",
  "revisando gates efectivos",
  "escaneando workers",
  "esperando el próximo tick",
];

const STATIC_FACTORY_PROJECT_STATUSES = new Set([
  "blocked",
  "cancelled",
  "closed",
  "completed",
  "delivery_hold",
  "hold",
  "on_hold",
  "paused",
]);

const ACTIVE_FACTORY_RUN_STATUSES = new Set(["queued", "running"]);

const STATUS_COPY: Record<string, { label: string; description: string }> = {
  intake: {
    label: "Intake",
    description: "Entrada/deuda: el proyecto fue registrado o quedó medio ejecutado, pero no se cerró la metodología (PRD, plan, gates, docs/Notion).",
  },
  active: {
    label: "Activo",
    description: "Trabajo vivo o en seguimiento dentro de Factory DB.",
  },
  completed: {
    label: "Completado",
    description: "Cerrado administrativamente o entregado según su evidencia registrada.",
  },
  delivery_hold: {
    label: "Delivery hold",
    description: "Trabajo técnico terminado o en pausa de entrega; queda abierto por documentación, Notion o cierre explícito de Jean/UI, sin marcarse como bloqueo de programación.",
  },
  planned: {
    label: "Planificado",
    description: "Registrado para ejecución futura; aún no está en trabajo activo.",
  },
};

function formatDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

function formatShortDate(value?: string | null): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

function formatRelativeAge(value?: string | null, now = Date.now()): string {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const seconds = Math.max(0, Math.floor((now - date.getTime()) / 1000));
  if (seconds < 60) return `hace ${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `hace ${minutes}m ${seconds % 60}s`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `hace ${hours}h ${minutes % 60}m`;
  const days = Math.floor(hours / 24);
  return `hace ${days}d ${hours % 24}h`;
}

function truncateText(value?: string | null, limit = 360): string {
  if (!value) return "—";
  const compact = value.replace(/\s+/g, " ").trim();
  if (compact.length <= limit) return compact;
  return `${compact.slice(0, limit - 1)}…`;
}

function statusInfo(status?: string | null): { label: string; description: string } {
  const normalized = (status || "").toLowerCase();
  return STATUS_COPY[normalized] ?? {
    label: status || "unknown",
    description: status ? `Estado Factory DB: ${status}.` : "Estado Factory DB no disponible.",
  };
}

function isStaticFactoryProjectStatus(status?: string | null): boolean {
  return STATIC_FACTORY_PROJECT_STATUSES.has((status || "").toLowerCase());
}

function activeRunIsLive(project: FactoryProject): boolean {
  const runStatus = (project.dashboard?.active_run?.status || "").toLowerCase();
  return ACTIVE_FACTORY_RUN_STATUSES.has(runStatus);
}

function workflowIdleCopy(project: FactoryProject): string {
  const status = (project.status || "").toLowerCase();
  if (status === "completed" || status === "closed") return "cerrado; sin proceso activo";
  if (status === "delivery_hold" || status === "hold" || status === "on_hold") {
    return "en hold de entrega; sin proceso activo";
  }
  if (status === "blocked") return "bloqueado; esperando reparación o decisión";
  if (status === "paused" || status === "cancelled") return "pausado; sin proceso activo";
  if (status === "planned" || status === "intake") return "planificado; sin ejecución activa";
  return "sin ejecución activa";
}

function statusTone(status?: string | null): BadgeTone {
  const normalized = (status || "").toLowerCase();
  if (["done", "completed", "verified", "passed", "active"].includes(normalized)) {
    return "success";
  }
  if (["failed", "blocked"].includes(normalized)) return "destructive";
  if (["todo", "planned", "pending", "review_pending_human", "intake", "delivery_hold"].includes(normalized)) {
    return "warning";
  }
  return "secondary";
}

function findingTone(finding: FactoryFinding): BadgeTone {
  if (finding.severity === "destructive") return "destructive";
  if (finding.severity === "warning") return "warning";
  return "secondary";
}

function countLabel(counts?: Record<string, number>): string {
  if (!counts) return "—";
  const entries = Object.entries(counts);
  if (entries.length === 0) return "—";
  return entries.map(([key, value]) => `${key}: ${value}`).join(" · ");
}

function projectActivityAt(project: FactoryProject): string | null {
  return (
    project.dashboard?.latest_event?.created_at ??
    project.dashboard?.current_task?.updated_at ??
    project.dashboard?.current_task?.created_at ??
    project.updated_at ??
    project.started_at ??
    null
  );
}

function projectActivityTime(project: FactoryProject): number {
  const value = projectActivityAt(project);
  if (!value) return 0;
  const time = new Date(value).getTime();
  return Number.isNaN(time) ? 0 : time;
}

function projectMetaString(project: FactoryProject, keys: string[]): string | null {
  const meta = project.metadata ?? {};
  for (const key of keys) {
    const value = meta[key];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

function projectActualNotion(project: FactoryProject): { url: string; label: string } | null {
  const tracker = projectMetaString(project, [
    "notion_tracker_url",
    "notion_project_url",
    "notion_page_url",
    "notion_url",
  ]);
  if (tracker) return { url: tracker, label: "Abrir Notion PM" };
  const pageId = projectMetaString(project, ["notion_tracker_page_id", "notion_project_page_id", "notion_page_id"]);
  if (pageId) return { url: `https://www.notion.so/${pageId.replace(/-/g, "")}`, label: "Abrir Notion PM" };
  return null;
}

function projectTemplateNotion(project: FactoryProject): { url: string; label: string } | null {
  const template = projectMetaString(project, ["notion_template_url"]);
  if (template) return { url: template, label: "Plantilla Factory" };
  const templateId = projectMetaString(project, ["notion_template_page_id"]);
  if (templateId) return { url: `https://www.notion.so/${templateId.replace(/-/g, "")}`, label: "Plantilla Factory" };
  return null;
}

async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      // Fall through to the DOM fallback below. Some browser/dashboard binds
      // expose navigator.clipboard but deny writeText outside secure contexts.
    }
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "true");
    textarea.style.position = "fixed";
    textarea.style.left = "-9999px";
    textarea.style.top = "0";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    const copied = document.execCommand("copy");
    document.body.removeChild(textarea);
    return copied;
  } catch {
    return false;
  }
}

function projectSearchBlob(project: FactoryProject): string {
  const notion = projectActualNotion(project)?.url ?? "";
  const template = projectTemplateNotion(project)?.url ?? "";
  return [
    project.name,
    project.project_id,
    project.status,
    project.methodology,
    project.risk_level,
    project.summary,
    project.repo_path,
    notion,
    template,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

function projectStatusPrompt(project: FactoryProject): string {
  const dashboard = project.dashboard;
  const notion = projectActualNotion(project);
  const template = projectTemplateNotion(project);
  return [
    `Proyecto Factory: ${project.name} (${project.project_id})`,
    `Estado: ${project.status} (${statusInfo(project.status).description}) · metodología: ${project.methodology} · riesgo: ${project.risk_level}`,
    `Inicio: ${formatDate(project.started_at)} · última actividad: ${formatDate(projectActivityAt(project))}`,
    dashboard?.quick_status ? `Status rápido: ${dashboard.quick_status}` : "Status rápido no disponible.",
    dashboard?.current_task
      ? `Tarea actual/relevante: ${dashboard.current_task.title} [${dashboard.current_task.status}]`
      : "Tarea actual/relevante: —",
    dashboard?.findings?.length
      ? `Anomalías: ${dashboard.findings.map((f) => `${f.code}(${f.count ?? 1})`).join(", ")}`
      : "Anomalías: ninguna detectada por Factory DB.",
    project.repo_path ? `Repo: ${project.repo_path}` : "Repo: —",
    notion ? `${notion.label}: ${notion.url}` : "Notion PM: ANOMALÍA — falta página propia del proyecto.",
    template ? `${template.label}: ${template.url}` : "Plantilla Factory: —",
    "Pide a Zeus que triangule DB + repo + GitHub/CI + Notion antes de responder si el proyecto es crítico.",
  ].join("\n");
}

export default function FactoryPage() {
  const [data, setData] = useState<FactoryDashboardResponse | null>(null);
  const [selectedProjectId, setSelectedProjectId] = useState<string>("");
  const [showAllProjects, setShowAllProjects] = useState(false);
  const [projectSearch, setProjectSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [notionBusyProjectId, setNotionBusyProjectId] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [lastLoadedAt, setLastLoadedAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const navigate = useNavigate();
  const { toast, showToast } = useToast();
  const { setEnd, setTitle } = usePageHeader();

  const load = useCallback(async (options?: { force?: boolean }) => {
    setRefreshing(true);
    try {
      const payload = await api.getFactoryDashboard(undefined, options);
      setData(payload);
      setLastLoadedAt(Date.now());
      setSelectedProjectId((current) => {
        if (current && payload.projects.some((project) => project.project_id === current)) {
          return current;
        }
        return payload.projects[0]?.project_id ?? "";
      });
    } catch (error) {
      showToast(`Factory status error: ${error}`, "error");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [showToast]);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void load({ force: false });
    }, FACTORY_AUTO_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [load]);

  useLayoutEffect(() => {
    setTitle("Software Factory");
    setEnd(
      <div className="flex items-center gap-2">
        <span className="hidden text-xs text-muted-foreground sm:inline">
          Auto {Math.round(FACTORY_AUTO_REFRESH_MS / 1000)}s · último {lastLoadedAt ? formatRelativeAge(new Date(lastLoadedAt).toISOString(), now) : "—"}
        </span>
        <Button
          size="sm"
          ghost
          onClick={() => void load({ force: true })}
          disabled={refreshing}
          prefix={refreshing ? <Spinner /> : <RefreshCw className="h-4 w-4" />}
        >
          Forzar status
        </Button>
      </div>,
    );
    return () => {
      setTitle(null);
      setEnd(null);
    };
  }, [lastLoadedAt, load, now, refreshing, setEnd, setTitle]);

  const sortedProjects = useMemo(() => {
    if (!data) return [];
    return [...data.projects].sort((a, b) => {
      const byActivity = projectActivityTime(b) - projectActivityTime(a);
      if (byActivity !== 0) return byActivity;
      return a.name.localeCompare(b.name);
    });
  }, [data]);

  const statusOptions = useMemo(() => {
    const options = new Set<string>();
    for (const project of sortedProjects) {
      if (project.status) options.add(project.status);
    }
    return Array.from(options).sort((a, b) => statusInfo(a).label.localeCompare(statusInfo(b).label));
  }, [sortedProjects]);

  const filteredProjects = useMemo(() => {
    const query = projectSearch.trim().toLowerCase();
    return sortedProjects.filter((project) => {
      if (statusFilter !== "all" && project.status !== statusFilter) return false;
      if (!query) return true;
      return projectSearchBlob(project).includes(query);
    });
  }, [projectSearch, sortedProjects, statusFilter]);

  const visibleProjects = useMemo(
    () => (showAllProjects ? filteredProjects : filteredProjects.slice(0, PROJECT_PREVIEW_LIMIT)),
    [filteredProjects, showAllProjects],
  );

  const hiddenProjectCount = Math.max(filteredProjects.length - visibleProjects.length, 0);

  const selectedProject = useMemo(() => {
    if (!data) return null;
    return (
      data.projects.find((project) => project.project_id === selectedProjectId) ??
      sortedProjects[0] ??
      null
    );
  }, [data, selectedProjectId, sortedProjects]);

  const selectedTasks = useMemo(() => {
    if (!data || !selectedProject) return [];
    return data.tasks.filter((task) => task.project_id === selectedProject.project_id);
  }, [data, selectedProject]);

  const selectedGates = useMemo(() => {
    if (!data || !selectedProject) return [];
    return data.gates.filter((gate) => gate.project_id === selectedProject.project_id);
  }, [data, selectedProject]);

  const selectedLanes = useMemo(() => {
    if (!data || !selectedProject) return [];
    return data.lanes.filter((lane) => lane.project_id === selectedProject.project_id);
  }, [data, selectedProject]);

  const copyQuickStatus = useCallback(async () => {
    if (!selectedProject) return;
    const copied = await copyText(projectStatusPrompt(selectedProject));
    showToast(copied ? "Status rápido copiado" : "No se pudo copiar el status", copied ? "success" : "error");
  }, [selectedProject, showToast]);

  const resumeProjectInChat = useCallback(() => {
    if (!selectedProject) return;
    const qs = new URLSearchParams({
      factory_project: selectedProject.project_id,
      handoff: Date.now().toString(36),
    });
    navigate(`/chat?${qs.toString()}`);
  }, [navigate, selectedProject]);

  const generateNotion = useCallback(async () => {
    if (!selectedProject) return;
    setNotionBusyProjectId(selectedProject.project_id);
    try {
      const result = await api.createFactoryProjectNotion(selectedProject.project_id);
      showToast(result.created ? "Página Notion generada" : "El proyecto ya tenía Notion", "success");
      await load();
      if (result.url) {
        window.open(result.url, "_blank", "noopener,noreferrer");
      }
    } catch (error) {
      showToast(`No se pudo generar Notion: ${error}`, "error");
    } finally {
      setNotionBusyProjectId(null);
    }
  }, [load, selectedProject, showToast]);

  const syncNotion = useCallback(async () => {
    if (!selectedProject) return;
    setNotionBusyProjectId(selectedProject.project_id);
    try {
      const result = await api.syncFactoryProjectNotion(selectedProject.project_id);
      showToast(result.created ? "Notion generado y sincronizado" : "Notion sincronizado con Factory DB", "success");
      await load();
      if (result.url) {
        window.open(result.url, "_blank", "noopener,noreferrer");
      }
    } catch (error) {
      showToast(`No se pudo sincronizar Notion: ${error}`, "error");
    } finally {
      setNotionBusyProjectId(null);
    }
  }, [load, selectedProject, showToast]);

  const runProjectAction = useCallback(async (action: FactoryProjectAction) => {
    if (!selectedProject) return;
    const busyKey = `${selectedProject.project_id}:${action}`;
    setActionBusy(busyKey);
    try {
      const result = await api.runFactoryProjectAction(selectedProject.project_id, action);
      const claimed = result.claimed && typeof result.claimed === "object" ? " · worker encolado" : "";
      showToast(`Factory ${action}: ${result.status ?? "ok"}${claimed}`, "success");
      await load();
    } catch (error) {
      showToast(`Factory ${action} falló: ${error}`, "error");
    } finally {
      setActionBusy(null);
    }
  }, [load, selectedProject, showToast]);

  const selectedActualNotion = selectedProject ? projectActualNotion(selectedProject) : null;
  const selectedMissingNotion = Boolean(selectedProject && !selectedActualNotion);
  const notionBusy = Boolean(selectedProject && notionBusyProjectId === selectedProject.project_id);
  const selectedEffectiveGates = selectedProject?.dashboard?.effective_gates ?? [];
  const selectedBlockedTasks = selectedProject?.dashboard?.blocked_tasks ?? [];

  if (loading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center text-muted-foreground">
        <Spinner />
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex flex-col gap-4">
        <Toast toast={toast} />
        <Card>
          <CardContent className="py-8 text-center text-sm text-muted-foreground">
            No se pudo cargar la Factory DB.
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <Toast toast={toast} />

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard icon={<Database />} label="Proyectos" value={data.counts.projects ?? data.projects.length} />
        <MetricCard icon={<GitBranch />} label="Lanes" value={data.counts.lanes ?? data.lanes.length} />
        <MetricCard icon={<ShieldCheck />} label="Gates" value={data.counts.gates ?? data.gates.length} />
        <MetricCard icon={<AlertTriangle />} label="Anomalías" value={data.counts.findings ?? data.findings.length} tone={data.findings.length ? "warning" : "success"} />
      </section>

      <section className="grid gap-4 xl:grid-cols-[minmax(260px,0.8fr)_minmax(0,1.2fr)]">
        <Card className="min-w-0 overflow-hidden">
          <CardHeader className="pb-3">
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <CardTitle className="text-sm">
                  {showAllProjects ? "Todos los proyectos" : "Modificados más recientes"}
                </CardTitle>
                <p className="mt-1 text-xs text-muted-foreground">
                  Vista rápida por última actividad Factory. Selecciona uno para cargar detalles.
                </p>
              </div>
              <Badge tone="outline" className="max-w-[180px] truncate text-xs" title={data.db_path}>
                {data.db_path}
              </Badge>
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-[minmax(0,1fr)_180px]">
              <div className={cn("relative", !showAllProjects && "opacity-70")}>
                <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={projectSearch}
                  onChange={(event) => setProjectSearch(event.target.value)}
                  placeholder={showAllProjects ? "Buscar por nombre, id, metodología, repo o Notion…" : "Activa “ver todos” para buscar"}
                  className="h-8 pl-7 text-xs"
                  disabled={!showAllProjects}
                />
              </div>
              <Select id="factory-status-filter" value={statusFilter} onValueChange={setStatusFilter}>
                <SelectOption value="all">Todos los status</SelectOption>
                {statusOptions.map((status) => (
                  <SelectOption key={status} value={status}>
                    {statusInfo(status).label}
                  </SelectOption>
                ))}
              </Select>
            </div>
          </CardHeader>
          <CardContent className="grid gap-3">
            {visibleProjects.length === 0 ? (
              <p className="py-6 text-center text-sm text-muted-foreground">
                No hay proyectos que coincidan con el filtro.
              </p>
            ) : (
              visibleProjects.map((project) => {
                const dash = project.dashboard;
                const selected = project.project_id === selectedProject?.project_id;
                const status = statusInfo(project.status);
                const notion = projectActualNotion(project);
                return (
                  <button
                    key={project.project_id}
                    type="button"
                    onClick={() => setSelectedProjectId(project.project_id)}
                    className={cn(
                      "w-full border border-border bg-background/30 p-3 text-left transition hover:border-primary/40 hover:bg-background/50",
                      selected && "border-primary/60 bg-primary/5",
                    )}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0">
                        <p className="truncate text-sm font-semibold">{project.name}</p>
                        <p className="truncate text-xs text-muted-foreground">{project.project_id}</p>
                      </div>
                      <Badge tone={statusTone(project.status)} title={status.description}>{status.label}</Badge>
                    </div>
                    <p className="mt-2 text-[11px] leading-4 text-muted-foreground">
                      Actividad: {formatShortDate(projectActivityAt(project))} · Inicio: {formatShortDate(project.started_at)}
                    </p>
                    <p className="mt-1 text-[11px] leading-4 text-muted-foreground">{status.description}</p>
                    <div className="mt-3 flex flex-wrap gap-2 text-xs">
                      <Badge tone="outline">{project.methodology}</Badge>
                      <Badge tone={project.risk_level === "critical" ? "destructive" : "secondary"}>
                        {project.risk_level}
                      </Badge>
                      {notion ? <Badge tone="outline">Notion PM</Badge> : <Badge tone="warning">sin Notion PM</Badge>}
                      {dash?.open_task_count ? (
                        <Badge tone="warning">open {dash.open_task_count}</Badge>
                      ) : (
                        <Badge tone="success">sin open tasks</Badge>
                      )}
                    </div>
                  </button>
                );
              })
            )}
            <div className="flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-2 text-xs text-muted-foreground">
              <span>
                Mostrando {visibleProjects.length}/{filteredProjects.length} proyecto(s)
                {hiddenProjectCount ? ` · ${hiddenProjectCount} más ocultos` : ""}
              </span>
              <button
                type="button"
                onClick={() => setShowAllProjects((value) => !value)}
                className="font-medium text-primary underline-offset-4 hover:underline"
              >
                {showAllProjects ? "Volver a recientes" : "Ver todos y buscar"}
              </button>
            </div>
          </CardContent>
        </Card>

        <Card className="min-w-0 overflow-hidden">
          <CardHeader className="pb-3">
            <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
              <div>
                <CardTitle className="text-sm">Status rápido AI-ready</CardTitle>
                <p className="mt-1 text-xs text-muted-foreground">
                  Snapshot generado desde Factory DB. Úsalo como contexto para Zeus/PM.
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Select
                  id="factory-project-select"
                  value={selectedProject?.project_id ?? ""}
                  onValueChange={(value) => setSelectedProjectId(value)}
                >
                  {sortedProjects.map((project) => (
                    <SelectOption key={project.project_id} value={project.project_id}>
                      {project.name}
                    </SelectOption>
                  ))}
                </Select>
                <Button size="sm" ghost onClick={copyQuickStatus} prefix={<Clipboard className="h-4 w-4" />}>
                  Copiar status
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4">
            {selectedProject ? (
              <>
                <div className="rounded-md border border-border/80 bg-background/30 p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <H2 variant="sm" className="mr-auto text-muted-foreground">
                      {selectedProject.name}
                    </H2>
                    <Badge tone={statusTone(selectedProject.status)} title={statusInfo(selectedProject.status).description}>
                      {statusInfo(selectedProject.status).label}
                    </Badge>
                    <Badge tone="outline">{selectedProject.methodology}</Badge>
                    <Badge tone={selectedProject.risk_level === "critical" ? "destructive" : "secondary"}>
                      {selectedProject.risk_level}
                    </Badge>
                  </div>
                  <p className="mt-2 text-xs leading-5 text-muted-foreground">
                    {statusInfo(selectedProject.status).description}
                  </p>
                  <p className="mt-3 text-sm leading-6 text-text-secondary">
                    {selectedProject.dashboard?.quick_status ?? "Sin resumen disponible."}
                  </p>
                  {selectedProject.summary && (
                    <p className="mt-2 text-xs leading-5 text-muted-foreground">{selectedProject.summary}</p>
                  )}
                </div>

                <ProjectDetailsGrid project={selectedProject} />

                <FactoryWorkflowPanel project={selectedProject} now={now} refreshing={refreshing} />

                <ProjectControlPanel
                  actionBusy={actionBusy}
                  missingNotion={selectedMissingNotion}
                  notionBusy={notionBusy}
                  project={selectedProject}
                  onGenerateNotion={generateNotion}
                  onProjectAction={runProjectAction}
                  onResumeChat={resumeProjectInChat}
                  onSyncNotion={syncNotion}
                />

                <div className="grid gap-3 md:grid-cols-3">
                  <MiniStat label="Tasks" value={countLabel(selectedProject.dashboard?.task_counts)} />
                  <MiniStat label="Gates" value={countLabel(selectedProject.dashboard?.gate_counts)} />
                  <MiniStat
                    label="Docs factory/"
                    value={`${selectedProject.dashboard?.required_docs.filter((doc) => doc.exists).length ?? 0}/${selectedProject.dashboard?.required_docs.length ?? 0}`}
                  />
                </div>

                <Findings findings={selectedProject.dashboard?.findings ?? []} />
              </>
            ) : (
              <p className="text-sm text-muted-foreground">No hay proyectos Factory registrados.</p>
            )}
          </CardContent>
        </Card>
      </section>

      {selectedProject && (
        <section className="grid gap-4 xl:grid-cols-[minmax(0,1.35fr)_minmax(320px,0.65fr)]">
          <Card className="min-w-0 overflow-hidden">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Tareas Factory DB</CardTitle>
            </CardHeader>
            <CardContent>
              <TasksTable tasks={selectedTasks} />
            </CardContent>
          </Card>

          <div className="grid gap-4">
            <Card className="overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Lanes metodológicas</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-3 text-xs">
                {selectedLanes.length ? (
                  selectedLanes.map((lane) => (
                    <div key={lane.lane_id} className="border border-border/80 bg-background/30 p-3">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium">{lane.name}</span>
                        <Badge tone={statusTone(lane.status)}>{lane.status}</Badge>
                      </div>
                      <p className="mt-2 text-muted-foreground">{lane.methodology}</p>
                      <p className="mt-1 break-all text-muted-foreground">branch: {lane.branch ?? "—"}</p>
                      <p className="mt-1 break-all text-muted-foreground">worktree: {lane.worktree_path ?? "—"}</p>
                    </div>
                  ))
                ) : (
                  <p className="text-muted-foreground">Sin lanes registradas.</p>
                )}
              </CardContent>
            </Card>

            <Card className="overflow-hidden">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Documentos requeridos</CardTitle>
              </CardHeader>
              <CardContent className="grid gap-2 text-xs">
                {(selectedProject.dashboard?.required_docs ?? []).map((doc) => (
                  <div key={doc.name} className="flex items-center justify-between gap-3 border-b border-border/50 pb-2 last:border-b-0 last:pb-0">
                    <span className="flex min-w-0 items-center gap-2">
                      <FileText className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
                      {doc.url ? (
                        <a
                          href={doc.url}
                          target="_blank"
                          rel="noreferrer"
                          className="truncate text-primary underline-offset-4 hover:underline"
                          title={doc.url}
                        >
                          {doc.name}
                        </a>
                      ) : (
                        <span className="truncate" title={doc.path}>{doc.name}</span>
                      )}
                    </span>
                    <Badge tone={doc.exists ? "success" : "warning"}>
                      {doc.exists ? `${Math.round(doc.size / 1024)}KB` : "missing"}
                    </Badge>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        </section>
      )}

      <section>
        <Card className="overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Agentes especializados sembrados</CardTitle>
          </CardHeader>
          <CardContent className="grid gap-2 text-xs">
            {data.agents.map((agent) => (
              <div key={agent.agent_id} className="flex items-start gap-3 border-b border-border/50 pb-2 last:border-b-0 last:pb-0">
                <Bot className="mt-0.5 h-4 w-4 text-muted-foreground" />
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-medium">{agent.display_name}</span>
                    <Badge tone={agent.active ? "success" : "outline"}>{agent.preferred_engine ?? "engine —"}</Badge>
                  </div>
                  <p className="mt-1 text-muted-foreground">{agent.role}</p>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </section>

      {selectedEffectiveGates.length ? (
        <Card className="overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Gates efectivos — decisión vigente por tipo</CardTitle>
          </CardHeader>
          <CardContent>
            <GatesTable gates={selectedEffectiveGates} />
          </CardContent>
        </Card>
      ) : null}

      {selectedBlockedTasks.length ? (
        <Card className="overflow-hidden border-destructive/30 bg-destructive/5">
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-sm">
              <AlertTriangle className="h-4 w-4 text-destructive" />
              Bloqueos activos detectados
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p className="text-xs leading-5 text-muted-foreground">
              Zeus debe resolverlos desde Factory DB: audita gates efectivos, causa raíz y reabre solo si el bloqueo ya quedó resuelto.
            </p>
            <TasksTable tasks={selectedBlockedTasks.slice(0, 6)} />
          </CardContent>
        </Card>
      ) : null}

      {selectedGates.length > 0 && (
        <Card className="overflow-hidden">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Auditoría histórica de gates del proyecto seleccionado</CardTitle>
          </CardHeader>
          <CardContent>
            <GatesTable gates={selectedGates.slice(0, 20)} />
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function MetricCard({
  icon,
  label,
  tone = "secondary",
  value,
}: {
  icon: ReactNode;
  label: string;
  tone?: BadgeTone;
  value: number | string;
}) {
  return (
    <Card className="overflow-hidden">
      <CardContent className="flex items-center gap-3 p-4">
        <div className="flex h-10 w-10 items-center justify-center rounded-full border border-border/80 text-muted-foreground">
          {icon}
        </div>
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
          <Badge tone={tone} className="mt-1 text-sm">
            {value}
          </Badge>
        </div>
      </CardContent>
    </Card>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-border/80 bg-background/30 p-3">
      <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">{label}</p>
      <p className="mt-2 text-sm text-text-secondary">{value}</p>
    </div>
  );
}

function ProjectDetailsGrid({ project }: { project: FactoryProject }) {
  const notion = projectActualNotion(project);
  const template = projectTemplateNotion(project);
  return (
    <div className="grid gap-3 md:grid-cols-3">
      <MiniStat label="Inicio" value={formatDate(project.started_at)} />
      <MiniStat label="Última actividad" value={formatDate(projectActivityAt(project))} />
      <div className={cn("rounded-md border p-3", notion ? "border-border/80 bg-background/30" : "border-warning/40 bg-warning/5")}>
        <p className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Notion PM</p>
        {notion ? (
          <a
            href={notion.url}
            target="_blank"
            rel="noreferrer"
            className="mt-2 inline-flex max-w-full items-center gap-1 break-all text-sm font-medium text-primary underline-offset-4 hover:underline"
          >
            {notion.label}
            <ExternalLink className="h-3.5 w-3.5 shrink-0" />
          </a>
        ) : (
          <div className="mt-2 grid gap-1 text-sm">
            <p className="font-medium text-warning">Anomalía: falta página Notion propia.</p>
            <p className="text-xs leading-5 text-muted-foreground">
              El proyecto solo está en Factory DB o tiene plantilla; debe documentarse con la plantilla Factory.
            </p>
            {template ? (
              <a
                href={template.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex max-w-full items-center gap-1 break-all text-xs font-medium text-primary underline-offset-4 hover:underline"
              >
                Ver plantilla base
                <ExternalLink className="h-3.5 w-3.5 shrink-0" />
              </a>
            ) : null}
          </div>
        )}
      </div>
    </div>
  );
}

function FactoryWorkflowPanel({ project, now, refreshing }: { project: FactoryProject; now: number; refreshing: boolean }) {
  const workflow = project.dashboard?.workflow;
  const activeRun = project.dashboard?.active_run;
  const staticProject = isStaticFactoryProjectStatus(project.status);
  const showLiveAnimation = !staticProject && (activeRunIsLive(project) || Boolean(workflow?.operative));
  const heartbeatAt = workflow?.heartbeat_at ?? activeRun?.heartbeat_at;
  const workingText = refreshing
    ? "actualizando snapshot"
    : THINKING_MESSAGES[Math.floor(now / 4000) % THINKING_MESSAGES.length];
  const statusText = showLiveAnimation ? `${workingText} · contador ${Math.floor(now / 1000) % 60}s` : workflowIdleCopy(project);
  const heartbeatAge = heartbeatAt ? formatRelativeAge(heartbeatAt, now) : "sin heartbeat";
  const stages = [
    { id: "planned", label: "Plan" },
    { id: "running", label: "Worker" },
    { id: "review", label: "QA/Gates" },
    { id: "completed", label: "Cierre" },
  ];
  const activeIndex = Math.max(0, stages.findIndex((stage) => stage.id === workflow?.stage));
  return (
    <div className="rounded-md border border-border/80 bg-background/30 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Activity className="h-4 w-4 text-primary" />
        <p className="text-sm font-semibold">Ciclo productivo Factory</p>
        <Badge tone={showLiveAnimation ? "success" : staticProject ? statusTone(project.status) : "secondary"}>
          {showLiveAnimation ? "operativo" : "sin ejecución activa"}
        </Badge>
        {workflow?.single_active_increment ? <Badge tone="outline">1 incremento activo</Badge> : null}
        <span className="ml-auto inline-flex items-center gap-2 text-xs text-muted-foreground">
          {showLiveAnimation ? <Spinner /> : <PauseCircle className="h-3.5 w-3.5" />}
          {statusText}
        </span>
      </div>
      <div className="mt-4 grid gap-2 md:grid-cols-4">
        {stages.map((stage, index) => {
          const active = index <= activeIndex;
          return (
            <div key={stage.id} className={cn("rounded-md border p-3 text-xs", active ? "border-primary/40 bg-primary/5" : "border-border/60 bg-background/20")}>
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium">{stage.label}</span>
                <Badge tone={index === activeIndex ? statusTone(workflow?.stage) : "outline"}>{index + 1}</Badge>
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-3 grid gap-2 text-xs text-muted-foreground md:grid-cols-3">
        <span>Worker: {workflow?.worker ?? activeRun?.worker_profile ?? "—"}</span>
        <span>Tarea: {workflow?.current_task_id ?? activeRun?.task_id ?? project.dashboard?.current_task?.task_id ?? "—"}</span>
        <span>Heartbeat: {formatDate(heartbeatAt)} ({heartbeatAge})</span>
      </div>
    </div>
  );
}

interface ProjectControlPanelProps {
  actionBusy: string | null;
  missingNotion: boolean;
  notionBusy: boolean;
  project: FactoryProject;
  onGenerateNotion: () => void;
  onProjectAction: (action: FactoryProjectAction) => void;
  onResumeChat: () => void;
  onSyncNotion: () => void;
}

function ProjectControlPanel({
  actionBusy,
  missingNotion,
  notionBusy,
  project,
  onGenerateNotion,
  onProjectAction,
  onResumeChat,
  onSyncNotion,
}: ProjectControlPanelProps) {
  const isBusy = (action: FactoryProjectAction) => actionBusy === `${project.project_id}:${action}`;
  const actionDisabled = Boolean(actionBusy);
  return (
    <div className="rounded-md border border-border/80 bg-card/50 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Workflow className="h-4 w-4 text-primary" />
        <p className="mr-auto text-sm font-semibold">Panel de control del proyecto</p>
        <Badge tone="outline">acciones determinísticas</Badge>
      </div>
      <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
        <Button size="sm" ghost onClick={() => onProjectAction("resume")} disabled={actionDisabled} prefix={isBusy("resume") ? <Spinner /> : <PlayCircle className="h-4 w-4" />}>
          Poner autónomo
        </Button>
        <Button size="sm" ghost onClick={() => onProjectAction("pause")} disabled={actionDisabled} prefix={isBusy("pause") ? <Spinner /> : <PauseCircle className="h-4 w-4" />}>
          Pausar fábrica
        </Button>
        <Button size="sm" ghost onClick={() => onProjectAction("tick")} disabled={actionDisabled} prefix={isBusy("tick") ? <Spinner /> : <Zap className="h-4 w-4" />}>
          Ejecutar tick
        </Button>
        <Button size="sm" ghost onClick={() => onProjectAction("reconcile")} disabled={actionDisabled} prefix={isBusy("reconcile") ? <Spinner /> : <RotateCcw className="h-4 w-4" />}>
          Reconciliar estado
        </Button>
        <Button size="sm" ghost onClick={() => onProjectAction("unblock")} disabled={actionDisabled} prefix={isBusy("unblock") ? <Spinner /> : <ShieldCheck className="h-4 w-4" />}>
          Resolver bloqueos
        </Button>
        <Button size="sm" ghost onClick={missingNotion ? onGenerateNotion : onSyncNotion} disabled={notionBusy} prefix={notionBusy ? <Spinner /> : <FilePlus2 className="h-4 w-4" />}>
          {missingNotion ? "Generar Notion" : "Actualizar Notion"}
        </Button>
        <Button size="sm" ghost onClick={onResumeChat} prefix={<MessageSquare className="h-4 w-4" />}>
          Abrir chat con contexto
        </Button>
      </div>
      <p className="mt-3 text-xs leading-5 text-muted-foreground">
        “Abrir chat” solo entrega contexto a Zeus. La ejecución autónoma se controla con Factory DB: poner autónomo, tick, reconciliar, resolver bloqueos canónicos y pausar.
      </p>
    </div>
  );
}

function Findings({ findings }: { findings: FactoryFinding[] }) {
  if (findings.length === 0) {
    return (
      <div className="rounded-md border border-success/30 bg-success/5 p-3 text-sm text-text-secondary">
        No hay anomalías detectadas por Factory DB para este proyecto.
      </div>
    );
  }
  return (
    <div className="grid gap-2">
      {findings.map((finding) => (
        <div key={`${finding.code}-${finding.title}`} className="rounded-md border border-border/80 bg-background/30 p-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={findingTone(finding)}>{finding.code}</Badge>
            <span className="text-sm font-medium">{finding.title}</span>
            {finding.count ? <Badge tone="outline">{finding.count}</Badge> : null}
          </div>
          <p className="mt-2 text-xs leading-5 text-muted-foreground">{finding.message}</p>
        </div>
      ))}
    </div>
  );
}

function TasksTable({ tasks }: { tasks: FactoryTask[] }) {
  if (tasks.length === 0) {
    return <p className="py-6 text-center text-sm text-muted-foreground">Sin tareas registradas.</p>;
  }
  return (
    <div className="max-h-[520px] overflow-auto border border-border/80">
      <table className="min-w-full divide-y divide-border/80 text-left text-xs">
        <thead className="sticky top-0 bg-card text-muted-foreground">
          <tr>
            <th className="px-3 py-2 font-medium">Tarea</th>
            <th className="px-3 py-2 font-medium">Estado</th>
            <th className="px-3 py-2 font-medium">Owner</th>
            <th className="px-3 py-2 font-medium">Reviewer</th>
            <th className="px-3 py-2 font-medium">Actualizada</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/50">
          {tasks.map((task) => (
            <tr key={task.task_id} className="align-top hover:bg-background/30">
              <td className="max-w-[360px] px-3 py-2">
                <p className="font-medium text-text-secondary">{task.title}</p>
                <p className="mt-1 text-muted-foreground">{task.phase} · {task.engine}</p>
              </td>
              <td className="px-3 py-2"><Badge tone={statusTone(task.status)}>{task.status}</Badge></td>
              <td className="px-3 py-2 text-muted-foreground">{task.owner_agent_id ?? "—"}</td>
              <td className="px-3 py-2 text-muted-foreground">{task.reviewer_agent_id ?? "—"}</td>
              <td className="px-3 py-2 text-muted-foreground">{formatDate(task.updated_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function GatesTable({ gates }: { gates: FactoryGate[] }) {
  return (
    <div className="overflow-auto border border-border/80">
      <table className="min-w-full divide-y divide-border/80 text-left text-xs">
        <thead className="bg-card text-muted-foreground">
          <tr>
            <th className="px-3 py-2 font-medium">Gate</th>
            <th className="px-3 py-2 font-medium">Estado</th>
            <th className="px-3 py-2 font-medium">Reviewer</th>
            <th className="px-3 py-2 font-medium">Notas</th>
            <th className="px-3 py-2 font-medium">Fecha</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border/50">
          {gates.map((gate) => (
            <tr key={gate.gate_id} className="align-top hover:bg-background/30">
              <td className="px-3 py-2 font-medium text-text-secondary">{gate.gate_type}</td>
              <td className="px-3 py-2"><Badge tone={statusTone(gate.status)}>{gate.status}</Badge></td>
              <td className="px-3 py-2 text-muted-foreground">{gate.reviewer ?? "—"}</td>
              <td className="max-w-[520px] px-3 py-2 text-muted-foreground" title={gate.notes ?? undefined}>{truncateText(gate.notes)}</td>
              <td className="px-3 py-2 text-muted-foreground">{formatDate(gate.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
