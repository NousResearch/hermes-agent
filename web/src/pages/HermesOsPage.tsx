import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, CheckCircle2, GalleryVerticalEnd, GitBranch, RefreshCw, ShieldCheck } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  ChartPanel,
  DashboardEmptyState,
  DashboardErrorState,
  DashboardHeader,
  DashboardLoadingState,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  HealthBadge,
  InsightPanel,
  KpiCard,
  MetricGrid,
  SimpleBarChart,
  StatusPill,
  type DataTableColumn,
  type DashboardTone,
} from "@hermes/dashboard-kit";
import { api } from "@/lib/api";
import type { HermesOsPanel, HermesOsSummary } from "@/lib/api";

function panel(summary: HermesOsSummary | null, id: string): HermesOsPanel | null {
  return summary?.panels.find((item) => item.panel_id === id) ?? null;
}

function asList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function toneForStatus(status: unknown): DashboardTone {
  const text = String(status ?? "").toLowerCase();
  if (["ready", "healthy", "complete", "completed", "success", "clear"].includes(text)) return "success";
  if (["blocked", "failed", "critical", "error"].includes(text)) return "critical";
  if (["warning", "pending", "needs_review"].includes(text)) return "warning";
  if (text) return "info";
  return "unknown";
}

function compactValue(value: unknown): string {
  if (Array.isArray(value)) return String(value.length);
  if (typeof value === "object" && value) return "configured";
  return String(value ?? "");
}

interface TaskRow {
  id: string;
  title: string;
  status: string;
}

interface ModuleRow {
  key: string;
  value: string;
}

export default function HermesOsPage() {
  const [summary, setSummary] = useState<HermesOsSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = () => {
    setLoading(true);
    setError("");
    api
      .getHermesOsSummary()
      .then(setSummary)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)))
      .finally(() => setLoading(false));
  };

  useEffect(load, []);

  const score = panel(summary, "architecture-score");
  const gaps = panel(summary, "architecture-gaps");
  const graph = panel(summary, "work-graph-summary");
  const runtime = panel(summary, "runtime-delegation");
  const assignments = panel(summary, "agent-assignments");
  const tasks = panel(summary, "task-backlog");
  const templates = panel(summary, "templates");
  const dryRun = panel(summary, "dry-run-execution");
  const scoreData = score?.data ?? {};
  const gapData = gaps?.data ?? {};
  const graphData = graph?.data ?? {};
  const runtimeData = runtime?.data ?? {};
  const assignmentData = assignments?.data ?? {};
  const taskData = tasks?.data ?? {};
  const templateData = templates?.data ?? {};
  const dryRunData = dryRun?.data ?? {};
  const runtimeModules = [
    "project-runtime-services",
    "workspace-snapshots",
    "snapshot-restore-preview",
    "agent-trace-timeline",
    "agent-message-detail",
    "runtime-cost-budget",
    "runtime-approval-queue",
    "infrastructure-registry",
    "vector-registry",
    "template-packs",
    "activity-feed",
    "col-active-context",
    "col-chief-of-staff",
    "col-workflow-preview",
    "col-agent-hierarchy",
  ].map((id) => panel(summary, id)).filter(Boolean) as HermesOsPanel[];

  const byAgent = useMemo(
    () => Object.entries(asRecord(assignmentData.assignments_by_agent)),
    [assignmentData.assignments_by_agent],
  );

  const taskRows = useMemo<TaskRow[]>(() => {
    if (!Array.isArray(taskData.tasks)) return [];
    return taskData.tasks.slice(0, 12).map((task) => {
      const row = asRecord(task);
      return {
        id: String(row.id ?? ""),
        title: String(row.title ?? ""),
        status: String(row.status ?? "unknown"),
      };
    });
  }, [taskData.tasks]);

  const taskColumns = useMemo<DataTableColumn<TaskRow>[]>(() => [
    {
      id: "id",
      header: "ID",
      accessor: (row) => <span className="font-mono-ui text-xs text-muted-foreground">{row.id}</span>,
      sortValue: (row) => row.id,
      cellClassName: "w-40",
    },
    {
      id: "title",
      header: "Title",
      accessor: (row) => row.title,
      sortValue: (row) => row.title,
    },
    {
      id: "status",
      header: "Status",
      accessor: (row) => <StatusPill tone={toneForStatus(row.status)}>{row.status}</StatusPill>,
      sortValue: (row) => row.status,
      cellClassName: "w-32",
    },
  ], []);

  if (loading) {
    return <DashboardLoadingState label="Loading Hermes OS" className="min-h-[50vh]" />;
  }

  const sidebarItems = [
    { id: "overview", label: "Overview", href: "#overview", active: true, icon: ShieldCheck },
    { id: "backlog", label: "Task Backlog", href: "#backlog", icon: Activity },
    { id: "gaps", label: "Architecture Gaps", href: "#gaps", icon: AlertTriangle },
    { id: "agents", label: "Agent Assignments", href: "#agents", icon: GitBranch },
    { id: "design-system", label: "Design System", href: "/design-system", icon: GalleryVerticalEnd },
  ];

  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="Hermes OS"
          description="Reference dashboard kit implementation."
          items={sidebarItems}
          footer={<div className="text-xs text-muted-foreground">Built with `web/src/dashboard-kit`.</div>}
        />
      )}
      header={(
        <DashboardHeader
          title="Hermes OS"
          eyebrow="Operating layer"
          description={summary?.project_path ?? "No project loaded"}
          actions={(
            <>
              <a
                className="inline-flex h-9 items-center gap-2 rounded-md border border-border bg-background px-3 text-sm font-medium text-foreground transition hover:bg-muted"
                href="/design-system"
              >
                <GalleryVerticalEnd className="h-4 w-4" />
                Design System
              </a>
              <Button onClick={load}>
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>
            </>
          )}
          meta={(
            <HealthBadge
              label={scoreData.blocked ? "blocked" : "ready"}
              tone={scoreData.blocked ? "critical" : "success"}
            />
          )}
        />
      )}
    >
      {error ? <DashboardErrorState message={error} /> : null}

      <MetricGrid columns={4} className="scroll-mt-4" id="overview">
        <KpiCard
          label="Architecture Score"
          value={String(scoreData.score ?? "0")}
          tone={scoreData.blocked ? "critical" : "success"}
          icon={ShieldCheck}
          detail={scoreData.blocked ? "Blocked by architecture gaps" : "Ready for staged work"}
        />
        <KpiCard
          label="Work Graph"
          value={String(graphData.node_count ?? "0")}
          icon={GitBranch}
          detail={`${String(graphData.blocked_count ?? 0)} blocked, ${String(graphData.approval_count ?? 0)} approvals`}
        />
        <KpiCard
          label="Runtime"
          value={String(runtimeData.provider ?? "official-hermes-agent")}
          tone={runtimeData.available ? "success" : "warning"}
          icon={runtimeData.available ? CheckCircle2 : AlertTriangle}
          detail={String(runtimeData.mode ?? "dry_run")}
        />
        <KpiCard
          label="Assignments"
          value={String(graphData.assignment_count ?? "0")}
          icon={Activity}
          detail={`${String(assignmentData.fallback_count ?? 0)} fallback`}
        />
        <KpiCard
          label="Tasks"
          value={String(taskData.task_count ?? "0")}
          detail={`${String(taskData.blocked_count ?? 0)} blocked`}
        />
        <KpiCard
          label="Templates"
          value={String(templateData.template_count ?? "0")}
          detail={`${String(templateData.compile_failure_count ?? 0)} failures`}
        />
        <KpiCard
          label="Dry-run"
          value={String(dryRunData.batch_count ?? "0")}
          detail="execution batches"
        />
      </MetricGrid>

      <div className="grid gap-4 lg:grid-cols-2">
        <DashboardSection
          className="scroll-mt-4"
          title="Task Backlog"
          description="Generated architecture and runtime work that still needs attention."
          id="backlog"
        >
          <DataTable
            rows={taskRows}
            columns={taskColumns}
            getRowKey={(row) => row.id}
            emptyTitle="No tasks generated"
            emptyDescription="The operating layer has not produced backlog items yet."
          />
        </DashboardSection>

        <div id="gaps" className="scroll-mt-4">
        <InsightPanel title="Architecture Gaps" tone={scoreData.blocked ? "warning" : "success"}>
          {["missing_documents", "missing_schemas", "missing_dashboards", "missing_approvals"].map((key) => {
            const values = asList(gapData[key]);
            return (
              <div key={key}>
                <div className="mb-2 text-xs uppercase tracking-wide text-muted-foreground">{key.replaceAll("_", " ")}</div>
                <div className="flex flex-wrap gap-2">
                  {values.length
                    ? values.map((item) => <StatusPill key={item} tone="warning">{item}</StatusPill>)
                    : <StatusPill tone="success">clear</StatusPill>}
                </div>
              </div>
            );
          })}
        </InsightPanel>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartPanel title="Agent Assignments" description="Current work distribution by agent role." className="scroll-mt-4" id="agents">
          {byAgent.length ? (
            <SimpleBarChart
              data={byAgent.map(([agent, count]) => ({ label: agent, value: Number(count) || 0 }))}
              valueLabel="assignments"
            />
          ) : (
            <DashboardEmptyState title="No assignments" description="No agent assignments have been generated yet." />
          )}
        </ChartPanel>

        <DashboardSection title="Runtime Modules" description="Readiness summary for Hermes OS runtime surfaces.">
          <div className="grid gap-3 md:grid-cols-2">
            {runtimeModules.map((module) => {
              const rows: ModuleRow[] = Object.entries(asRecord(module.data)).slice(0, 5).map(([key, value]) => ({
                key: key.replaceAll("_", " "),
                value: compactValue(value),
              }));
              return (
                <div key={module.panel_id} className="rounded-lg border border-border bg-background p-3">
                  <div className="mb-3 flex items-center gap-2 text-sm font-medium text-foreground">
                    <Activity className="h-4 w-4" />
                    {module.title}
                  </div>
                  <div className="space-y-2 text-sm">
                    {rows.map((row) => (
                      <div key={row.key} className="flex items-start justify-between gap-3 border-b border-border pb-2 last:border-b-0 last:pb-0">
                        <span className="text-muted-foreground">{row.key}</span>
                        <span className="max-w-[60%] truncate text-right font-mono-ui text-xs text-foreground">{row.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </DashboardSection>
      </div>
    </DashboardShell>
  );
}
