import { useEffect, useMemo, useState } from "react";
import { Activity, AlertTriangle, CheckCircle2, GitBranch, ShieldCheck } from "lucide-react";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
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

  if (loading) {
    return (
      <div className="flex min-h-[50vh] items-center justify-center">
        <Spinner />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <H2>Hermes OS</H2>
          <p className="text-sm text-muted-foreground">{summary?.project_path ?? "No project loaded"}</p>
        </div>
        <Button onClick={load}>
          <Activity className="h-4 w-4" />
          Refresh
        </Button>
      </div>

      {error ? (
        <Card>
          <CardContent className="flex items-center gap-3 py-4 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4" />
            <span>{error}</span>
          </CardContent>
        </Card>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <Card>
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Architecture Score</span>
              <Badge tone={scoreData.blocked ? "destructive" : "success"}>
                {scoreData.blocked ? "blocked" : "ready"}
              </Badge>
            </div>
            <div className="mt-3 text-4xl font-semibold">{String(scoreData.score ?? "0")}</div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <span className="text-sm text-muted-foreground">Work Graph</span>
            <div className="mt-3 flex items-end gap-2">
              <span className="text-4xl font-semibold">{String(graphData.node_count ?? "0")}</span>
              <span className="pb-1 text-xs text-muted-foreground">nodes</span>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">
              {String(graphData.blocked_count ?? 0)} blocked, {String(graphData.approval_count ?? 0)} approvals
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <span className="text-sm text-muted-foreground">Runtime</span>
            <div className="mt-3 flex items-center gap-2">
              {runtimeData.available ? <CheckCircle2 className="h-5 w-5 text-success" /> : <AlertTriangle className="h-5 w-5 text-warning" />}
              <span className="font-medium">{String(runtimeData.provider ?? "official-hermes-agent")}</span>
            </div>
            <p className="mt-2 text-xs text-muted-foreground">{String(runtimeData.mode ?? "dry_run")}</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <span className="text-sm text-muted-foreground">Assignments</span>
            <div className="mt-3 text-4xl font-semibold">{String(graphData.assignment_count ?? "0")}</div>
            <p className="mt-2 text-xs text-muted-foreground">{String(assignmentData.fallback_count ?? 0)} fallback</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <span className="text-sm text-muted-foreground">Tasks</span>
            <div className="mt-3 text-4xl font-semibold">{String(taskData.task_count ?? "0")}</div>
            <p className="mt-2 text-xs text-muted-foreground">{String(taskData.blocked_count ?? 0)} blocked</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <span className="text-sm text-muted-foreground">Templates</span>
            <div className="mt-3 text-4xl font-semibold">{String(templateData.template_count ?? "0")}</div>
            <p className="mt-2 text-xs text-muted-foreground">{String(templateData.compile_failure_count ?? 0)} failures</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <span className="text-sm text-muted-foreground">Dry-run</span>
            <div className="mt-3 text-4xl font-semibold">{String(dryRunData.batch_count ?? "0")}</div>
            <p className="mt-2 text-xs text-muted-foreground">execution batches</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardContent className="py-4">
            <div className="mb-3 flex items-center gap-2 font-medium">
              <Activity className="h-4 w-4" />
              Task Backlog
            </div>
            <div className="space-y-2">
              {Array.isArray(taskData.tasks) && taskData.tasks.length
                ? taskData.tasks.slice(0, 12).map((task) => {
                    const row = asRecord(task);
                    return (
                      <div key={String(row.id)} className="flex items-start justify-between gap-3 border-b border-border pb-2 text-sm">
                        <div>
                          <div className="font-mono-ui text-xs text-muted-foreground">{String(row.id)}</div>
                          <div>{String(row.title)}</div>
                        </div>
                        <Badge tone={row.status === "blocked" ? "warning" : "success"}>{String(row.status)}</Badge>
                      </div>
                    );
                  })
                : <p className="text-sm text-muted-foreground">No tasks generated yet.</p>}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <div className="mb-3 flex items-center gap-2 font-medium">
              <ShieldCheck className="h-4 w-4" />
              Architecture Gaps
            </div>
            {["missing_documents", "missing_schemas", "missing_dashboards", "missing_approvals"].map((key) => (
              <div key={key} className="mb-3">
                <div className="mb-1 text-xs uppercase text-muted-foreground">{key.replaceAll("_", " ")}</div>
                <div className="flex flex-wrap gap-2">
                  {asList(gapData[key]).length
                    ? asList(gapData[key]).map((item) => <Badge key={item} tone="warning">{item}</Badge>)
                    : <Badge tone="success">clear</Badge>}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardContent className="py-4">
            <div className="mb-3 flex items-center gap-2 font-medium">
              <GitBranch className="h-4 w-4" />
              Agent Assignments
            </div>
            <div className="space-y-2">
              {byAgent.length ? byAgent.map(([agent, count]) => (
                <div key={agent} className="flex items-center justify-between border-b border-border pb-2 text-sm">
                  <span>{agent}</span>
                  <Badge>{String(count)}</Badge>
                </div>
              )) : <p className="text-sm text-muted-foreground">No assignments yet.</p>}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        {runtimeModules.map((module) => (
          <Card key={module.panel_id}>
            <CardContent className="py-4">
              <div className="mb-3 flex items-center gap-2 font-medium">
                <Activity className="h-4 w-4" />
                {module.title}
              </div>
              <div className="space-y-2 text-sm">
                {Object.entries(asRecord(module.data)).slice(0, 5).map(([key, value]) => (
                  <div key={key} className="flex items-start justify-between gap-3 border-b border-border pb-2">
                    <span className="text-muted-foreground">{key.replaceAll("_", " ")}</span>
                    <span className="max-w-[60%] truncate text-right font-mono-ui text-xs">
                      {Array.isArray(value) ? String(value.length) : typeof value === "object" && value ? "configured" : String(value ?? "")}
                    </span>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
}
