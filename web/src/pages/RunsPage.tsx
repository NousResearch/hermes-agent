import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation } from "react-router-dom";
import { AlertTriangle, ChevronDown, ChevronUp, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type {
  AgentEffectivenessEntry,
  AgentRunExecutorSchedulerStatus,
  AgentRunEntry,
  RunEntry,
  RunProject,
  RunTaskSummary,
  RunsSummaryResponse,
  RunsWatchdogResponse,
} from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import { isoTimeAgo } from "@/lib/utils";

const LIMIT = 50;
const CLASSIFICATIONS = ["ok", "timeout", "process_error", "permission_error", "auth_error", "rate_limited"];
const TASK_STATUSES = ["ok", "running", "stale", "timeout", "failed", "unknown"];
const HANDOFF_STATUSES = ["queued", "running", "completed", "failed", "cancelled"];

function classificationTone(value?: string | null): "success" | "warning" | "secondary" | "destructive" {
  if (value === "ok") return "success";
  if (value === "timeout" || value === "rate_limited") return "warning";
  if (value?.includes("error")) return "destructive";
  return "secondary";
}

function taskStatusTone(value?: string | null): "success" | "warning" | "secondary" | "destructive" {
  if (value === "ok") return "success";
  if (value === "running") return "secondary";
  if (value === "stale" || value === "timeout") return "warning";
  if (value === "failed") return "destructive";
  return "secondary";
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "-";
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`;
}

function startedLabel(value?: string | null): string {
  if (!value) return "-";
  return isoTimeAgo(value);
}

function taskStatusLabel(t: ReturnType<typeof useI18n>["t"], status?: string | null): string {
  if (status === "ok") return t.runs.ok;
  if (status === "running") return t.runs.running;
  if (status === "stale") return t.runs.stale;
  if (status === "timeout") return t.runs.timeout;
  if (status === "failed") return t.runs.failed;
  return t.runs.unknown;
}

function runStatusTone(value?: string | null): "success" | "warning" | "secondary" | "destructive" {
  if (value === "completed") return "success";
  if (value === "queued" || value === "running") return "warning";
  if (value === "failed") return "destructive";
  return "secondary";
}

function scoreTone(score: number): "success" | "warning" | "secondary" | "destructive" {
  if (score >= 80) return "success";
  if (score >= 50) return "warning";
  if (score > 0) return "destructive";
  return "secondary";
}

export default function RunsPage() {
  const { t } = useI18n();
  const { setTitle } = usePageHeader();
  const { search } = useLocation();
  const initialParams = useMemo(() => new URLSearchParams(search), [search]);
  const initialStatus = initialParams.get("status") || "";
  const initialView = initialParams.get("view") === "runs" || ["queued", "completed", "cancelled"].includes(initialStatus)
    ? "runs"
    : "tasks";
  const [project, setProject] = useState(initialParams.get("project") || "staam");
  const [projects, setProjects] = useState<RunProject[]>([]);
  const [view, setView] = useState<"tasks" | "runs">(initialView);
  const [classification, setClassification] = useState(initialParams.get("classification") || "");
  const [taskStatus, setTaskStatus] = useState(TASK_STATUSES.includes(initialStatus) ? initialStatus : "");
  const [agentId, setAgentId] = useState(initialParams.get("agent_id") || "");
  const [tasks, setTasks] = useState<RunTaskSummary[]>([]);
  const [taskTotal, setTaskTotal] = useState(0);
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [summary, setSummary] = useState<RunsSummaryResponse | null>(null);
  const [watchdog, setWatchdog] = useState<RunsWatchdogResponse | null>(null);
  const [executorScheduler, setExecutorScheduler] = useState<AgentRunExecutorSchedulerStatus | null>(null);
  const [agentRuns, setAgentRuns] = useState<AgentRunEntry[]>([]);
  const [agentRunTotal, setAgentRunTotal] = useState(0);
  const [effectiveness, setEffectiveness] = useState<AgentEffectivenessEntry[]>([]);
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null);
  const [expandedTaskId, setExpandedTaskId] = useState<string | null>(null);
  const [applyingTaskId, setApplyingTaskId] = useState<string | null>(null);
  const [executorBusy, setExecutorBusy] = useState(false);
  const [agentRunBusyId, setAgentRunBusyId] = useState<string | null>(null);
  const [handoffStatus, setHandoffStatus] = useState(HANDOFF_STATUSES.includes(initialStatus) ? initialStatus : "");
  const [smokeBusy, setSmokeBusy] = useState(false);
  const [smokeResult, setSmokeResult] = useState<string | null>(null);
  const [evalBusy, setEvalBusy] = useState(false);
  const [evalResult, setEvalResult] = useState<string | null>(null);
  const [schedulerBusy, setSchedulerBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [offset, setOffset] = useState(0);
  const [taskOffset, setTaskOffset] = useState(0);

  useEffect(() => {
    let cancelled = false;
    api.getRunProjects()
      .then((response) => {
        if (cancelled) return;
        setProjects(response.projects);
        if (response.projects.length > 0 && project === "staam") {
          setProject(response.default_project || response.projects[0].name);
          setOffset(0);
        }
      })
      .catch(() => {
        if (!cancelled) setProjects([]);
      });
    return () => {
      cancelled = true;
    };
  }, [project]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const selectedProject = project.trim() || "staam";
      const [tasksResp, runsResp, summaryResp, watchdogResp, schedulerResp, agentRunsResp, effectivenessResp] = await Promise.all([
        api.getRunTasks({
          project: selectedProject,
          status: taskStatus || undefined,
          agent_id: agentId || undefined,
          include_policy: true,
          task_type: "tests",
          risk_level: "R1",
          limit: LIMIT,
          offset: taskOffset,
        }),
        api.getRuns({
          project: selectedProject,
          classification: classification || undefined,
          agent_id: agentId || undefined,
          limit: LIMIT,
          offset,
        }),
        api.getRunsSummary({ project: selectedProject, agent_id: agentId || undefined }),
        api.getRunsWatchdog({ project: selectedProject, agent_id: agentId || undefined }),
        api.getAgentRunExecutorScheduler(),
        api.getAgentRuns({
          agent_id: agentId || undefined,
          status: handoffStatus || undefined,
          limit: 25,
        }),
        api.getAgentEffectiveness({ project: selectedProject, agent_id: agentId || undefined }),
      ]);
      setTasks(tasksResp.tasks);
      setTaskTotal(tasksResp.total);
      setRuns(runsResp.runs);
      setTotal(runsResp.total);
      setSummary(summaryResp);
      setWatchdog(watchdogResp);
      setExecutorScheduler(schedulerResp);
      setAgentRuns(agentRunsResp.runs);
      setAgentRunTotal(agentRunsResp.total);
      setEffectiveness(effectivenessResp.agents);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [agentId, classification, handoffStatus, offset, project, taskOffset, taskStatus]);

  useEffect(() => {
    setTitle(t.runs.title);
    return () => setTitle(null);
  }, [setTitle, t.runs.title]);

  useEffect(() => {
    load();
  }, [load]);

  const counts = summary?.classification_counts ?? {};
  const agentOptions = useMemo(
    () => Object.keys(summary?.agent_counts ?? {}).sort(),
    [summary?.agent_counts],
  );
  const taskTotalPages = Math.max(1, Math.ceil(taskTotal / LIMIT));
  const taskPage = Math.floor(taskOffset / LIMIT) + 1;
  const totalPages = Math.max(1, Math.ceil(total / LIMIT));
  const page = Math.floor(offset / LIMIT) + 1;
  const expandedTask = useMemo(
    () => tasks.find((task) => task.task_id === expandedTaskId) ?? null,
    [expandedTaskId, tasks],
  );
  const expandedRun = useMemo(
    () => runs.find((run) => run.run_id === expandedRunId) ?? null,
    [expandedRunId, runs],
  );

  const applyPolicy = async (taskId: string) => {
    setApplyingTaskId(taskId);
    setError(null);
    try {
      await api.applyRunTaskExecutionPolicy(taskId, {
        project: project.trim() || "staam",
        task_type: "tests",
        risk_level: "R1",
      });
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplyingTaskId(null);
    }
  };

  const runExecutorTick = async () => {
    setExecutorBusy(true);
    setError(null);
    try {
      await api.tickAgentRunExecutor(1);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setExecutorBusy(false);
    }
  };

  const runKernelizationSmoke = async () => {
    setSmokeBusy(true);
    setSmokeResult(null);
    setError(null);
    try {
      const result = await api.runKernelizationSmoke({
        project: project.trim() || "staam",
        task_type: "tests",
        risk_level: "R1",
      });
      setSmokeResult(`${result.task_id} · ${result.agent_run?.status ?? "unknown"} · ${result.steps?.length ?? 0} steps`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSmokeBusy(false);
    }
  };

  const runExternalEval = async () => {
    setEvalBusy(true);
    setEvalResult(null);
    setError(null);
    try {
      const result = await api.runExternalAgentEval({
        project: project.trim() || "staam",
        agent_ids: agentId ? [agentId] : undefined,
        timeout_seconds: 10,
      });
      const summary = result.results.map((item) => `${item.agent_id}:${item.classification}`).join(", ");
      setEvalResult(`${result.batch_id} · ${summary || "no eligible agents"}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setEvalBusy(false);
    }
  };

  const executeHandoffRun = async (runId: string) => {
    setAgentRunBusyId(runId);
    setError(null);
    try {
      await api.executeAgentRun(runId);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAgentRunBusyId(null);
    }
  };

  const cancelHandoffRun = async (runId: string) => {
    setAgentRunBusyId(runId);
    setError(null);
    try {
      await api.cancelAgentRun(runId);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setAgentRunBusyId(null);
    }
  };

  const toggleExecutorScheduler = async () => {
    setSchedulerBusy(true);
    setError(null);
    try {
      if (executorScheduler?.enabled) {
        setExecutorScheduler(await api.stopAgentRunExecutorScheduler());
      } else {
        setExecutorScheduler(await api.startAgentRunExecutorScheduler({
          interval_seconds: 5,
          timeout_seconds: 180,
        }));
      }
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSchedulerBusy(false);
    }
  };

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2">
          <input
            list="run-projects"
            className="h-8 w-40 border border-border bg-background px-2 text-xs"
            aria-label={t.runs.project}
            value={project}
            onChange={(event) => {
              setProject(event.target.value);
              setOffset(0);
            }}
          />
          <datalist id="run-projects">
            {projects.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name} ({item.total_runs})
              </option>
            ))}
          </datalist>
          <select
            className="h-8 border border-border bg-background px-2 text-xs"
            aria-label={t.runs.classification}
            value={view === "tasks" ? taskStatus : classification}
            onChange={(event) => {
              if (view === "tasks") {
                setTaskStatus(event.target.value);
                setTaskOffset(0);
              } else {
                setClassification(event.target.value);
                setOffset(0);
              }
            }}
          >
            <option value="">{view === "tasks" ? t.runs.status : t.runs.allClassifications}</option>
            {(view === "tasks" ? TASK_STATUSES : CLASSIFICATIONS).map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
          <select
            className="h-8 border border-border bg-background px-2 text-xs"
            aria-label={t.runs.agent}
            value={agentId}
            onChange={(event) => {
              setAgentId(event.target.value);
              setTaskOffset(0);
              setOffset(0);
            }}
          >
            <option value="">{t.runs.allAgents}</option>
            {agentOptions.map((item) => (
              <option key={item} value={item}>
                {item}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex h-8 overflow-hidden border border-border">
            <button
              type="button"
              className={`px-3 text-xs ${view === "tasks" ? "bg-muted text-foreground" : "text-muted-foreground"}`}
              onClick={() => setView("tasks")}
            >
              {t.runs.taskView}
            </button>
            <button
              type="button"
              className={`border-l border-border px-3 text-xs ${view === "runs" ? "bg-muted text-foreground" : "text-muted-foreground"}`}
              onClick={() => setView("runs")}
            >
              {t.runs.runView}
            </button>
          </div>
          <Button
            size="sm"
            outlined
            onClick={load}
            disabled={loading}
            prefix={loading ? <Spinner /> : <RefreshCw className="h-4 w-4" />}
          >
            {t.common.refresh}
          </Button>
          <Button
            size="sm"
            outlined
            onClick={() => void runExecutorTick()}
            disabled={executorBusy}
          >
            {executorBusy ? t.common.loading : "Run Executor"}
          </Button>
          <Button
            size="sm"
            outlined
            onClick={() => void runKernelizationSmoke()}
            disabled={smokeBusy}
          >
            {smokeBusy ? t.common.loading : "Kernel Smoke"}
          </Button>
          <Button
            size="sm"
            outlined
            onClick={() => void runExternalEval()}
            disabled={evalBusy}
          >
            {evalBusy ? t.common.loading : "Agent Eval"}
          </Button>
          <Button
            size="sm"
            outlined
            onClick={() => void toggleExecutorScheduler()}
            disabled={schedulerBusy}
          >
            {schedulerBusy
              ? t.common.loading
              : executorScheduler?.enabled
                ? "Stop Scheduler"
                : "Start Scheduler"}
          </Button>
        </div>
      </div>

      {error ? (
        <Card className="border-destructive/40">
          <CardContent className="flex items-center gap-2 py-4 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardContent className="py-4">
          <Stats
            items={[
              { label: t.runs.totalRuns, value: String(summary?.total ?? 0) },
              { label: t.runs.ok, value: String(counts.ok ?? 0) },
              { label: t.runs.timeout, value: String(counts.timeout ?? 0) },
              { label: t.runs.processError, value: String(counts.process_error ?? 0) },
              { label: t.runs.avgDuration, value: formatDuration(summary?.avg_duration_seconds) },
              { label: "Queued", value: String(executorScheduler?.queued_count ?? 0) },
              { label: "Active", value: String(executorScheduler?.active_count ?? 0) },
              { label: "Scheduler", value: executorScheduler?.enabled ? "on" : "off" },
            ]}
          />
          {executorScheduler?.last_error ? (
            <div className="mt-3 border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              {executorScheduler.last_error}
            </div>
          ) : null}
          {smokeResult ? (
            <div className="mt-3 border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
              Kernel smoke: {smokeResult}
            </div>
          ) : null}
          {evalResult ? (
            <div className="mt-3 border border-border bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
              Agent eval: {evalResult}
            </div>
          ) : null}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle className="text-base">{t.runs.watchdog}</CardTitle>
            <div className="text-xs text-muted-foreground">
              {watchdog?.attention_count === 1
                ? t.runs.taskNeedsAttention
                : `${watchdog?.attention_count ?? 0} ${t.runs.tasksNeedAttention}`}
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <Stats
            items={[
              { label: t.runs.attention, value: String(watchdog?.attention_count ?? 0) },
              { label: t.runs.running, value: String(watchdog?.status_counts.running ?? 0) },
              { label: t.runs.stale, value: String(watchdog?.status_counts.stale ?? 0) },
              { label: t.runs.timeout, value: String(watchdog?.status_counts.timeout ?? 0) },
              { label: t.runs.failed, value: String(watchdog?.status_counts.failed ?? 0) },
            ]}
          />
          {watchdog?.attention_tasks.length ? (
            <div className="grid gap-2 lg:grid-cols-3">
              {watchdog.attention_tasks.slice(0, 3).map((task) => (
                <div key={task.task_id} className="border border-border bg-background p-3">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <div className="min-w-0 truncate font-mono text-xs">{task.task_id}</div>
                    <Badge tone={taskStatusTone(task.status)}>{taskStatusLabel(t, task.status)}</Badge>
                  </div>
                  <div className="mb-2 text-xs text-muted-foreground">
                    {task.agents.join(", ") || task.latest_agent_id || "-"} · {startedLabel(task.latest_started_at)}
                  </div>
                  <div className="text-xs">{task.reason}</div>
                  <div className="mt-2 text-xs text-muted-foreground">
                    <span className="font-medium text-foreground">{t.runs.suggestedAction}: </span>
                    {task.suggested_action}
                  </div>
                  {task.execution_policy ? (
                    <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
                      <div className="flex flex-wrap gap-1">
                        <Badge tone={task.execution_policy.requires_human_approval ? "warning" : "secondary"}>
                          {task.execution_policy.action}
                        </Badge>
                        <span className="font-mono text-muted-foreground">
                          {task.execution_policy.next_agent_id || "manual"} / {task.execution_policy.next_model_ref || "none"}
                        </span>
                      </div>
                      <Button
                        size="sm"
                        outlined
                        disabled={applyingTaskId === task.task_id}
                        onClick={() => void applyPolicy(task.task_id)}
                      >
                        {applyingTaskId === task.task_id ? t.common.loading : "Apply"}
                      </Button>
                    </div>
                  ) : null}
                </div>
              ))}
            </div>
          ) : (
            <div className="border border-border bg-muted/20 px-3 py-4 text-sm text-muted-foreground">
              {t.runs.noAttention}
            </div>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-base">Executor Handoffs</CardTitle>
              <div className="flex items-center gap-2">
                <select
                  className="h-8 border border-border bg-background px-2 text-xs"
                  value={handoffStatus}
                  onChange={(event) => setHandoffStatus(event.target.value)}
                  aria-label="Handoff status"
                >
                  <option value="">All statuses</option>
                  {HANDOFF_STATUSES.map((status) => (
                    <option key={status} value={status}>{status}</option>
                  ))}
                </select>
                <span className="text-xs text-muted-foreground">{agentRunTotal} runs</span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {agentRuns.length ? (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[760px] text-left text-sm">
                  <thead className="border-b border-border text-xs text-muted-foreground">
                    <tr>
                      <th className="py-2 pr-4 font-medium">Run</th>
                      <th className="py-2 pr-4 font-medium">Agent</th>
                      <th className="py-2 pr-4 font-medium">Status</th>
                      <th className="py-2 pr-4 font-medium">Task</th>
                      <th className="py-2 pr-4 font-medium">Executor</th>
                      <th className="py-2 font-medium">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {agentRuns.slice(0, 10).map((run) => {
                      const busy = agentRunBusyId === run.run_id;
                      const executor = run.executor || {};
                      return (
                        <tr key={run.run_id} className="align-top">
                          <td className="py-2 pr-4 font-mono text-xs">
                            <div className="max-w-[11rem] truncate">{run.run_id}</div>
                            <div className="text-[11px] text-muted-foreground">{startedLabel(run.updated_at || run.created_at)}</div>
                          </td>
                          <td className="py-2 pr-4 text-xs">{run.agent_id}</td>
                          <td className="py-2 pr-4">
                            <Badge tone={runStatusTone(run.status)}>{run.status}</Badge>
                            {run.error ? (
                              <div className="mt-1 max-w-[12rem] truncate text-[11px] text-destructive">{run.error}</div>
                            ) : null}
                          </td>
                          <td className="py-2 pr-4 font-mono text-xs">
                            <div className="max-w-[12rem] truncate">{run.task_id || "-"}</div>
                            <div className="text-[11px] text-muted-foreground">{run.model_ref || "model n/a"}</div>
                          </td>
                          <td className="py-2 pr-4 text-xs text-muted-foreground">
                            <div>{String(executor.mode || "-")}</div>
                            {executor.pid ? <div>pid {String(executor.pid)}</div> : null}
                          </td>
                          <td className="py-2">
                            <div className="flex flex-wrap gap-1">
                              {run.status === "queued" ? (
                                <Button
                                  size="sm"
                                  outlined
                                  disabled={busy}
                                  onClick={() => void executeHandoffRun(run.run_id)}
                                >
                                  {busy ? t.common.loading : "Execute"}
                                </Button>
                              ) : null}
                              {run.status === "queued" || run.status === "running" ? (
                                <Button
                                  size="sm"
                                  outlined
                                  disabled={busy}
                                  onClick={() => void cancelHandoffRun(run.run_id)}
                                >
                                  {busy ? t.common.loading : "Cancel"}
                                </Button>
                              ) : null}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="border border-border bg-muted/20 px-3 py-4 text-sm text-muted-foreground">
                No executor handoffs match the current filters.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="text-base">Agent Effectiveness</CardTitle>
              <span className="text-xs text-muted-foreground">{effectiveness.length} agents</span>
            </div>
          </CardHeader>
          <CardContent>
            {effectiveness.length ? (
              <div className="space-y-2">
                {effectiveness.slice(0, 6).map((agent) => (
                  <div key={agent.agent_id} className="border border-border bg-background p-3">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <div className="font-mono text-xs">{agent.agent_id}</div>
                      <Badge tone={scoreTone(agent.effectiveness_score)}>
                        {agent.effectiveness_score.toFixed(1)}
                      </Badge>
                    </div>
                    <div className="grid grid-cols-4 gap-2 text-xs">
                      <div>
                        <div className="text-muted-foreground">Success</div>
                        <div>{agent.success_rate.toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Timeout</div>
                        <div>{agent.timeout_rate.toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Handoff</div>
                        <div>{agent.handoff_success_rate.toFixed(1)}%</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Avg</div>
                        <div>{formatDuration(agent.avg_duration_seconds)}</div>
                      </div>
                    </div>
                    <div className="mt-2 text-[11px] text-muted-foreground">
                      {agent.run_count} ledger runs · {agent.handoff_count} handoffs · {agent.revision_needed_count} revisions
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="border border-border bg-muted/20 px-3 py-4 text-sm text-muted-foreground">
                No effectiveness data yet.
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-2">
            <CardTitle className="text-base">{view === "tasks" ? t.runs.taskView : t.runs.title}</CardTitle>
            <div className="text-xs text-muted-foreground">
              {view === "tasks" ? taskTotal : total} {t.common.match} · {t.common.page} {view === "tasks" ? taskPage : page} {t.common.of} {view === "tasks" ? taskTotalPages : totalPages}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {loading && tasks.length === 0 && runs.length === 0 ? (
            <div className="flex items-center gap-2 py-10 text-sm text-muted-foreground">
              <Spinner /> {t.common.loading}
            </div>
          ) : view === "tasks" && tasks.length === 0 ? (
            <div className="py-10 text-center text-sm text-muted-foreground">
              {t.runs.noTasks}
            </div>
          ) : view === "runs" && runs.length === 0 ? (
            <div className="py-10 text-center text-sm text-muted-foreground">
              {t.runs.noRuns}
            </div>
          ) : view === "tasks" ? (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1000px] text-left text-sm">
                <thead className="border-b border-border text-xs text-muted-foreground">
                  <tr>
                    <th className="w-8 py-2 pr-2 font-medium" />
                    <th className="py-2 pr-4 font-medium">{t.runs.taskId}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.status}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.currentPhase}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.agents}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.runs}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.latestRun}</th>
                    <th className="py-2 pr-4 font-medium">Policy</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.duration}</th>
                    <th className="py-2 font-medium">{t.runs.lastError}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {tasks.map((task) => {
                    const expanded = expandedTaskId === task.task_id;
                    return (
                      <tr key={task.task_id} className="align-top">
                        <td className="py-2 pr-2">
                          <button
                            type="button"
                            className="text-muted-foreground hover:text-foreground"
                            onClick={() => setExpandedTaskId(expanded ? null : task.task_id)}
                            aria-label={expanded ? t.common.collapse : t.common.expand}
                          >
                            {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                          </button>
                        </td>
                        <td className="py-2 pr-4 font-mono text-xs">
                          <div className="max-w-[18rem] truncate">{task.task_id}</div>
                        </td>
                        <td className="py-2 pr-4">
                          <Badge tone={taskStatusTone(task.status)}>{taskStatusLabel(t, task.status)}</Badge>
                        </td>
                        <td className="py-2 pr-4 text-xs text-muted-foreground">{task.current_phase || "-"}</td>
                        <td className="py-2 pr-4 text-xs">{task.agents.join(", ") || "-"}</td>
                        <td className="py-2 pr-4 text-xs text-muted-foreground">{task.run_count}</td>
                        <td className="py-2 pr-4 text-xs text-muted-foreground">
                          {startedLabel(task.latest_started_at)}
                        </td>
                        <td className="py-2 pr-4 text-xs">
                          {task.execution_policy ? (
                            <div className="space-y-1">
                              <Badge tone={task.execution_policy.requires_human_approval ? "warning" : "secondary"}>
                                {task.execution_policy.action}
                              </Badge>
                              <div className="font-mono text-[11px] text-muted-foreground">
                                {task.execution_policy.next_agent_id || "manual"} / {task.execution_policy.next_model_ref || "none"}
                              </div>
                              <Button
                                size="sm"
                                outlined
                                disabled={applyingTaskId === task.task_id}
                                onClick={() => void applyPolicy(task.task_id)}
                              >
                                {applyingTaskId === task.task_id ? t.common.loading : "Apply"}
                              </Button>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">-</span>
                          )}
                        </td>
                        <td className="py-2 pr-4 text-xs">{formatDuration(task.total_duration_seconds)}</td>
                        <td className="py-2 text-xs text-muted-foreground">
                          <div className="max-w-[18rem] truncate">{task.last_error_excerpt || "-"}</div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {expandedTask ? (
                <div className="border-t border-border bg-muted/20 px-4 py-3">
                  <div className="mb-2 text-xs font-medium text-muted-foreground">{t.runs.lifecycle}</div>
                  {expandedTask.execution_policy ? (
                    <div className="mb-4 grid gap-2 border border-border bg-background p-3 text-xs md:grid-cols-4">
                      <div>
                        <div className="text-muted-foreground">Policy action</div>
                        <div className="font-mono">{expandedTask.execution_policy.action}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Next agent</div>
                        <div className="font-mono">{expandedTask.execution_policy.next_agent_id || "manual"}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Next model</div>
                        <div className="font-mono">{expandedTask.execution_policy.next_model_ref || "none"}</div>
                      </div>
                      <div>
                        <div className="text-muted-foreground">Attempts</div>
                        <div className="font-mono">
                          {expandedTask.execution_policy.attempt_count}/{expandedTask.execution_policy.max_attempts}
                        </div>
                      </div>
                      <div className="md:col-span-4 text-muted-foreground">
                        {expandedTask.execution_policy.reason}
                      </div>
                      <div className="md:col-span-4">
                        <Button
                          size="sm"
                          outlined
                          disabled={applyingTaskId === expandedTask.task_id}
                          onClick={() => void applyPolicy(expandedTask.task_id)}
                        >
                          {applyingTaskId === expandedTask.task_id ? t.common.loading : "Apply Policy"}
                        </Button>
                      </div>
                    </div>
                  ) : null}
                  {expandedTask.lifecycle?.length ? (
                    <div className="mb-4 space-y-2">
                      {expandedTask.lifecycle.map((item) => (
                        <div key={item.event_id || `${item.phase}-${item.started_at}`} className="grid grid-cols-[9rem_8rem_9rem_1fr] gap-2 text-xs">
                          <div className="font-mono truncate">{item.phase}</div>
                          <Badge tone={classificationTone(item.status)}>{item.status}</Badge>
                          <div className="truncate text-muted-foreground">{startedLabel(item.started_at)}</div>
                          <div className="truncate text-muted-foreground">
                            {item.message || item.decision || item.policy_action || item.revision_task_id || "-"}
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="mb-4 text-xs text-muted-foreground">-</div>
                  )}
                  <div className="mb-2 text-xs font-medium text-muted-foreground">{t.runs.latestRun}</div>
                  <div className="space-y-2">
                    {expandedTask.runs.map((run) => (
                      <div key={run.run_id || `${expandedTask.task_id}-${run.agent_id}`} className="grid grid-cols-[9rem_8rem_8rem_1fr] gap-2 text-xs">
                        <div className="font-mono truncate">{run.run_id || "-"}</div>
                        <div>{run.agent_id || "-"}</div>
                        <Badge tone={classificationTone(run.classification)}>{run.classification || run.run_type || "unknown"}</Badge>
                        <div className="truncate text-muted-foreground">{run.stderr_tail || run.stdout_tail || run.command || "-"}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1000px] text-left text-sm">
                <thead className="border-b border-border text-xs text-muted-foreground">
                  <tr>
                    <th className="w-8 py-2 pr-2 font-medium" />
                    <th className="py-2 pr-4 font-medium">{t.runs.runId}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.taskId}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.agent}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.classification}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.runType}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.startedAt}</th>
                    <th className="py-2 pr-4 font-medium">{t.runs.duration}</th>
                    <th className="py-2 font-medium">{t.runs.exitCode}</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {runs.map((run, idx) => {
                    const rowId = run.run_id || `${run.task_id || "run"}-${idx}`;
                    const expanded = expandedRunId === rowId;
                    return (
                      <tr key={rowId} className="align-top">
                        <td className="py-2 pr-2">
                          <button
                            type="button"
                            className="text-muted-foreground hover:text-foreground"
                            onClick={() => setExpandedRunId(expanded ? null : rowId)}
                            aria-label={expanded ? t.common.collapse : t.common.expand}
                          >
                            {expanded ? (
                              <ChevronUp className="h-4 w-4" />
                            ) : (
                              <ChevronDown className="h-4 w-4" />
                            )}
                          </button>
                        </td>
                        <td className="py-2 pr-4 font-mono text-xs">
                          <div className="max-w-[15rem] truncate">{run.run_id || "-"}</div>
                        </td>
                        <td className="py-2 pr-4 font-mono text-xs">
                          <div className="max-w-[13rem] truncate">{run.task_id || "-"}</div>
                        </td>
                        <td className="py-2 pr-4 text-xs">{run.agent_id || "-"}</td>
                        <td className="py-2 pr-4">
                          <Badge tone={classificationTone(run.classification)}>
                            {run.classification || "unknown"}
                          </Badge>
                        </td>
                        <td className="py-2 pr-4 text-xs text-muted-foreground">
                          {run.run_type || "-"}
                        </td>
                        <td className="py-2 pr-4 text-xs text-muted-foreground">
                          {startedLabel(run.started_at)}
                        </td>
                        <td className="py-2 pr-4 text-xs">
                          {formatDuration(run.duration_seconds)}
                        </td>
                        <td className="py-2 text-xs text-muted-foreground">
                          {run.exit_code ?? "-"}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>

              {expandedRun ? (
                <div className="border-t border-border bg-muted/20 px-4 py-3">
                  <div className="space-y-3 text-xs">
                    {expandedRun.command ? (
                      <div>
                        <div className="mb-1 font-medium text-muted-foreground">{t.runs.command}</div>
                        <pre className="max-h-32 overflow-auto border border-border bg-background p-2 font-mono text-[11px] whitespace-pre-wrap">
                          {expandedRun.command}
                        </pre>
                      </div>
                    ) : null}
                    {expandedRun.stdout_tail ? (
                      <div>
                        <div className="mb-1 font-medium text-muted-foreground">{t.runs.stdoutTail}</div>
                        <pre className="max-h-48 overflow-auto border border-border bg-background p-2 font-mono text-[11px] whitespace-pre-wrap">
                          {expandedRun.stdout_tail}
                        </pre>
                      </div>
                    ) : null}
                    {expandedRun.stderr_tail ? (
                      <div>
                        <div className="mb-1 font-medium text-muted-foreground">{t.runs.stderrTail}</div>
                        <pre className="max-h-48 overflow-auto border border-border bg-background p-2 font-mono text-[11px] whitespace-pre-wrap text-destructive">
                          {expandedRun.stderr_tail}
                        </pre>
                      </div>
                    ) : null}
                  </div>
                </div>
              ) : null}
            </div>
          )}

          {view === "tasks" && taskTotal > LIMIT ? (
            <div className="mt-3 flex items-center justify-between border-t border-border pt-3">
              <Button
                size="sm"
                outlined
                disabled={taskOffset === 0}
                onClick={() => setTaskOffset(Math.max(0, taskOffset - LIMIT))}
              >
                {t.common.collapse}
              </Button>
              <span className="text-xs text-muted-foreground">
                {taskOffset + 1}-{Math.min(taskOffset + LIMIT, taskTotal)} {t.common.of} {taskTotal}
              </span>
              <Button
                size="sm"
                outlined
                disabled={taskOffset + LIMIT >= taskTotal}
                onClick={() => setTaskOffset(taskOffset + LIMIT)}
              >
                {t.common.expand}
              </Button>
            </div>
          ) : null}

          {view === "runs" && total > LIMIT ? (
            <div className="mt-3 flex items-center justify-between border-t border-border pt-3">
              <Button
                size="sm"
                outlined
                disabled={offset === 0}
                onClick={() => setOffset(Math.max(0, offset - LIMIT))}
              >
                {t.common.collapse}
              </Button>
              <span className="text-xs text-muted-foreground">
                {offset + 1}-{Math.min(offset + LIMIT, total)} {t.common.of} {total}
              </span>
              <Button
                size="sm"
                outlined
                disabled={offset + LIMIT >= total}
                onClick={() => setOffset(offset + LIMIT)}
              >
                {t.common.expand}
              </Button>
            </div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}
