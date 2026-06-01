import { useCallback, useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  Activity,
  AlertTriangle,
  Bot,
  CheckCircle2,
  ClipboardList,
  FileText,
  GitBranch,
  HelpCircle,
  Play,
  RefreshCw,
  Route,
  ShieldAlert,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  AgentCapabilityMatrixResponse,
  AgentEffectivenessEntry,
  AgentRunEntry,
  AgentRunExecutorSchedulerStatus,
  ManagedAgentEntry,
  RunsWatchdogResponse,
} from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { usePageHeader } from "@/contexts/usePageHeader";
import { isoTimeAgo, timeAgo } from "@/lib/utils";

type KnownState = "healthy" | "degraded" | "critical";
type UncertainState = "unknown" | "unavailable" | "external-managed";
type OperationHealthState = KnownState | UncertainState;
type EvalState = "ok" | "timeout" | "failed" | "unknown" | "unavailable";
type AlertSeverity = "critical" | "warning" | "info";
type AlertActionType =
  | "run_eval"
  | "run_collaboration_smoke"
  | "view_runs"
  | "view_agents"
  | "view_routing"
  | "view_logs"
  | "view_models";

interface HandoffStatus {
  queued: number;
  running: number;
  completed: number;
  failed: number;
  cancelled: number;
  source: "agent-runs" | "scheduler" | "unavailable";
}

interface EvalResult {
  state: EvalState;
  classification: string;
  status: string;
  run_id?: string;
  source: "external_agent_eval" | "agent-runs" | "unavailable";
  observed_at?: string;
  detail: string;
}

interface AlertAction {
  type: AlertActionType;
  label: string;
  agent_id?: string;
  params?: Record<string, string | number | boolean | undefined>;
}

interface OperationAlert {
  id: string;
  severity: AlertSeverity;
  title: string;
  detail: string;
  source: string;
  uncertainty: "known" | "partial" | "unknown" | "external-managed";
  action: AlertAction;
}

interface OperationAgentHealth {
  agent_id: string;
  display_name: string;
  runtime: string;
  model_ref: string;
  health_state: OperationHealthState;
  health_score: number | null;
  health_source: "effectiveness" | "external-runtime" | "unavailable";
  routing_rank: number | null;
  routing_source: "adaptive_effectiveness" | "static_capability" | "unknown";
  eval_result: EvalResult;
  handoff_status: HandoffStatus;
  alerts: OperationAlert[];
}

const EXTERNAL_AGENT_IDS = ["claude", "codex", "deepseek-tui", "opencode"];
const DEFAULT_PROJECT = "staam";
const ROUTE_TASK_TYPES = ["tests", "implementation", "code_review", "architecture_review"];
const ROUTE_RISKS = ["R0", "R1", "R2", "R3", "R4"];
const ROUTE_FAILURES = ["timeout", "rate_limited", "revision_needed", "auth_error", "ineffective", "empty_final_content"];

function isExternalRuntime(agent: ManagedAgentEntry): boolean {
  return agent.runtime.endsWith("_cli");
}

function runtimeName(runtime: string): string {
  if (runtime === "claude_code_cli") return "Claude Code CLI";
  if (runtime === "codex_cli") return "Codex CLI";
  if (runtime === "deepseek_tui_cli") return "DeepSeek TUI";
  if (runtime === "opencode_cli") return "OpenCode CLI";
  return runtime || "native";
}

function stateTone(state: OperationHealthState): "success" | "warning" | "secondary" | "destructive" {
  if (state === "healthy") return "success";
  if (state === "degraded" || state === "external-managed") return "warning";
  if (state === "critical") return "destructive";
  return "secondary";
}

function severityTone(severity: AlertSeverity): "success" | "warning" | "secondary" | "destructive" {
  if (severity === "critical") return "destructive";
  if (severity === "warning") return "warning";
  return "secondary";
}

function evalTone(state: EvalState): "success" | "warning" | "secondary" | "destructive" {
  if (state === "ok") return "success";
  if (state === "timeout") return "warning";
  if (state === "failed") return "destructive";
  return "secondary";
}

function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "unknown";
  return `${Math.round(value)}%`;
}

function runObservedAt(run: AgentRunEntry): string | undefined {
  if (run.updated_at) return run.updated_at;
  if (run.created_at) return run.created_at;
  if (run.ended_at) return new Date(run.ended_at * 1000).toISOString();
  if (run.started_at) return new Date(run.started_at * 1000).toISOString();
  return undefined;
}

function runSource(run: AgentRunEntry): string {
  const executor = run.executor ?? {};
  const source = executor.source;
  return typeof source === "string" ? source : "agent-runs";
}

function runClassification(run: AgentRunEntry): string {
  const executor = run.executor ?? {};
  const classification = executor.classification;
  if (typeof classification === "string") return classification;
  if (run.error?.toLowerCase().includes("timeout")) return "timeout";
  if (run.status === "completed") return "ok";
  if (run.status === "failed") return "failed";
  return "unknown";
}

function evalFromRuns(agentId: string, runs: AgentRunEntry[]): EvalResult {
  const agentRuns = runs
    .filter((run) => run.agent_id === agentId)
    .sort((a, b) => {
      const aTime = runObservedAt(a) ? new Date(runObservedAt(a) as string).getTime() : 0;
      const bTime = runObservedAt(b) ? new Date(runObservedAt(b) as string).getTime() : 0;
      return bTime - aTime;
    });
  const evalRun = agentRuns.find((run) => runSource(run) === "external_agent_eval") ?? agentRuns[0];
  if (!evalRun) {
    return {
      state: "unknown",
      classification: "unknown",
      status: "unknown",
      source: "unavailable",
      detail: "No ledger-backed eval sample is available for this agent.",
    };
  }

  const classification = runClassification(evalRun);
  const failed = evalRun.status === "failed" || classification === "failed" || classification.includes("error");
  const state: EvalState =
    classification === "ok" || evalRun.status === "completed"
      ? "ok"
      : classification === "timeout"
        ? "timeout"
        : failed
          ? "failed"
          : "unknown";
  return {
    state,
    classification,
    status: evalRun.status,
    run_id: evalRun.run_id,
    source: runSource(evalRun) === "external_agent_eval" ? "external_agent_eval" : "agent-runs",
    observed_at: runObservedAt(evalRun),
    detail: evalRun.error || evalRun.result_summary || "Ledger sample has no summary.",
  };
}

function handoffFromRuns(agentId: string, runs: AgentRunEntry[]): HandoffStatus {
  const counts: HandoffStatus = {
    queued: 0,
    running: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
    source: "agent-runs",
  };
  for (const run of runs) {
    if (run.agent_id !== agentId) continue;
    if (run.status === "queued") counts.queued += 1;
    else if (run.status === "running") counts.running += 1;
    else if (run.status === "completed") counts.completed += 1;
    else if (run.status === "failed") counts.failed += 1;
    else if (run.status === "cancelled") counts.cancelled += 1;
  }
  return counts;
}

function healthState(
  agent: ManagedAgentEntry,
  effectiveness?: AgentEffectivenessEntry,
  evalResult?: EvalResult,
): OperationHealthState {
  if (!effectiveness) return isExternalRuntime(agent) ? "external-managed" : "unknown";
  if (evalResult?.state === "timeout" || effectiveness.timeout_rate >= 40 || effectiveness.effectiveness_score < 35) {
    return "critical";
  }
  if (evalResult?.state === "failed" || effectiveness.failed_rate >= 30 || effectiveness.effectiveness_score < 70) {
    return "degraded";
  }
  return "healthy";
}

function buildAlerts(args: {
  agent: ManagedAgentEntry;
  effectiveness?: AgentEffectivenessEntry;
  evalResult: EvalResult;
  handoff: HandoffStatus;
  routingRank: number | null;
  watchdog?: RunsWatchdogResponse | null;
}): OperationAlert[] {
  const { agent, effectiveness, evalResult, handoff, routingRank, watchdog } = args;
  const alerts: OperationAlert[] = [];
  if (!effectiveness) {
    alerts.push({
      id: `${agent.agent_id}:effectiveness-unknown`,
      severity: "info",
      title: "Effectiveness unknown",
      detail: "No effectiveness sample is available. Treat this agent as unranked until a smoke or eval run lands in the ledger.",
      source: "effectiveness",
      uncertainty: isExternalRuntime(agent) ? "external-managed" : "unknown",
      action: { type: "run_eval", label: "Run eval", agent_id: agent.agent_id, params: { source: "effectiveness" } },
    });
  }
  if (isExternalRuntime(agent)) {
    alerts.push({
      id: `${agent.agent_id}:external-managed`,
      severity: "info",
      title: "External-managed runtime",
      detail: `${runtimeName(agent.runtime)} can hang or fail outside Hermes. Its health must come from bounded eval samples, not assumed availability.`,
      source: "runtime",
      uncertainty: "external-managed",
      action: { type: "run_eval", label: "Run bounded eval", agent_id: agent.agent_id, params: { source: "runtime" } },
    });
  }
  if (evalResult.state === "timeout") {
    alerts.push({
      id: `${agent.agent_id}:eval-timeout`,
      severity: "warning",
      title: "Latest eval timed out",
      detail: evalResult.detail,
      source: evalResult.source,
      uncertainty: "known",
      action: { type: "view_routing", label: "View routing", agent_id: agent.agent_id, params: { failure: "timeout" } },
    });
  } else if (evalResult.state === "failed") {
    alerts.push({
      id: `${agent.agent_id}:eval-failed`,
      severity: "critical",
      title: "Latest eval failed",
      detail: evalResult.detail,
      source: evalResult.source,
      uncertainty: "known",
      action: { type: "view_logs", label: "View logs", agent_id: agent.agent_id, params: { search: agent.agent_id } },
    });
  } else if (evalResult.state === "unknown") {
    alerts.push({
      id: `${agent.agent_id}:eval-unknown`,
      severity: "info",
      title: "Eval state unknown",
      detail: evalResult.detail,
      source: evalResult.source,
      uncertainty: "unknown",
      action: { type: "run_eval", label: "Run eval", agent_id: agent.agent_id, params: { source: "eval" } },
    });
  }
  if (effectiveness && effectiveness.timeout_rate >= 30) {
    alerts.push({
      id: `${agent.agent_id}:timeout-rate`,
      severity: "warning",
      title: "Timeout rate above routing threshold",
      detail: `${formatPercent(effectiveness.timeout_rate)} timeout rate. Consider demotion or a shorter retry path before routing more work here.`,
      source: "effectiveness",
      uncertainty: "known",
      action: { type: "view_routing", label: "Review demotion", agent_id: agent.agent_id, params: { failure: "timeout" } },
    });
  }
  if (effectiveness && effectiveness.handoff_count > 0 && effectiveness.handoff_success_rate < 70) {
    alerts.push({
      id: `${agent.agent_id}:handoff-success`,
      severity: "warning",
      title: "Handoff success is weak",
      detail: `${formatPercent(effectiveness.handoff_success_rate)} handoff success across ${effectiveness.handoff_count} handoffs.`,
      source: "agent-runs",
      uncertainty: "partial",
      action: { type: "view_runs", label: "View runs", agent_id: agent.agent_id, params: { status: "failed" } },
    });
  }
  if (handoff.queued > 0 || handoff.running > 0) {
    alerts.push({
      id: `${agent.agent_id}:handoff-active`,
      severity: "info",
      title: "Handoff work in flight",
      detail: `${handoff.queued} queued, ${handoff.running} running for this agent.`,
      source: handoff.source,
      uncertainty: "known",
      action: { type: "view_runs", label: "View handoffs", agent_id: agent.agent_id, params: { status: handoff.running > 0 ? "running" : "queued" } },
    });
  }
  const watchdogCount = watchdog?.agent_counts?.[agent.agent_id];
  const watchdogAttention = watchdogCount
    ? Object.entries(watchdogCount).reduce((sum, [status, count]) => (
        status === "stale" || status === "timeout" || status === "failed" ? sum + count : sum
      ), 0)
    : 0;
  if (watchdogAttention > 0) {
    alerts.push({
      id: `${agent.agent_id}:watchdog`,
      severity: "critical",
      title: "Watchdog needs attention",
      detail: `${watchdogAttention} task(s) are stale, timed out, or failed for this agent.`,
      source: "watchdog",
      uncertainty: "known",
      action: { type: "view_runs", label: "View runs", agent_id: agent.agent_id, params: { status: "attention" } },
    });
  }
  if (routingRank === null) {
    alerts.push({
      id: `${agent.agent_id}:routing-unranked`,
      severity: "info",
      title: "Routing rank unavailable",
      detail: "This agent is not in the current tests/R1 routing preview, or the preview failed to load.",
      source: "routing",
      uncertainty: "unknown",
      action: { type: "view_routing", label: "View routing", agent_id: agent.agent_id },
    });
  }
  return alerts;
}

function actionPath(action: AlertAction, project: string): string {
  let path = "/runs";
  if (action.type === "view_agents" || action.type === "view_routing") path = "/agents";
  else if (action.type === "view_logs") path = "/logs";
  else if (action.type === "view_models") path = "/models";

  const params = new URLSearchParams();
  if (project) params.set("project", project);
  if (action.agent_id) params.set("agent_id", action.agent_id);
  if (action.type === "view_routing") params.set("section", "routing");
  for (const [key, value] of Object.entries(action.params ?? {})) {
    if (value !== undefined) params.set(key, String(value));
  }
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

export default function OperationsPage() {
  const navigate = useNavigate();
  const { setTitle } = usePageHeader();
  const [project, setProject] = useState(DEFAULT_PROJECT);
  const [routeTaskType, setRouteTaskType] = useState("tests");
  const [routeRisk, setRouteRisk] = useState("R1");
  const [routeFailure, setRouteFailure] = useState("timeout");
  const [routeFailedAgent, setRouteFailedAgent] = useState("deepseek-tui");
  const [routeFailedModel, setRouteFailedModel] = useState("opencode_go_deepseek_flash");
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [agents, setAgents] = useState<ManagedAgentEntry[]>([]);
  const [effectiveness, setEffectiveness] = useState<AgentEffectivenessEntry[]>([]);
  const [agentRuns, setAgentRuns] = useState<AgentRunEntry[]>([]);
  const [scheduler, setScheduler] = useState<AgentRunExecutorSchedulerStatus | null>(null);
  const [watchdog, setWatchdog] = useState<RunsWatchdogResponse | null>(null);
  const [routing, setRouting] = useState<AgentCapabilityMatrixResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [loadWarnings, setLoadWarnings] = useState<string[]>([]);
  const [busyAction, setBusyAction] = useState<string | null>(null);
  const [actionResult, setActionResult] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);

  useEffect(() => {
    setTitle("Operations");
    return () => setTitle(null);
  }, [setTitle]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setLoadWarnings([]);
    const selectedProject = project.trim() || DEFAULT_PROJECT;
    try {
      const managedResp = await api.getManagedAgents(30);
      const [effectivenessResp, runsResp, schedulerResp, watchdogResp, routingResp] = await Promise.allSettled([
        api.getAgentEffectiveness({ project: selectedProject }),
        api.getAgentRuns({ limit: 100 }),
        api.getAgentRunExecutorScheduler(),
        api.getRunsWatchdog({ project: selectedProject }),
        api.getAgentCapabilityMatrix({
          task_type: routeTaskType,
          risk_level: routeRisk,
          failure: routeFailure,
          failed_agent_id: routeFailedAgent || undefined,
          failed_model_ref: routeFailedModel || undefined,
        }),
      ]);
      const warnings: string[] = [];
      const warn = (name: string, result: PromiseSettledResult<unknown>) => {
        if (result.status === "rejected") {
          warnings.push(`${name}: ${result.reason instanceof Error ? result.reason.message : String(result.reason)}`);
        }
      };
      warn("effectiveness", effectivenessResp);
      warn("agent-runs", runsResp);
      warn("scheduler", schedulerResp);
      warn("watchdog", watchdogResp);
      warn("routing", routingResp);
      setAgents(managedResp.agents);
      setEffectiveness(effectivenessResp.status === "fulfilled" ? effectivenessResp.value.agents : []);
      setAgentRuns(runsResp.status === "fulfilled" ? runsResp.value.runs : []);
      setScheduler(schedulerResp.status === "fulfilled" ? schedulerResp.value : null);
      setWatchdog(watchdogResp.status === "fulfilled" ? watchdogResp.value : null);
      setRouting(routingResp.status === "fulfilled" ? routingResp.value : null);
      setLoadWarnings(warnings);
      setLastUpdatedAt(new Date().toISOString());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [project, routeFailedAgent, routeFailedModel, routeFailure, routeRisk, routeTaskType]);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    if (!autoRefresh || busyAction !== null) return;
    const interval = window.setInterval(() => {
      void load();
    }, 30000);
    return () => window.clearInterval(interval);
  }, [autoRefresh, busyAction, load]);

  const operations = useMemo<OperationAgentHealth[]>(() => {
    const effectivenessByAgent = new Map(effectiveness.map((entry) => [entry.agent_id, entry]));
    const candidateAgents = routing?.preview?.candidate_agents ?? [];
    return agents.map((agent) => {
      const eff = effectivenessByAgent.get(agent.agent_id);
      const evalResult = evalFromRuns(agent.agent_id, agentRuns);
      const handoff = handoffFromRuns(agent.agent_id, agentRuns);
      const candidateIndex = candidateAgents.indexOf(agent.agent_id);
      const routingRank = candidateIndex >= 0 ? candidateIndex + 1 : null;
      const state = healthState(agent, eff, evalResult);
      const alerts = buildAlerts({
        agent,
        effectiveness: eff,
        evalResult,
        handoff,
        routingRank,
        watchdog,
      });
      return {
        agent_id: agent.agent_id,
        display_name: agent.display_name,
        runtime: agent.runtime,
        model_ref: agent.model_ref,
        health_state: state,
        health_score: eff?.effectiveness_score ?? null,
        health_source: eff ? "effectiveness" : isExternalRuntime(agent) ? "external-runtime" : "unavailable",
        routing_rank: routingRank,
        routing_source: routingRank ? "adaptive_effectiveness" : routing ? "static_capability" : "unknown",
        eval_result: evalResult,
        handoff_status: handoff,
        alerts,
      };
    });
  }, [agentRuns, agents, effectiveness, routing, watchdog]);

  const allAlerts = useMemo(
    () => operations.flatMap((agent) => agent.alerts.map((alert) => ({ ...alert, agent_id: agent.agent_id }))),
    [operations],
  );
  const criticalCount = allAlerts.filter((alert) => alert.severity === "critical").length;
  const warningCount = allAlerts.filter((alert) => alert.severity === "warning").length;
  const unknownCount = operations.filter((agent) => (
    agent.health_state === "unknown" ||
    agent.health_state === "unavailable" ||
    agent.health_state === "external-managed"
  )).length;
  const externalCount = agents.filter(isExternalRuntime).length;
  const queuedCount = scheduler?.queued_count ?? operations.reduce((sum, agent) => sum + agent.handoff_status.queued, 0);
  const activeCount = scheduler?.active_count ?? operations.reduce((sum, agent) => sum + agent.handoff_status.running, 0);

  const runEval = async (agentId: string) => {
    setBusyAction(`eval:${agentId}`);
    setError(null);
    setActionResult(null);
    const selectedProject = project.trim() || DEFAULT_PROJECT;
    try {
      const resp = await api.runExternalAgentEval({
        project: selectedProject,
        agent_ids: [agentId],
        timeout_seconds: 10,
      });
      const result = resp.results[0];
      setActionResult(`${agentId} eval recorded: ${result?.classification ?? "unknown"} / ${result?.status ?? "unknown"}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyAction(null);
    }
  };

  const runCollaborationSmoke = async () => {
    setBusyAction("collaboration-smoke");
    setError(null);
    setActionResult(null);
    const selectedProject = project.trim() || DEFAULT_PROJECT;
    try {
      const resp = await api.runExternalAgentEval({
        project: selectedProject,
        agent_ids: EXTERNAL_AGENT_IDS,
        timeout_seconds: 8,
        prompt: "Hermes collaboration smoke: respond with one short readiness line. Do not edit files.",
      });
      const summary = resp.results
        .map((result) => `${result.agent_id}:${result.classification || result.status}`)
        .join(", ");
      setActionResult(`Collaboration sample recorded: ${summary}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyAction(null);
    }
  };

  const runKernelSmoke = async () => {
    setBusyAction("kernel-smoke");
    setError(null);
    setActionResult(null);
    const selectedProject = project.trim() || DEFAULT_PROJECT;
    try {
      const resp = await api.runKernelizationSmoke({
        project: selectedProject,
        task_type: routeTaskType,
        risk_level: routeRisk,
        failed_agent_id: routeFailedAgent || undefined,
        failed_model_ref: routeFailedModel || undefined,
        classification: routeFailure,
      });
      setActionResult(resp.ok ? `Kernel smoke recorded: ${resp.task_id}` : `Kernel smoke failed: ${resp.error ?? "unknown"}`);
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyAction(null);
    }
  };

  const handleAlertAction = (alert: OperationAlert) => {
    if (alert.action.type === "run_eval" && alert.action.agent_id) {
      void runEval(alert.action.agent_id);
      return;
    }
    if (alert.action.type === "run_collaboration_smoke") {
      void runCollaborationSmoke();
      return;
    }
    navigate(actionPath(alert.action, project.trim() || DEFAULT_PROJECT));
  };

  return (
    <div className="space-y-4 p-4 lg:p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold tracking-normal text-foreground">Agent Operations</h1>
          <p className="mt-1 max-w-3xl text-sm normal-case text-muted-foreground">
            Single cockpit for health, queue pressure, routing explanation, risk alerts, and ledger-backed collaboration samples.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <label className="flex items-center gap-2 text-xs normal-case text-muted-foreground">
            <span>Project</span>
            <input
              className="h-8 w-32 rounded-md border border-border bg-background px-2 font-mono text-xs"
              value={project}
              onChange={(event) => setProject(event.target.value)}
            />
          </label>
          <label className="flex items-center gap-2 text-xs normal-case text-muted-foreground">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(event) => setAutoRefresh(event.target.checked)}
            />
            Auto refresh
          </label>
          <Button
            size="sm"
            outlined
            onClick={load}
            disabled={loading || busyAction !== null}
            prefix={loading ? <Spinner /> : <RefreshCw className="h-4 w-4" />}
          >
            Refresh
          </Button>
          <Button
            size="sm"
            outlined
            onClick={runKernelSmoke}
            disabled={busyAction !== null}
            prefix={busyAction === "kernel-smoke" ? <Spinner /> : <GitBranch className="h-4 w-4" />}
          >
            Kernel Smoke
          </Button>
          <Button
            size="sm"
            onClick={runCollaborationSmoke}
            disabled={busyAction !== null}
            prefix={busyAction === "collaboration-smoke" ? <Spinner /> : <Play className="h-4 w-4" />}
          >
            Collaboration Smoke
          </Button>
        </div>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-end gap-3 py-3">
          <label className="space-y-1 text-xs normal-case">
            <span className="text-muted-foreground">Task type</span>
            <select
              className="block h-8 w-40 rounded-md border border-border bg-background px-2 text-xs"
              value={routeTaskType}
              onChange={(event) => setRouteTaskType(event.target.value)}
            >
              {ROUTE_TASK_TYPES.map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs normal-case">
            <span className="text-muted-foreground">Risk</span>
            <select
              className="block h-8 w-24 rounded-md border border-border bg-background px-2 text-xs"
              value={routeRisk}
              onChange={(event) => setRouteRisk(event.target.value)}
            >
              {ROUTE_RISKS.map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs normal-case">
            <span className="text-muted-foreground">Failure</span>
            <select
              className="block h-8 w-44 rounded-md border border-border bg-background px-2 text-xs"
              value={routeFailure}
              onChange={(event) => setRouteFailure(event.target.value)}
            >
              {ROUTE_FAILURES.map((value) => (
                <option key={value} value={value}>{value}</option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs normal-case">
            <span className="text-muted-foreground">Failed agent</span>
            <select
              className="block h-8 w-44 rounded-md border border-border bg-background px-2 text-xs"
              value={routeFailedAgent}
              onChange={(event) => setRouteFailedAgent(event.target.value)}
            >
              <option value="">none</option>
              {agents.map((agent) => (
                <option key={agent.agent_id} value={agent.agent_id}>{agent.agent_id}</option>
              ))}
            </select>
          </label>
          <label className="space-y-1 text-xs normal-case">
            <span className="text-muted-foreground">Failed model</span>
            <input
              className="block h-8 w-56 rounded-md border border-border bg-background px-2 font-mono text-xs"
              value={routeFailedModel}
              onChange={(event) => setRouteFailedModel(event.target.value)}
              placeholder="model_ref"
            />
          </label>
          <div className="ml-auto text-[11px] normal-case text-muted-foreground">
            {lastUpdatedAt ? `updated ${isoTimeAgo(lastUpdatedAt)}` : "not loaded yet"}
          </div>
        </CardContent>
      </Card>

      {error ? (
        <Card className="border-destructive/40">
          <CardContent className="flex items-center gap-2 py-4 text-sm normal-case text-destructive">
            <AlertTriangle className="h-4 w-4" />
            {error}
          </CardContent>
        </Card>
      ) : null}

      {loadWarnings.length ? (
        <Card className="border-warning/40">
          <CardContent className="space-y-1 py-4 text-sm normal-case text-muted-foreground">
            <div className="flex items-center gap-2 text-amber-600 dark:text-amber-400">
              <AlertTriangle className="h-4 w-4" />
              Some operations data sources are unavailable; the dashboard is showing partial truth.
            </div>
            {loadWarnings.map((warning) => (
              <div key={warning} className="truncate font-mono text-[11px]">
                {warning}
              </div>
            ))}
          </CardContent>
        </Card>
      ) : null}

      {actionResult ? (
        <Card className="border-primary/30">
          <CardContent className="flex items-center gap-2 py-4 text-sm normal-case">
            <CheckCircle2 className="h-4 w-4 text-primary" />
            {actionResult}
          </CardContent>
        </Card>
      ) : null}

      <Card>
        <CardContent className="py-4">
          <Stats
            items={[
              { label: "Agents", value: String(agents.length) },
              { label: "External", value: String(externalCount) },
              { label: "Queue", value: `${queuedCount}/${activeCount}` },
              { label: "Alerts", value: `${criticalCount}/${warningCount}` },
              { label: "Uncertain", value: String(unknownCount) },
            ]}
          />
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[1.5fr_1fr]">
        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Bot className="h-4 w-4" />
                Agent Health Contract
              </CardTitle>
              <Badge tone={routing?.preview ? "success" : "secondary"}>
                routing {routing?.preview ? "available" : "unknown"}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            {loading && !operations.length ? (
              <div className="flex items-center gap-2 py-8 text-sm normal-case text-muted-foreground">
                <Spinner /> Loading operations data...
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[1120px] text-left text-sm">
                  <thead className="border-b border-border text-xs text-muted-foreground">
                    <tr>
                      <th className="py-2 pr-3 font-medium">Agent</th>
                      <th className="py-2 pr-3 font-medium">Health</th>
                      <th className="py-2 pr-3 font-medium">Routing</th>
                      <th className="py-2 pr-3 font-medium">Eval Result</th>
                      <th className="py-2 pr-3 font-medium">Handoff Status</th>
                      <th className="py-2 pr-3 font-medium">Source</th>
                      <th className="py-2 pr-3 font-medium">Action</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {operations.map((agent) => (
                      <tr key={agent.agent_id} className="align-top">
                        <td className="py-3 pr-3">
                          <div className="font-medium">{agent.display_name || agent.agent_id}</div>
                          <div className="mt-0.5 font-mono text-[11px] normal-case text-muted-foreground">
                            {agent.agent_id} · {runtimeName(agent.runtime)}
                          </div>
                          <div className="mt-1 font-mono text-[11px] normal-case text-muted-foreground">
                            {agent.model_ref || "model unavailable"}
                          </div>
                        </td>
                        <td className="py-3 pr-3">
                          <div className="flex items-center gap-2">
                            <Badge tone={stateTone(agent.health_state)}>{agent.health_state}</Badge>
                            <span className="text-xs text-muted-foreground">
                              {agent.health_score === null ? "score unknown" : `${Math.round(agent.health_score)}/100`}
                            </span>
                          </div>
                        </td>
                        <td className="py-3 pr-3">
                          <div className="text-xs">
                            {agent.routing_rank ? `#${agent.routing_rank}` : "unranked"}
                          </div>
                          <div className="mt-1 text-[11px] normal-case text-muted-foreground">
                            {agent.routing_source}
                          </div>
                        </td>
                        <td className="py-3 pr-3">
                          <div className="flex items-center gap-2">
                            <Badge tone={evalTone(agent.eval_result.state)}>{agent.eval_result.state}</Badge>
                            <span className="text-xs normal-case text-muted-foreground">
                              {agent.eval_result.classification}
                            </span>
                          </div>
                          <div className="mt-1 max-w-[16rem] truncate text-[11px] normal-case text-muted-foreground">
                            {agent.eval_result.observed_at ? isoTimeAgo(agent.eval_result.observed_at) : agent.eval_result.detail}
                          </div>
                        </td>
                        <td className="py-3 pr-3">
                          <div className="grid grid-cols-5 gap-1 text-[11px]">
                            <span>Q {agent.handoff_status.queued}</span>
                            <span>R {agent.handoff_status.running}</span>
                            <span>OK {agent.handoff_status.completed}</span>
                            <span>F {agent.handoff_status.failed}</span>
                            <span>C {agent.handoff_status.cancelled}</span>
                          </div>
                        </td>
                        <td className="py-3 pr-3">
                          <div className="text-xs normal-case">{agent.health_source}</div>
                          <div className="mt-1 text-[11px] normal-case text-muted-foreground">
                            eval: {agent.eval_result.source}
                          </div>
                        </td>
                        <td className="py-3 pr-3">
                          <Button
                            size="sm"
                            outlined
                            onClick={() => runEval(agent.agent_id)}
                            disabled={busyAction !== null}
                            prefix={busyAction === `eval:${agent.agent_id}` ? <Spinner /> : <Play className="h-3.5 w-3.5" />}
                          >
                            Run Eval
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <ClipboardList className="h-4 w-4" />
                Queue And Scheduler
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground">Scheduler</div>
                  <div className="mt-1">
                    <Badge tone={scheduler?.enabled ? "success" : "secondary"}>
                      {scheduler ? (scheduler.enabled ? "enabled" : "disabled") : "unknown"}
                    </Badge>
                  </div>
                </div>
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground">Last tick</div>
                  <div className="mt-1 text-xs normal-case">
                    {scheduler?.last_tick ? isoTimeAgo(scheduler.last_tick) : "unknown"}
                  </div>
                </div>
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground">Queued</div>
                  <div className="mt-1 text-lg">{queuedCount}</div>
                </div>
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground">Active</div>
                  <div className="mt-1 text-lg">{activeCount}</div>
                </div>
              </div>
              {scheduler?.last_error ? (
                <div className="rounded-md border border-destructive/40 p-3 text-xs normal-case text-destructive">
                  {scheduler.last_error}
                </div>
              ) : null}
              <div className="flex flex-wrap gap-2">
                <Button size="sm" outlined onClick={() => navigate("/runs")}>
                  View Runs
                </Button>
                <Button size="sm" outlined onClick={() => navigate("/logs")}>
                  View Logs
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-base">
                <Route className="h-4 w-4" />
                Routing Explanation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="grid gap-3">
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground">Primary</div>
                  <div className="mt-1 font-mono text-xs normal-case">
                    {routing?.preview?.primary_agent || "unknown"}
                  </div>
                </div>
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground">Candidate order</div>
                  <div className="mt-2 flex flex-wrap gap-1">
                    {(routing?.preview?.candidate_agents ?? []).length ? (
                      routing?.preview?.candidate_agents.map((agentId, index) => (
                        <span key={agentId} className="bg-muted px-1.5 py-0.5 font-mono text-[10px] normal-case text-muted-foreground">
                          #{index + 1} {agentId}
                        </span>
                      ))
                    ) : (
                      <span className="text-xs normal-case text-muted-foreground">unknown</span>
                    )}
                  </div>
                </div>
                <div className="rounded-md border border-border p-3">
                  <div className="text-xs text-muted-foreground">Reroute</div>
                  <div className="mt-1 text-xs normal-case">
                    {routing?.reroute?.action ?? "unknown"} · {routing?.reroute?.next_agent_id ?? "manual/unknown"}
                  </div>
                  <div className="mt-1 text-[11px] normal-case text-muted-foreground">
                    {routing?.reroute?.reason ?? "No routing preview loaded."}
                  </div>
                </div>
              </div>
              <Button size="sm" outlined onClick={() => navigate("/agents")}>
                View Agents
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <ShieldAlert className="h-4 w-4" />
              Risk Alerts
            </CardTitle>
          </CardHeader>
          <CardContent>
            {allAlerts.length ? (
              <div className="space-y-2">
                {allAlerts.slice(0, 12).map((alert) => (
                  <div key={alert.id} className="flex flex-wrap items-start justify-between gap-3 rounded-md border border-border p-3 text-sm">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={severityTone(alert.severity)}>{alert.severity}</Badge>
                        <span className="font-medium">{alert.title}</span>
                        <span className="font-mono text-[11px] normal-case text-muted-foreground">{alert.agent_id}</span>
                      </div>
                      <div className="mt-1 text-xs normal-case text-muted-foreground">{alert.detail}</div>
                      <div className="mt-1 text-[11px] normal-case text-muted-foreground">
                        source: {alert.source} · uncertainty: {alert.uncertainty}
                      </div>
                    </div>
                    <Button
                      size="sm"
                      outlined
                      onClick={() => handleAlertAction(alert)}
                      disabled={busyAction !== null}
                    >
                      {alert.action.label}
                    </Button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center gap-2 py-8 text-sm normal-case text-muted-foreground">
                <CheckCircle2 className="h-4 w-4" />
                No risk alerts from the current contract.
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-base">
              <Activity className="h-4 w-4" />
              Collaboration Samples
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="grid gap-2">
              {agentRuns.slice(0, 8).map((run) => (
                <div key={run.run_id} className="flex flex-wrap items-center justify-between gap-3 rounded-md border border-border p-3 text-sm">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge tone={evalTone(runClassification(run) as EvalState)}>{runClassification(run)}</Badge>
                      <span className="font-mono text-xs normal-case">{run.agent_id}</span>
                      <span className="text-xs normal-case text-muted-foreground">{run.status}</span>
                    </div>
                    <div className="mt-1 max-w-[32rem] truncate text-[11px] normal-case text-muted-foreground">
                      {runSource(run)} · {runObservedAt(run) ? isoTimeAgo(runObservedAt(run) as string) : run.started_at ? timeAgo(run.started_at) : "unknown time"} · {run.error || run.result_summary || "no summary"}
                    </div>
                  </div>
                  <Button size="sm" outlined onClick={() => navigate("/runs")}>
                    View
                  </Button>
                </div>
              ))}
              {!agentRuns.length ? (
                <div className="flex items-center gap-2 py-8 text-sm normal-case text-muted-foreground">
                  <HelpCircle className="h-4 w-4" />
                  No collaboration samples are available yet. Run Collaboration Smoke to create a bounded ledger sample.
                </div>
              ) : null}
            </div>
            <div className="flex flex-wrap gap-2">
              <Button size="sm" outlined onClick={() => navigate("/delegations")}>
                Delegations
              </Button>
              <Button size="sm" outlined onClick={() => navigate("/models")}>
                Models
              </Button>
              <Button size="sm" outlined onClick={() => navigate("/logs")} prefix={<FileText className="h-3.5 w-3.5" />}>
                Logs
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
