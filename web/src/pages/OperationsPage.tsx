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
  X,
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

type RiskCategory = "system" | "agent" | "routing" | "scheduler";

interface OperationAlert {
  id: string;
  severity: AlertSeverity;
  title: string;
  detail: string;
  source: string;
  uncertainty: "known" | "partial" | "unknown" | "external-managed";
  category: RiskCategory;
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
      category: "system",
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
      category: "system",
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
      category: "agent",
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
      category: "agent",
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
      category: "agent",
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
      category: "routing",
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
      category: "agent",
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
      category: "agent",
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
      category: "scheduler",
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
      category: "routing",
    });
  }
  return alerts;
}

function actionPath(action: AlertAction, project: string, extras?: { source?: string; alertSeverity?: string; alertCategory?: string }): string {
  let path = "/runs";
  if (action.type === "view_agents" || action.type === "view_routing") path = "/agents";
  else if (action.type === "view_logs") path = "/logs";
  else if (action.type === "view_models") path = "/models";

  const params = new URLSearchParams();
  params.set("from", "operations");
  if (project) params.set("project", project);
  if (action.agent_id) params.set("agent_id", action.agent_id);
  if (action.type === "view_routing") params.set("section", "routing");
  if (extras?.source) params.set("source", extras.source);
  if (extras?.alertSeverity) params.set("severity", extras.alertSeverity);
  if (extras?.alertCategory) params.set("category", extras.alertCategory);
  for (const [key, value] of Object.entries(action.params ?? {})) {
    if (value !== undefined) params.set(key, String(value));
  }
  const query = params.toString();
  return query ? `${path}?${query}` : path;
}

function actionWhy(action: AlertAction, alert: OperationAlert): string {
  if (action.type === 'run_eval') return `Eval baseline missing or stale (source: ${alert.source})`;
  if (action.type === 'view_routing') return 'Drill into routing weights and demotion rules';
  return `See detail in ${action.type.replace('view_', '').replace('_', ' ')} page`;
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
  const [actionResults, setActionResults] = useState<Array<{
    id: string;
    timestamp: string;
    type: string;
    summary: string;
  }>>([]);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<string | null>(null);
  const [alertFilter, setAlertFilter] = useState<AlertSeverity | "all">("all");
  const [acknowledgedAlerts, setAcknowledgedAlerts] = useState<Set<string>>(new Set());
  const [categoryFilter, setCategoryFilter] = useState<RiskCategory | "all">("all");


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

  const stabilityStats = useMemo(() => {
    interface StabilityEntry {
      total: number;
      ok: number;
      timeout: number;
      failed: number;
      lastState: string;
      lastTime: string | null;
      lastTimeMs: number;
      timeoutRate: number;
      okRate: number;
      avgDurationMs: number;
      p95DurationMs: number;
      handoffSuccessRate: number;
      latestError: string | null;
    }
    const stats: Record<string, StabilityEntry> = {};

    for (const agentId of EXTERNAL_AGENT_IDS) {
      const runs = agentRuns.filter((r) => r.agent_id === agentId);
      if (!runs.length) {
        stats[agentId] = { total: 0, ok: 0, timeout: 0, failed: 0, lastState: "no sample", lastTime: null, lastTimeMs: 0, timeoutRate: 0, okRate: 0, avgDurationMs: 0, p95DurationMs: 0, handoffSuccessRate: 0, latestError: null };
        continue;
      }
      const sorted = [...runs].sort((a, b) => {
        const aTime = runObservedAt(a) ? new Date(runObservedAt(a) as string).getTime() : 0;
        const bTime = runObservedAt(b) ? new Date(runObservedAt(b) as string).getTime() : 0;
        return bTime - aTime;
      });
      const last = sorted[0];
      const classification = runClassification(last);
      const lastTimeMs = runObservedAt(last) ? new Date(runObservedAt(last) as string).getTime() : 0;
      const completedRuns = runs.filter((r) => r.status === "completed" || runClassification(r) === "ok");
      const handoffRuns = runs.filter((r) => r.status === "completed" || r.status === "failed");
      const durations = completedRuns
        .map((r) => (r as any).duration_ms as number | undefined)
        .filter((d): d is number => typeof d === "number" && d > 0)
        .sort((a, b) => a - b);
      const avgDurationMs = durations.length > 0 ? durations.reduce((sum, d) => sum + d, 0) / durations.length : 0;
      const p95Idx = Math.max(0, Math.ceil(durations.length * 0.95) - 1);
      const p95DurationMs = durations.length > 0 ? durations[p95Idx] : 0;
      stats[agentId] = {
        total: runs.length,
        ok: runs.filter((r) => runClassification(r) === "ok" || r.status === "completed").length,
        timeout: runs.filter((r) => runClassification(r) === "timeout" || (r.error && r.error.toLowerCase().includes("timeout"))).length,
        failed: runs.filter((r) => r.status === "failed" || runClassification(r) === "failed").length,
        lastState: classification,
        lastTime: runObservedAt(last) ?? null,
        lastTimeMs,
        timeoutRate: runs.length > 0 ? (runs.filter((r) => runClassification(r) === "timeout" || (r.error && r.error.toLowerCase().includes("timeout"))).length / runs.length) * 100 : 0,
        okRate: runs.length > 0 ? (runs.filter((r) => runClassification(r) === "ok" || r.status === "completed").length / runs.length) * 100 : 0,
        avgDurationMs,
        p95DurationMs,
        handoffSuccessRate: handoffRuns.length > 0 ? (handoffRuns.filter((r) => r.status === "completed").length / handoffRuns.length) * 100 : 0,
        latestError: sorted.find((r) => r.error)?.error ?? null,
      };
    }
    return stats;
  }, [agentRuns]);

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
    setError(null);
    const selectedProject = project.trim() || DEFAULT_PROJECT;
    try {
      const resp = await api.runExternalAgentEval({
        project: selectedProject,
        agent_ids: [agentId],
        timeout_seconds: 10,
      });
      const result = resp.results[0];
      setActionResults((prev) => [
        {
          id: `eval:${agentId}:${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: "eval",
          summary: `${agentId} eval recorded: ${result?.classification ?? "unknown"} / ${result?.status ?? "unknown"}`,
        },
        ...prev,
      ].slice(0, 10));
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
    setError(null);
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
      setActionResults((prev) => [
        {
          id: `smoke:${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: "collaboration-smoke",
          summary: `Collaboration sample recorded: ${summary}`,
        },
        ...prev,
      ].slice(0, 10));
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
    setError(null);
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
      setActionResults((prev) => [
        {
          id: `kernel:${Date.now()}`,
          timestamp: new Date().toISOString(),
          type: "kernel-smoke",
          summary: resp.ok ? `Kernel smoke recorded: ${resp.task_id}` : `Kernel smoke failed: ${resp.error ?? "unknown"}`,
        },
        ...prev,
      ].slice(0, 10));
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
    navigate(actionPath(alert.action, project.trim() || DEFAULT_PROJECT, {
      source: alert.source,
      alertSeverity: alert.severity,
      alertCategory: alert.category,
    }));
  };

  const acknowledgeAlert = (alertId: string) => {
    setAcknowledgedAlerts((prev) => {
      const next = new Set(prev);
      next.add(alertId);
      return next;
    });
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

      {actionResults.length > 0 ? (
        <Card className="border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center justify-between text-sm">
              <span className="flex items-center gap-2">
                <CheckCircle2 className="h-4 w-4 text-primary" />
                Action Results
              </span>
              <button
                className="text-[11px] text-muted-foreground hover:text-foreground transition-colors"
                onClick={() => setActionResults([])}
              >
                Clear all
              </button>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 pt-0">
            {actionResults.map((result) => (
              <div key={result.id} className="flex items-start justify-between gap-3 rounded-md border border-border p-2.5 text-sm">
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-[10px] uppercase text-muted-foreground">{result.type}</span>
                    <span className="text-[11px] text-muted-foreground">{isoTimeAgo(result.timestamp)}</span>
                  </div>
                  <div className="mt-0.5 text-xs normal-case">{result.summary}</div>
                </div>
                <button
                  className="shrink-0 text-muted-foreground hover:text-foreground transition-colors"
                  onClick={() => setActionResults((prev) => prev.filter((r) => r.id !== result.id))}
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
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
            <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
              {(["all", "system", "agent", "routing", "scheduler"] as const).map((cat) => {
                const count =
                  cat === "all"
                    ? allAlerts.length
                    : allAlerts.filter((a) => a.category === cat).length;
                return (
                  <button
                    key={cat}
                    className={`rounded-full px-2.5 py-0.5 text-[10px] font-medium transition-colors border ${
                      categoryFilter === cat
                        ? "bg-muted border-border text-foreground"
                        : "border-transparent text-muted-foreground hover:border-border/50"
                    }`}
                    onClick={() => setCategoryFilter(cat)}
                  >
                    {cat === "all" ? `All` : `${cat} ${count}`}
                  </button>
                );
              })}
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
            <div className="mt-2 flex flex-wrap items-center gap-2">
              {(["all", "critical", "warning", "info"] as const).map((filter) => {
                const count =
                  filter === "all"
                    ? allAlerts.length
                    : allAlerts.filter((a) => a.severity === filter).length;
                return (
                  <button
                    key={filter}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition-colors ${
                      alertFilter === filter
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted text-muted-foreground hover:bg-muted/80"
                    }`}
                    onClick={() => setAlertFilter(filter)}
                  >
                    {filter === "all" ? `All ${count}` : `${filter} ${count}`}
                  </button>
                );
              })}
            </div>
          </CardHeader>
          <CardContent>
            {allAlerts.length ? (
              <div className="space-y-2">
                {allAlerts
                  .filter((alert) => {
                    if (acknowledgedAlerts.has(alert.id)) return false;
                    if (alertFilter === "all") {
                      if (categoryFilter === "all") return true;
                      return alert.category === categoryFilter;
                    }
                    if (alert.severity !== alertFilter) return false;
                    if (categoryFilter !== "all" && alert.category !== categoryFilter) return false;
                    return true;
                  })
                  .slice(0, 12)
                  .map((alert) => {
                    const borderColor =
                      alert.severity === "critical"
                        ? "border-l-destructive"
                        : alert.severity === "warning"
                          ? "border-l-amber-500"
                          : "border-l-blue-500";
                    return (
                  <div key={alert.id} className={`flex flex-wrap items-start justify-between gap-3 rounded-md border border-border p-3 text-sm border-l-4 ${borderColor}`}>
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge tone={severityTone(alert.severity)}>{alert.severity}</Badge>
                        <span className="font-medium">{alert.title}</span>
                        <span className="font-mono text-[11px] normal-case text-muted-foreground">{alert.agent_id}</span>
                      </div>
                      <div className="mt-1 text-xs normal-case text-muted-foreground">{alert.detail}</div>
                      <div className="mt-1 text-[11px] normal-case text-muted-foreground">
                        source: {alert.source} · {alert.category} · uncertainty: {alert.uncertainty}
                        <span className="ml-1 text-[11px] italic text-muted-foreground/70">
                          → {actionWhy(alert.action, alert)}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      outlined
                      onClick={() => handleAlertAction(alert)}
                      disabled={busyAction !== null}
                    >
                      {alert.action.label}
                    </Button>
                    <Button
                      size="sm"
                      outlined
                      onClick={() => acknowledgeAlert(alert.id)}
                      title="Acknowledge and hide this alert"
                      prefix={<X className="h-3 w-3" />}
                    />
                    </div>
                  </div>
                    );
                  })}
              </div>
            ) : (
              <div className="flex items-center gap-2 py-8 text-sm normal-case text-muted-foreground">
                <CheckCircle2 className="h-4 w-4" />
                {allAlerts.length === 0
                  ? "No risk alerts from the current contract."
                  : acknowledgedAlerts.size > 0 && alertFilter === "all"
                    ? "All alerts acknowledged."
                    : `No ${alertFilter} alerts.`}
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
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
              {EXTERNAL_AGENT_IDS.map((agentId) => {
                const stat = stabilityStats[agentId];
                if (!stat) return null;
                const tone =
                  stat.lastState === "ok" ? "success"
                  : stat.lastState.includes("timeout") ? "warning"
                  : stat.total === 0 ? "secondary"
                  : "destructive";
                const cardBorder =
                  tone === "success" ? "border-l-green-500"
                  : tone === "warning" ? "border-l-amber-500"
                  : tone === "destructive" ? "border-l-destructive"
                  : "border-l-border";
                const staleMs = stat.lastTimeMs > 0 ? Date.now() - stat.lastTimeMs : Infinity;
                const isStale = stat.total > 0 && staleMs > 3600000;
                return (
                  <div key={agentId} className={"rounded-md border border-border p-3 text-xs border-l-4 " + cardBorder}>
                    <div className="flex items-center justify-between gap-1">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono font-medium text-[11px] normal-case">{agentId}</span>
                        <span className="text-[9px] uppercase text-muted-foreground/60 border border-border/50 rounded px-1 py-0.25">external</span>
                        {isStale ? (
                          <span className="text-[9px] text-amber-500" title={"Last sample " + Math.round(staleMs / 60000) + "m ago"}>
                            stale {Math.round(staleMs / 60000)}m
                          </span>
                        ) : null}
                        {stat.total === 0 ? (
                          <span className="text-[9px] text-muted-foreground/50">no sample</span>
                        ) : null}
                      </div>
                      <Button
                        size="sm"
                        outlined
                        className="h-6 px-1.5 text-[10px]"
                        onClick={async () => {
                          const selectedProject = project.trim() || DEFAULT_PROJECT;
                          setBusyAction("eval:" + agentId);
                          try {
                            const resp = await api.runExternalAgentEval({
                              project: selectedProject,
                              agent_ids: [agentId],
                              timeout_seconds: 8,
                              prompt: "Hermes collaboration smoke: respond with one short readiness line. Do not edit files.",
                            });
                            setActionResults((prev) => [
                              {
                                id: `smoke:${agentId}:${Date.now()}`,
                                timestamp: new Date().toISOString(),
                                type: "smoke",
                                summary: agentId + " smoke " + (resp.results[0]?.classification ?? "recorded"),
                              },
                              ...prev,
                            ].slice(0, 10));
                            await load();
                          } catch (e) {
                            setError(e instanceof Error ? e.message : String(e));
                          } finally {
                            setBusyAction(null);
                          }
                        }}
                        disabled={busyAction !== null}
                        title={"Smoke " + agentId}
                      >
                        {busyAction === "eval:" + agentId ? <Spinner /> : <Play className="h-2.5 w-2.5" />}
                      </Button>
                    </div>
                    <div className="mt-1.5 flex items-center gap-1.5">
                      <Badge tone={tone}>{stat.lastState}</Badge>
                      <span className="text-[10px] text-muted-foreground">
                        {stat.lastTime ? isoTimeAgo(stat.lastTime) : "never"}
                      </span>
                    </div>
                    <div className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
                      <span title="Success rate">{Math.round(stat.okRate)}% ok</span>
                      <span title="Timeout rate">{Math.round(stat.timeoutRate)}% timeout</span>
                      <span title="Average duration">{stat.avgDurationMs > 0 ? (stat.avgDurationMs / 1000).toFixed(1) + "s avg" : "avg -"}</span>
                      <span title="P95 duration">{stat.p95DurationMs > 0 ? (stat.p95DurationMs / 1000).toFixed(1) + "s p95" : "p95 -"}</span>
                      <span title="Handoff success">{Math.round(stat.handoffSuccessRate)}% hoff</span>
                      <span title="Total samples">{stat.total} runs</span>
                    </div>
                    {stat.latestError ? (
                      <div className="mt-1.5 max-w-[14rem] truncate text-[9px] text-destructive/80" title={stat.latestError}>
                        {stat.latestError}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
            {agentRuns.length ? (
              <details className="group">
                <summary className="cursor-pointer py-1 text-xs text-muted-foreground hover:text-foreground transition-colors">
                  Recent samples ({agentRuns.slice(0, 12).length} shown)
                </summary>
                <div className="mt-2 grid gap-2">
                  {agentRuns.slice(0, 12).map((run) => (
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
                </div>
              </details>
            ) : (
              <div className="flex items-center gap-2 py-8 text-sm normal-case text-muted-foreground">
                <HelpCircle className="h-4 w-4" />
                No collaboration samples are available yet. Run Collaboration Smoke to create a bounded ledger sample.
              </div>
            )}
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
