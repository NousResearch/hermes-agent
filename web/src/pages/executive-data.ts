import { useQuery } from "@tanstack/react-query";
import type {
  DashboardTone,
  DashboardSnapshot,
  ExecutiveActionItem,
  ExecutiveDomainTab,
  ExecutiveProjectScore,
  ExecutiveRollupMetric,
  QueueSnapshot,
} from "@hermes/dashboard-kit";
import { dashboardHealthScore, dashboardToneForHealth } from "@hermes/dashboard-kit";
import { api } from "@/lib/api";
import { buildKnownDashboardSnapshots, pluginToDashboardSnapshot } from "./dashboard-signals";

export interface ExecutiveSummaryData {
  actions: ExecutiveActionItem[];
  capacity: ExecutiveRollupMetric;
  cost: ExecutiveRollupMetric;
  projects: ExecutiveProjectScore[];
  snapshots: DashboardSnapshot[];
  source: "live" | "fallback";
  throughput: ExecutiveRollupMetric;
}

const fallbackActions: ExecutiveActionItem[] = [
  {
    id: "exec-live-registry",
    title: "Keep executive dashboard connected to live registry and plugin signals.",
    owner: "Hermes OS",
    urgency: "normal",
    source: "Dashboard registry",
  },
  {
    id: "exec-production-promotions",
    title: "Promote remaining local dashboards to production URLs.",
    owner: "Infrastructure",
    urgency: "normal",
    source: "Deployment rail",
  },
  {
    id: "exec-cost-sources",
    title: "Define per-project cost and capacity sources.",
    owner: "Operations",
    urgency: "normal",
    source: "Cost dashboard",
  },
];

export function useExecutiveSummaryData() {
  return useQuery({
    queryKey: ["executive-summary", "dashboard-plugins"],
    queryFn: loadExecutiveSummaryData,
    placeholderData: buildFallbackExecutiveSummary(),
  });
}

export function buildDomainTabs(projects: ExecutiveProjectScore[]): ExecutiveDomainTab[] {
  const counts = new Map<string, number>();
  for (const project of projects) {
    counts.set(project.domain, (counts.get(project.domain) ?? 0) + 1);
  }
  return [
    { id: "all", label: "All", status: String(projects.length), tone: "info" },
    ...Array.from(counts.entries())
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([domain, count]) => ({
        id: domainSlug(domain),
        label: domain,
        status: String(count),
        tone: toneForDomain(projects.filter((project) => project.domain === domain)),
      })),
  ];
}

export function domainSlug(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "");
}

async function loadExecutiveSummaryData(): Promise<ExecutiveSummaryData> {
  try {
    const response = await api.getDashboardSnapshots(false);
    if (response.snapshots.length) {
      return buildExecutiveSummary(response.snapshots as DashboardSnapshot[], "live");
    }
  } catch {
    /* Fall through to plugin-derived snapshots. */
  }
  try {
    const plugins = await api.getPlugins();
    const snapshots = mergeSnapshots(buildKnownDashboardSnapshots(), plugins.map((plugin) => pluginToDashboardSnapshot(plugin)));
    if (!snapshots.length) return buildFallbackExecutiveSummary();
    return buildExecutiveSummary(snapshots, "live");
  } catch {
    return buildFallbackExecutiveSummary();
  }
}

function buildFallbackExecutiveSummary(): ExecutiveSummaryData {
  const snapshots = buildKnownDashboardSnapshots();
  return {
    ...buildExecutiveSummary(snapshots, "fallback"),
    actions: [
      ...fallbackActions,
      ...snapshots.flatMap((snapshot) => snapshot.actions ?? []).map(actionToExecutiveAction),
    ],
  };
}

function buildExecutiveSummary(
  snapshots: DashboardSnapshot[],
  source: ExecutiveSummaryData["source"],
): ExecutiveSummaryData {
  const projects = snapshots.map(snapshotToProjectScore);
  const actions = snapshots.flatMap((snapshot) => snapshot.actions ?? []);
  const warningCount = projects.filter((project) => project.tone === "warning" || project.tone === "critical").length;
  const liveCount = projects.filter((project) => project.status === "live" || project.status === "running").length;
  const knownCostSignals = snapshots.filter((snapshot) => snapshot.cost?.known).length;
  const aggregateQueue = aggregateQueues(snapshots);
  const knownCapacitySignals = snapshots.filter((snapshot) => snapshot.capacity?.known).length;
  return {
    actions: actions.length ? actions.map(actionToExecutiveAction) : fallbackActions.slice(0, 1),
    capacity: {
      label: "Capacity",
      value: knownCapacitySignals ? `${knownCapacitySignals}/${snapshots.length}` : "partial",
      detail: `${aggregateQueue.running} running, ${aggregateQueue.failed + aggregateQueue.blocked + aggregateQueue.stale} need review`,
      tone: aggregateQueue.failed || aggregateQueue.blocked || warningCount ? "warning" : "success",
    },
    cost: {
      label: "Known Cost Signals",
      value: `${knownCostSignals}/${snapshots.length}`,
      detail: knownCostSignals ? "standard cost telemetry available" : "cost source adapters remain project-owned",
      tone: knownCostSignals === snapshots.length ? "success" : "warning",
    },
    projects,
    snapshots,
    source,
    throughput: {
      label: "Throughput",
      value: aggregateQueue.running ? `${aggregateQueue.running} running` : liveCount ? "active" : "limited",
      detail: `${liveCount} active dashboard signals, ${aggregateQueue.queued} queued`,
      tone: liveCount ? "success" : "warning",
    },
  };
}

function snapshotToProjectScore(snapshot: DashboardSnapshot): ExecutiveProjectScore {
  const tone = dashboardToneForHealth(snapshot.health.state);
  const queue = snapshot.queue;
  const costKnown = snapshot.cost?.known ? "known" : "missing";
  const capacityKnown = snapshot.capacity?.known ? "known" : "missing";
  return {
    id: snapshot.source.id,
    name: snapshot.source.label,
    owner: snapshot.source.owner,
    domain: readableDomain(snapshot.source.category),
    status: snapshot.health.state === "healthy" ? "live" : snapshot.health.state,
    tone,
    healthScore: dashboardHealthScore(snapshot),
    summary: snapshot.health.message || "Standard dashboard signal is registered.",
    metrics: [
      { label: "Queue", value: queue ? `${queue.running} running` : "unknown" },
      { label: "Cost", value: costKnown },
      { label: "Capacity", value: capacityKnown },
    ],
  };
}

function actionToExecutiveAction(action: import("@hermes/dashboard-kit").ActionNeeded): ExecutiveActionItem {
  return {
    id: action.id,
    title: action.title,
    owner: action.owner,
    urgency: action.severity,
    due: action.due,
    source: action.source ?? action.sourceDashboardId,
  };
}

function toneForDomain(projects: ExecutiveProjectScore[]): DashboardTone {
  if (projects.some((project) => project.tone === "critical")) return "critical";
  if (projects.some((project) => project.tone === "warning")) return "warning";
  if (projects.some((project) => project.tone === "success")) return "success";
  return "info";
}

function aggregateQueues(snapshots: DashboardSnapshot[]): QueueSnapshot {
  return snapshots.reduce<QueueSnapshot>((total, snapshot) => {
    const queue = snapshot.queue;
    if (!queue) return total;
    total.queued += queue.queued;
    total.running += queue.running;
    total.failed += queue.failed;
    total.blocked += queue.blocked;
    total.stale += queue.stale;
    total.completed = (total.completed ?? 0) + (queue.completed ?? 0);
    return total;
  }, { queued: 0, running: 0, failed: 0, blocked: 0, stale: 0, completed: 0 });
}

function mergeSnapshots(primary: DashboardSnapshot[], secondary: DashboardSnapshot[]): DashboardSnapshot[] {
  const byId = new Map<string, DashboardSnapshot>();
  for (const snapshot of primary) byId.set(snapshot.source.id, snapshot);
  for (const snapshot of secondary) byId.set(snapshot.source.id, snapshot);
  return Array.from(byId.values());
}

function readableDomain(category: string) {
  return category
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
