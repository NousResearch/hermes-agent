import type {
  ActionNeeded,
  DashboardSignalSource,
  DashboardSnapshot,
  QueueSnapshot,
} from "@hermes/dashboard-kit";
import type { PluginManifestResponse } from "@/lib/api";

export const knownDashboardSources: DashboardSignalSource[] = [
  {
    id: "khashi-vc.roc",
    label: "Khashi VC ROC",
    owner: "Khashi VC",
    category: "research-operations",
    projectName: "Khashi VC",
    projectPath: "/Users/hq/Workspace/projects/khashi-vc",
    url: "https://roc.tlccapitalgroup.com/",
    healthUrl: "https://roc.tlccapitalgroup.com/readyz",
  },
  {
    id: "media-engine.ops",
    label: "Media Engine Ops",
    owner: "Media Engine",
    category: "media-operations",
    projectName: "Media Engine",
    projectPath: "/Users/hq/Workspace/projects/media-engine",
    url: "https://media.tlccapitalgroup.com/dashboard",
    healthUrl: "https://media.tlccapitalgroup.com/health",
  },
  {
    id: "media-business-operations.main",
    label: "Media Business Operations",
    owner: "Media Business Operations",
    category: "media-operations",
    projectName: "Media Business Operations",
    projectPath: "/Users/hq/Workspace/projects/media-business-operations",
    url: "http://localhost:4100/dashboard",
    healthUrl: "http://localhost:4100/health",
  },
  {
    id: "business-mapper.workspace",
    label: "Business Mapper Workspace",
    owner: "Business Mapper",
    category: "business-operations",
    projectName: "Business Mapper",
    projectPath: "/Users/hq/Workspace/projects/business-mapper",
    url: "http://localhost:8765/dashboard",
    healthUrl: "http://localhost:8765/health",
  },
  {
    id: "meal-assistant.main",
    label: "Meal Assistant",
    owner: "Meal Assistant",
    category: "household-operations",
    projectName: "Meal Assistant",
    projectPath: "/Users/hq/Workspace/projects/Meal-assistant",
    url: "http://localhost:4184/",
    healthUrl: "http://localhost:4184/health",
  },
  {
    id: "investing-system.roc",
    label: "Investing System ROC",
    owner: "Investing System",
    category: "investment-operations",
    projectName: "Investing System",
    projectPath: "/Users/hq/Workspace/projects/investing-system",
    url: "http://localhost:3102/roc",
    healthUrl: "http://localhost:3102/health",
  },
  {
    id: "hermes.workspace",
    label: "Hermes Workspace",
    owner: "Hermes",
    category: "control-plane",
    projectName: "Hermes Workspace",
    projectPath: "/Users/hq/Workspace/projects/hermes",
    url: "http://localhost:3920/",
    healthUrl: "http://localhost:3920/api/dashboard-summary",
  },
  {
    id: "nous-hermes-agent.dashboard",
    label: "Nous Hermes Agent",
    owner: "Hermes",
    category: "control-plane",
    projectName: "Nous Hermes Agent",
    url: "/",
    healthUrl: "/api/status",
  },
];

const knownActions: Record<string, ActionNeeded[]> = {
  "khashi-vc.roc": [
    {
      id: "khashi-package-native-roc",
      title: "Convert ROC from static adapter to package-native dashboard.",
      owner: "Khashi VC",
      severity: "high",
      sourceDashboardId: "khashi-vc.roc",
      source: "V8 migration",
      nextStep: "Extract ROC API client and migrate run monitor first.",
    },
  ],
  "media-engine.ops": [
    {
      id: "media-engine-snapshot-api",
      title: "Split dashboard snapshot JSON from generated HTML.",
      owner: "Media Engine",
      severity: "high",
      sourceDashboardId: "media-engine.ops",
      source: "V8 migration",
      nextStep: "Expose package-native React dashboard from the same snapshot.",
    },
  ],
};

export function buildKnownDashboardSnapshots(now = new Date().toISOString()): DashboardSnapshot[] {
  return knownDashboardSources.map((source) => ({
    source,
    health: {
      state: source.id === "nous-hermes-agent.dashboard" ? "healthy" : "degraded",
      score: source.id === "nous-hermes-agent.dashboard" ? 84 : 68,
      message: source.id === "nous-hermes-agent.dashboard"
        ? "Package-native reference dashboard is available."
        : "Production dashboard exists, but package-native migration and standard signal endpoint are incomplete.",
      checkedAt: now,
      freshness: "fresh",
    },
    cost: {
      period: "30d",
      known: false,
      message: "Project has not yet exposed a standard cost signal.",
    },
    capacity: {
      known: false,
      pressure: "unknown",
      message: "Project has not yet exposed a standard capacity signal.",
    },
    queue: queueForSource(source.id),
    actions: knownActions[source.id] ?? [],
    deployment: {
      environment: source.url?.startsWith("https://") ? "production" : "local",
      status: "unknown",
      message: "Deployment status requires the V9 deployment signal adapter.",
    },
    updatedAt: now,
  }));
}

export function pluginToDashboardSnapshot(plugin: PluginManifestResponse, now = new Date().toISOString()): DashboardSnapshot {
  const source: DashboardSignalSource = {
    id: plugin.name,
    label: plugin.label,
    owner: sourceOwner(plugin.source),
    category: domainFromPlugin(plugin),
    url: plugin.tab.override ?? plugin.tab.path,
  };
  const hidden = plugin.tab.hidden === true;
  return {
    source,
    health: {
      state: hidden ? "degraded" : plugin.has_api ? "healthy" : "unknown",
      score: hidden ? 58 : plugin.has_api ? 82 : 62,
      message: plugin.description || "Registered Hermes dashboard plugin.",
      checkedAt: now,
      freshness: "fresh",
    },
    cost: {
      period: "unknown",
      known: false,
      message: "Plugin has not exposed standard cost telemetry.",
    },
    capacity: {
      known: false,
      pressure: "unknown",
      message: "Plugin has not exposed standard capacity telemetry.",
    },
    queue: { queued: 0, running: 0, failed: 0, blocked: hidden ? 1 : 0, stale: 0 },
    actions: hidden
      ? [{
          id: `${plugin.name}-hidden`,
          title: `${plugin.label} is hidden from navigation.`,
          owner: "Hermes OS",
          severity: "normal",
          sourceDashboardId: plugin.name,
          source: "Plugin registry",
        }]
      : [],
    updatedAt: now,
  };
}

function queueForSource(id: string): QueueSnapshot {
  if (id === "khashi-vc.roc") return { queued: 0, running: 30, failed: 0, blocked: 0, stale: 0 };
  if (id === "media-engine.ops") return { queued: 0, running: 0, failed: 1, blocked: 0, stale: 0 };
  return { queued: 0, running: 1, failed: 0, blocked: 0, stale: 0 };
}

function domainFromPlugin(plugin: PluginManifestResponse) {
  const text = `${plugin.name} ${plugin.label} ${plugin.description}`.toLowerCase();
  if (text.includes("media")) return "media-operations";
  if (text.includes("khashi") || text.includes("invest") || text.includes("market")) return "research-operations";
  if (text.includes("meal") || text.includes("personal")) return "personal-os";
  if (text.includes("business") || text.includes("client")) return "business-operations";
  return plugin.source === "builtin" ? "control-plane" : "extensions";
}

function sourceOwner(source: string) {
  if (source === "builtin") return "Hermes OS";
  if (source.includes("/")) return source.split("/").filter(Boolean).slice(-1)[0] ?? source;
  return source || "Dashboard owner";
}
