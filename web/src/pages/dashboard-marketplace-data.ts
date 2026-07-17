import type { DashboardPluginManifest } from "@hermes/dashboard-kit";

export const dashboardMarketplacePlugins: DashboardPluginManifest[] = [
  {
    id: "khashi-vc.roc",
    label: "Khashi VC ROC",
    owner: "Khashi VC",
    category: "research-operations",
    version: "0.1.0",
    description: "Research operations, market coverage, scheduler, experiments, and findings.",
    productionUrl: "https://roc.tlccapitalgroup.com/",
    healthUrl: "https://roc.tlccapitalgroup.com/readyz",
    panels: [
      { id: "coverage", label: "Coverage", surface: "panel", signalContract: "ResearchSignal" },
      { id: "runs", label: "Run Monitor", surface: "panel", signalContract: "QueueSnapshot" },
    ],
    commands: [
      { id: "sync-markets", label: "Sync Markets", permission: "operator", riskLevel: "medium" },
      { id: "retire-adapter", label: "Retire Adapter", permission: "admin", riskLevel: "high" },
    ],
    signals: ["DashboardSnapshot", "HealthSnapshot", "QueueSnapshot", "ResearchSignal"],
    permissions: ["viewer", "operator", "admin"],
  },
  {
    id: "media-engine.ops",
    label: "Media Engine Ops",
    owner: "Media Engine",
    category: "media-operations",
    version: "0.1.0",
    description: "Publishing operations, autopilot, brand generation, storage, and Discord delivery.",
    productionUrl: "https://media.tlccapitalgroup.com/dashboard",
    healthUrl: "https://media.tlccapitalgroup.com/health",
    panels: [
      { id: "autopilot", label: "Autopilot", surface: "panel", signalContract: "CapacitySnapshot" },
      { id: "publishing", label: "Publishing Queue", surface: "panel", signalContract: "QueueSnapshot" },
    ],
    commands: [
      { id: "autopilot-start", label: "Start Autopilot", permission: "operator", riskLevel: "medium" },
      { id: "publish-approved", label: "Publish Approved", permission: "admin", riskLevel: "high" },
    ],
    signals: ["DashboardSnapshot", "HealthSnapshot", "CapacitySnapshot", "QueueSnapshot", "ActionNeeded"],
    permissions: ["viewer", "operator", "admin"],
  },
  {
    id: "hermes.central-command",
    label: "Hermes Central Command",
    owner: "Hermes",
    category: "control-plane",
    version: "0.1.0",
    description: "Executive rollups, action queue, daily brief, themes, and plugin discovery.",
    localUrl: "/central-command",
    panels: [
      { id: "daily-brief", label: "Daily Brief", surface: "panel", signalContract: "DashboardSnapshot" },
      { id: "themes", label: "Theme Profiles", surface: "theme" },
    ],
    commands: [
      { id: "refresh-signals", label: "Refresh Signals", permission: "operator", riskLevel: "low" },
    ],
    signals: ["DashboardSnapshot", "ActionNeeded", "DeploymentSignal"],
    permissions: ["viewer", "operator"],
  },
];
