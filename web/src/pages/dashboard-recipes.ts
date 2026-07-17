import {
  Activity,
  BarChart3,
  Building2,
  Cpu,
  FlaskConical,
  GitBranch,
  LineChart,
  Radio,
  type LucideIcon,
} from "lucide-react";

export interface DashboardRecipe {
  id: string;
  title: string;
  useFor: string;
  layout: string[];
  dataContract: string[];
  components: string[];
  validation: string[];
  icon: LucideIcon;
}

export const dashboardRecipes: DashboardRecipe[] = [
  {
    id: "executive-command-center",
    title: "Executive Command Center",
    useFor: "TLC/Hermes top-level rollups across companies, projects, dashboards, and action queues.",
    layout: ["Global health header", "Portfolio KPI strip", "Domain tabs", "Action-needed queue", "Project scorecards", "Evidence and recommendation column"],
    dataContract: ["projects[]", "actions[]", "domains[]"],
    components: ["DashboardHeader", "MetricGrid", "KpiCard", "DashboardLauncher", "DataTable", "InsightPanel"],
    validation: ["Shows action-needed state", "Links back to source dashboards", "Distinguishes unknown, healthy, warning, and critical health"],
    icon: Building2,
  },
  {
    id: "operations-control-room",
    title: "Operations Control Room",
    useFor: "Schedulers, workers, queues, autopilot systems, market polling, production commands, and run capacity.",
    layout: ["Live state header", "Separated command bar", "Capacity row", "Queue timeline", "Run monitor table", "Worker health", "Audit events"],
    dataContract: ["commands[]", "capacity", "runs[]", "workers[]"],
    components: ["CommandBar", "CapacityMeter", "RunStatusPanel", "QueuePanel", "ActivityTimeline", "DataTable"],
    validation: ["Dangerous commands declare riskLevel", "Disabled commands explain why", "Stale and failed states are visible"],
    icon: Activity,
  },
  {
    id: "research-intelligence-dashboard",
    title: "Research Intelligence Dashboard",
    useFor: "Findings, evidence, experiments, promoted knowledge, tag/category research, and strategy readiness.",
    layout: ["Research objective header", "Evidence KPI strip", "Finding cards", "Coverage heatmap", "Experiment table", "Recommendation panel"],
    dataContract: ["findings[]", "coverage[]", "experiments[]"],
    components: ["InsightPanel", "FindingCard", "RecommendationCard", "HeatmapGrid", "FilterBar", "DataTable"],
    validation: ["Separates observation from recommendation", "Shows confidence and evidence count", "Includes coverage or blind-spot view"],
    icon: FlaskConical,
  },
  {
    id: "pipeline-workflow-dashboard",
    title: "Pipeline Workflow Dashboard",
    useFor: "Media Engine generation pipelines, approvals, publishing, Discord delivery, Search Console work, and failed package review.",
    layout: ["Production window header", "Brand/job health metrics", "Pipeline lanes", "Approval queue", "Discord output panel", "Failure review section"],
    dataContract: ["jobs[]", "approvals[]", "outputs[]"],
    components: ["RunStatusPanel", "QueuePanel", "DataTable", "ActivityTimeline", "CommandBar", "DashboardSection"],
    validation: ["Keeps internal logs out of public output", "Shows no-deliverable failure reason", "Shows brand enabled/disabled state"],
    icon: GitBranch,
  },
  {
    id: "cost-capacity-dashboard",
    title: "Cost And Capacity Dashboard",
    useFor: "Token spend, model/vendor usage, API calls, storage growth, CPU pressure, queue cost, and budget enforcement.",
    layout: ["Budget posture header", "7/30/90 range toggle", "Cost/capacity KPI strip", "Trend charts", "Breakdown table", "Budget risk column"],
    dataContract: ["usageSeries[]", "budgets[]", "resources[]"],
    components: ["DateRangeToggle", "MetricGrid", "CapacityMeter", "ChartPanel", "SimpleLineChart", "DataTable"],
    validation: ["Range controls change data", "Budget overage explains enforcement mode", "External API and storage usage are visible"],
    icon: Cpu,
  },
  {
    id: "market-asset-explorer",
    title: "Market Asset Explorer",
    useFor: "Kashi markets, investing assets, stocks, ETFs, categories, tags, liquidity, close windows, and watchlists.",
    layout: ["Universe/freshness header", "Filter rail", "Category/tag heatmap", "Dense result table", "Detail panel", "Research action bar"],
    dataContract: ["assets[]", "coverage[]", "selectedAsset"],
    components: ["FilterBar", "SearchInput", "SegmentedControl", "HeatmapGrid", "DataTable", "CommandBar"],
    validation: ["Exposes tags/subcategories", "Shows data freshness", "Distinguishes active, closing soon, closed, and unknown assets"],
    icon: LineChart,
  },
  {
    id: "brand-business-performance",
    title: "Brand Business Performance Dashboard",
    useFor: "Media brands, content output, channel analytics, engagement, posting consistency, and business performance.",
    layout: ["Brand selector header", "Output consistency metrics", "Content cadence table", "Channel charts", "Brand health cards", "Recommendation panel"],
    dataContract: ["brands[]", "channels[]", "content[]"],
    components: ["ProjectSwitcher", "MetricGrid", "KpiCard", "ChartPanel", "DataTable", "RecommendationCard"],
    validation: ["Shows enabled/disabled brand state", "Separates production from channel performance", "Makes missed cadence visible"],
    icon: Radio,
  },
  {
    id: "system-health-deployment",
    title: "System Health And Deployment Dashboard",
    useFor: "Production deployments, CI failures, dashboard manifests, service health, environment variables, DNS/routes, and release readiness.",
    layout: ["Environment/deploy header", "Service health cards", "Deployment timeline", "CI/check table", "Manifest health", "Secrets readiness", "Release action bar"],
    dataContract: ["services[]", "deployments[]", "checks[]", "secrets[]"],
    components: ["DashboardLauncher", "HealthBadge", "ActivityTimeline", "DataTable", "CommandBar", "AuditEventList"],
    validation: ["Never prints secret values", "Distinguishes production, local, and unknown URLs", "Shows next verification action"],
    icon: BarChart3,
  },
];
