export interface LiveSignalIntegration {
  id: string;
  project: string;
  endpoint: string;
  signals: string[];
  status: "ready" | "partial" | "missing";
  nextStep: string;
}

export interface RoutedTask {
  id: string;
  title: string;
  source: string;
  owner: string;
  priority: "low" | "normal" | "high" | "critical";
  status: "queued" | "assigned" | "blocked" | "done";
  nextStep: string;
}

export interface DecisionRecord {
  id: string;
  decision: string;
  reason: string;
  owner: string;
  status: "active" | "superseded" | "needs-review";
  reviewedAt: string;
}

export interface ModelRoutingPolicy {
  id: string;
  taskType: string;
  preferred: string;
  fallback: string;
  costMode: "free-local-first" | "cheap-api-first" | "premium-approval" | "blocked";
  approvalRequired: boolean;
}

export interface OperatingLoop {
  id: string;
  name: string;
  cadence: string;
  owner: string;
  status: "ready" | "draft" | "paused";
  output: string;
}

export interface PermissionPolicy {
  id: string;
  action: string;
  level: "viewer" | "operator" | "admin";
  approval: "none" | "confirm" | "explicit";
  audit: boolean;
}

export interface BusinessScorecard {
  id: string;
  business: string;
  health: number;
  revenueSignal: string;
  costSignal: string;
  operatingFocus: string;
}

export interface OperatingSystemStage {
  version:
    | "V22"
    | "V23"
    | "V24"
    | "V25"
    | "V26"
    | "V27"
    | "V28"
    | "V29"
    | "V30"
    | "V31"
    | "V32"
    | "V33"
    | "V34"
    | "V35"
    | "V36"
    | "V37"
    | "V38"
    | "V39"
    | "V40"
    | "V41"
    | "V42"
    | "V43"
    | "V44"
    | "V45"
    | "V46"
    | "V47"
    | "V48"
    | "V49"
    | "V50"
    | "V51"
    | "V52"
    | "V53"
    | "V54"
    | "V55"
    | "V56"
    | "V57"
    | "V58"
    | "V59"
    | "V60"
    | "V61"
    | "V62"
    | "V63"
    | "V64"
    | "V65"
    | "V66"
    | "V67"
    | "V68"
    | "V69"
    | "V70"
    | "V71"
    | "V72"
    | "V73"
    | "V74"
    | "V75"
    | "V76"
    | "V77"
    | "V78"
    | "V79"
    | "V80";
  route: string;
  title: string;
  eyebrow: string;
  description: string;
  sectionTitle: string;
  sectionDescription: string;
  status: "trackable" | "gated" | "ready";
  progress: number;
  owner: string;
  primaryMetric: string;
  risk: "low" | "medium" | "high";
  cards: Array<{
    label: string;
    value: string | number;
    detail: string;
    tone: "neutral" | "info" | "success" | "warning" | "critical";
  }>;
  rows: Array<{
    id: string;
    capability: string;
    state: "built" | "gated" | "planned" | "blocked";
    owner: string;
    nextStep: string;
  }>;
}

export const liveSignalIntegrations: LiveSignalIntegration[] = [
  {
    id: "hermes",
    project: "Hermes",
    endpoint: "/api/status + dashboard plugins",
    signals: ["HealthSnapshot", "DashboardSnapshot", "ActionNeeded"],
    status: "ready",
    nextStep: "Promote central command into production dashboard flow.",
  },
  {
    id: "media-engine",
    project: "Media Engine",
    endpoint: "https://media.tlccapitalgroup.com/dashboard-snapshot",
    signals: ["HealthSnapshot", "CapacitySnapshot", "QueueSnapshot", "ActionNeeded"],
    status: "partial",
    nextStep: "Add dashboard-snapshot endpoint after package-native migration work resumes.",
  },
  {
    id: "khashi-vc",
    project: "Khashi VC",
    endpoint: "https://roc.tlccapitalgroup.com/dashboard-snapshot",
    signals: ["HealthSnapshot", "CostSnapshot", "CapacitySnapshot", "ResearchSignal"],
    status: "partial",
    nextStep: "Add ROC snapshot endpoint after scheduler/market data contracts settle.",
  },
];

export const routedTasks: RoutedTask[] = [
  {
    id: "task-signal-endpoints",
    title: "Define project dashboard-snapshot endpoints",
    source: "V15 signal integration",
    owner: "Hermes",
    priority: "high",
    status: "assigned",
    nextStep: "Use V9 DashboardSnapshot contract as the required schema.",
  },
  {
    id: "task-model-routing",
    title: "Route coding tasks through cost-aware model policy",
    source: "V18 model routing",
    owner: "Hermes",
    priority: "normal",
    status: "queued",
    nextStep: "Use local Codex when available; require approval for premium API fallback.",
  },
  {
    id: "task-permission-rails",
    title: "Apply permission tiers before autonomous actions",
    source: "V20 security",
    owner: "Operations",
    priority: "critical",
    status: "assigned",
    nextStep: "Keep admin actions explicit until audit logging is live.",
  },
];

export const decisionLedger: DecisionRecord[] = [
  {
    id: "decision-delay-khashi-media-migrations",
    decision: "Delay Media Engine and Khashi VC package-native migrations.",
    reason: "Build control-plane governance through V21 before touching production-heavy dashboards.",
    owner: "Hermes",
    status: "active",
    reviewedAt: "2026-07-17",
  },
  {
    id: "decision-v15-v21-order",
    decision: "Build live signals, memory, model routing, task routing, permissions, loops, then business OS.",
    reason: "Usefulness improves without letting autonomy outrun safety and memory.",
    owner: "Hermes",
    status: "active",
    reviewedAt: "2026-07-17",
  },
];

export const modelRoutingPolicies: ModelRoutingPolicy[] = [
  {
    id: "coding-premium",
    taskType: "High-risk coding or production dashboard changes",
    preferred: "Local Codex session",
    fallback: "OpenAI API premium model",
    costMode: "premium-approval",
    approvalRequired: true,
  },
  {
    id: "analysis-cheap",
    taskType: "Summaries, classifications, routine analysis",
    preferred: "DeepSeek/Kimi/MiniMax class model",
    fallback: "OpenAI mini model",
    costMode: "cheap-api-first",
    approvalRequired: false,
  },
  {
    id: "local-first",
    taskType: "Repo inspection, planning, validation",
    preferred: "Local tools and Codex",
    fallback: "No API fallback without approval",
    costMode: "free-local-first",
    approvalRequired: false,
  },
];

export const operatingLoops: OperatingLoop[] = [
  {
    id: "daily-brief",
    name: "Daily cross-project brief",
    cadence: "Daily morning",
    owner: "Hermes",
    status: "ready",
    output: "What changed, what matters, and what needs attention.",
  },
  {
    id: "weekly-dashboard-review",
    name: "Weekly dashboard quality review",
    cadence: "Weekly",
    owner: "Hermes",
    status: "draft",
    output: "Dashboard scorecard, stale routes, failing checks, and migration priorities.",
  },
  {
    id: "cost-capacity-watch",
    name: "Cost and capacity watch",
    cadence: "Daily",
    owner: "Operations",
    status: "draft",
    output: "Token/API/storage/CPU pressure and recommended throttles.",
  },
];

export const permissionPolicies: PermissionPolicy[] = [
  { id: "view-dashboards", action: "View dashboards and reports", level: "viewer", approval: "none", audit: false },
  { id: "run-refresh", action: "Refresh signals and run read-only checks", level: "operator", approval: "none", audit: true },
  { id: "start-autopilot", action: "Start or stop operational autopilot", level: "operator", approval: "confirm", audit: true },
  { id: "deploy-production", action: "Deploy or rollback production dashboards", level: "admin", approval: "explicit", audit: true },
  { id: "change-secrets", action: "Change secrets or authentication", level: "admin", approval: "explicit", audit: true },
];

export const businessScorecards: BusinessScorecard[] = [
  {
    id: "tlc-holding",
    business: "TLC Capital Group OS",
    health: 72,
    revenueSignal: "not connected",
    costSignal: "partial",
    operatingFocus: "Central command, permissions, model routing, and dashboard signals.",
  },
  {
    id: "media-business",
    business: "Media Business",
    health: 64,
    revenueSignal: "not connected",
    costSignal: "partial",
    operatingFocus: "Consistent brand generation, channel analytics, publishing quality.",
  },
  {
    id: "research-investing",
    business: "Research / Investing",
    health: 68,
    revenueSignal: "strategy evidence only",
    costSignal: "partial",
    operatingFocus: "Market selection, experiment evidence, category/tag signal quality.",
  },
];

export const operatingSystemStages: OperatingSystemStage[] = [
  {
    version: "V22",
    route: "/project-snapshots",
    title: "Live Project Snapshot Contracts",
    eyebrow: "V22 live truth layer",
    description: "Standardizes the snapshot payload every project must expose before Hermes can treat dashboard data as operating truth.",
    sectionTitle: "Snapshot Contract Registry",
    sectionDescription: "The registry defines the contract now; each project still owns the production endpoint implementation.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 signal families",
    risk: "medium",
    cards: [
      { label: "Signal Families", value: 6, detail: "health, cost, capacity, queue, actions, risk", tone: "success" },
      { label: "Projects Ready", value: 1, detail: "Hermes reference source", tone: "warning" },
      { label: "Gated Endpoints", value: 2, detail: "Media Engine and Khashi VC still need project work", tone: "warning" },
    ],
    rows: [
      { id: "snapshot-schema", capability: "DashboardSnapshot schema", state: "built", owner: "Hermes", nextStep: "Use as the required project endpoint payload." },
      { id: "stale-detection", capability: "Stale and missing signal detection", state: "built", owner: "Hermes", nextStep: "Connect real endpoint timestamps once project APIs exist." },
      { id: "project-adapters", capability: "Project-owned endpoint adapters", state: "gated", owner: "Project teams", nextStep: "Resume Media Engine and Khashi VC migrations." },
    ],
  },
  {
    version: "V23",
    route: "/durable-memory",
    title: "Durable Memory And Decision Store",
    eyebrow: "V23 persistence layer",
    description: "Moves task, decision, review, and context records from static dashboard data toward durable operating memory.",
    sectionTitle: "Memory Store Readiness",
    sectionDescription: "The data model is trackable now; database migrations should be applied when runtime storage is selected.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "4 record types",
    risk: "medium",
    cards: [
      { label: "Record Types", value: 4, detail: "decisions, tasks, reviews, context links", tone: "success" },
      { label: "Review States", value: 3, detail: "active, superseded, needs-review", tone: "info" },
      { label: "Persistence Hook", value: "gated", detail: "awaits storage migration decision", tone: "warning" },
    ],
    rows: [
      { id: "decision-store", capability: "Decision records with reasons", state: "built", owner: "Hermes", nextStep: "Persist records beyond the frontend registry." },
      { id: "task-store", capability: "Task and next-step memory", state: "built", owner: "Hermes", nextStep: "Link routed tasks to projects and artifacts." },
      { id: "memory-db", capability: "Durable database store", state: "gated", owner: "Hermes", nextStep: "Choose SQLite/Postgres-backed runtime store." },
    ],
  },
  {
    version: "V24",
    route: "/permission-runtime",
    title: "Permission Enforcement Runtime",
    eyebrow: "V24 safety runtime",
    description: "Turns permission policy into an enforcement contract before Hermes can execute production-affecting commands.",
    sectionTitle: "Runtime Enforcement Gates",
    sectionDescription: "Policy coverage is complete; execution middleware and audit persistence remain the live-runtime hooks.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 guarded actions",
    risk: "high",
    cards: [
      { label: "Guarded Actions", value: 5, detail: "view, refresh, autopilot, deploy, secrets", tone: "critical" },
      { label: "Explicit Gates", value: 2, detail: "deploy and secret changes", tone: "critical" },
      { label: "Audit Required", value: 4, detail: "operator/admin actions", tone: "success" },
    ],
    rows: [
      { id: "policy-check", capability: "Permission check contract", state: "built", owner: "Hermes", nextStep: "Call before every command execution." },
      { id: "approval-gate", capability: "Explicit approval gate", state: "built", owner: "Operations", nextStep: "Wire into deploy and secret-changing commands." },
      { id: "audit-store", capability: "Runtime audit persistence", state: "gated", owner: "Operations", nextStep: "Persist approvals, denials, and command outputs." },
    ],
  },
  {
    version: "V25",
    route: "/cost-governor",
    title: "Model Router And Cost Governor",
    eyebrow: "V25 cost intelligence",
    description: "Makes model choice governable by availability, task type, expected quality, spend risk, and approval rules.",
    sectionTitle: "Provider Routing Policy",
    sectionDescription: "Routing policy is modeled now; live provider latency, cost, and quality scoring can be attached later.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "3 routing modes",
    risk: "medium",
    cards: [
      { label: "Routing Modes", value: 3, detail: "local, cheap API, premium approval", tone: "success" },
      { label: "Approval Gates", value: 1, detail: "premium model fallback", tone: "warning" },
      { label: "Provider Scores", value: "planned", detail: "quality/cost history not live yet", tone: "warning" },
    ],
    rows: [
      { id: "local-first", capability: "Local Codex availability path", state: "built", owner: "Hermes", nextStep: "Add runtime health probe for local worker." },
      { id: "cheap-routing", capability: "Cheap model routing mode", state: "built", owner: "Hermes", nextStep: "Add provider adapters for DeepSeek/Kimi/MiniMax style models." },
      { id: "quality-score", capability: "Outcome quality and cost scoring", state: "gated", owner: "Hermes", nextStep: "Record task outcomes by provider before automatic fallback." },
    ],
  },
  {
    version: "V26",
    route: "/loop-runner",
    title: "Operating Loop Runner",
    eyebrow: "V26 scheduled operations",
    description: "Defines the runner needed to execute daily briefs, weekly reviews, cost watches, and project health loops safely.",
    sectionTitle: "Loop Execution Readiness",
    sectionDescription: "Loop definitions are complete; actual scheduling waits for permissions, audit, and snapshot inputs.",
    status: "gated",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "3 loop types",
    risk: "high",
    cards: [
      { label: "Loop Types", value: 3, detail: "daily brief, quality review, cost watch", tone: "success" },
      { label: "Dry Run", value: "ready", detail: "manual safe path first", tone: "info" },
      { label: "Autonomy Gate", value: "locked", detail: "requires V24 enforcement", tone: "critical" },
    ],
    rows: [
      { id: "manual-runner", capability: "Manual loop run contract", state: "built", owner: "Hermes", nextStep: "Expose run controls after permission middleware." },
      { id: "dry-run-output", capability: "Dry-run output artifact", state: "built", owner: "Hermes", nextStep: "Store generated loop reports." },
      { id: "scheduler", capability: "Production scheduler", state: "gated", owner: "Operations", nextStep: "Enable only after audit and kill switch exist." },
    ],
  },
  {
    version: "V27",
    route: "/business-command",
    title: "Cross-Business Command Center",
    eyebrow: "V27 holding-company view",
    description: "Turns business scorecards into a command center with business-unit health, trend, attention, revenue, and cost signals.",
    sectionTitle: "Business Command Rollups",
    sectionDescription: "The command model is available now; live revenue and analytics feeds are the next data layer.",
    status: "trackable",
    progress: 100,
    owner: "TLC Capital Group OS",
    primaryMetric: "3 business units",
    risk: "medium",
    cards: [
      { label: "Business Units", value: 3, detail: "holding, media, research/investing", tone: "success" },
      { label: "Attention Queue", value: "modeled", detail: "needs live signal inputs", tone: "info" },
      { label: "Revenue Feeds", value: "gated", detail: "not connected yet", tone: "warning" },
    ],
    rows: [
      { id: "business-map", capability: "Project-to-business mapping", state: "built", owner: "Hermes", nextStep: "Register each future project under a business unit." },
      { id: "attention-rollup", capability: "Attention and action rollups", state: "built", owner: "Hermes", nextStep: "Feed from live project snapshots." },
      { id: "finance-feed", capability: "Revenue and finance signals", state: "gated", owner: "TLC Capital Group OS", nextStep: "Connect accounting or business KPI source." },
    ],
  },
  {
    version: "V28",
    route: "/agent-workbench",
    title: "Agent Workbench",
    eyebrow: "V28 supervised execution",
    description: "Gives Hermes a workspace for agent plans, approvals, artifacts, execution evidence, and completion reports.",
    sectionTitle: "Supervised Workbench Flow",
    sectionDescription: "The workbench contract is ready; live execution should stay approval-gated until V24 and V30 are active.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "5 workflow steps",
    risk: "high",
    cards: [
      { label: "Workflow Steps", value: 5, detail: "plan, approve, execute, evidence, report", tone: "success" },
      { label: "Artifact Slots", value: 4, detail: "diffs, screenshots, logs, reports", tone: "info" },
      { label: "Execution Gate", value: "explicit", detail: "high-risk work requires approval", tone: "critical" },
    ],
    rows: [
      { id: "plan-flow", capability: "Plan-to-approval flow", state: "built", owner: "Hermes", nextStep: "Connect routed tasks to workbench plans." },
      { id: "artifact-evidence", capability: "Artifact and evidence slots", state: "built", owner: "Hermes", nextStep: "Attach screenshots, logs, diffs, and validation reports." },
      { id: "live-execution", capability: "Live execution bridge", state: "gated", owner: "Operations", nextStep: "Require permission runtime and audit store first." },
    ],
  },
  {
    version: "V29",
    route: "/evaluation-gates",
    title: "Evaluation And Quality Gates",
    eyebrow: "V29 quality controls",
    description: "Adds measurable gates for coding, design, dashboard quality, model performance, and production promotion.",
    sectionTitle: "Quality Gate Registry",
    sectionDescription: "The gate model is complete; real scoring improves as model/provider/task history accumulates.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "5 gate families",
    risk: "medium",
    cards: [
      { label: "Gate Families", value: 5, detail: "code, design, dashboard, model, production", tone: "success" },
      { label: "Visual Coverage", value: "live", detail: "Playwright dashboard checks", tone: "success" },
      { label: "Model Evals", value: "planned", detail: "requires task outcome history", tone: "warning" },
    ],
    rows: [
      { id: "dashboard-gates", capability: "Dashboard visual/accessibility gates", state: "built", owner: "Hermes", nextStep: "Keep expanding route coverage." },
      { id: "model-evals", capability: "Model quality comparison gates", state: "built", owner: "Hermes", nextStep: "Record provider outcomes and costs." },
      { id: "promotion-policy", capability: "Promotion threshold policy", state: "gated", owner: "Operations", nextStep: "Block production promotion below defined scores." },
    ],
  },
  {
    version: "V30",
    route: "/autonomy-readiness",
    title: "Production Autonomy Readiness",
    eyebrow: "V30 autonomy launch gate",
    description: "Defines readiness levels, kill switches, budget breakers, audit requirements, and project autonomy modes.",
    sectionTitle: "Autonomy Readiness Checklist",
    sectionDescription: "The readiness framework is complete; autonomy should only move up when gates, audits, and rollback controls are live.",
    status: "ready",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 autonomy levels",
    risk: "high",
    cards: [
      { label: "Autonomy Levels", value: 5, detail: "manual to limited autonomous", tone: "success" },
      { label: "Kill Switch", value: "required", detail: "must exist before scheduled execution", tone: "critical" },
      { label: "Budget Breakers", value: "required", detail: "stop expensive runaway loops", tone: "critical" },
    ],
    rows: [
      { id: "autonomy-levels", capability: "Autonomy level framework", state: "built", owner: "Operations", nextStep: "Assign every project a maximum allowed level." },
      { id: "kill-switch", capability: "Global and per-project kill switch", state: "built", owner: "Operations", nextStep: "Wire to runtime runner before autonomous execution." },
      { id: "incident-review", capability: "Incident and rollback review log", state: "gated", owner: "Operations", nextStep: "Persist incidents after runtime execution exists." },
    ],
  },
  {
    version: "V31",
    route: "/project-registry",
    title: "Production Project Registry",
    eyebrow: "V31 project truth layer",
    description: "Turns the dashboard registry into a governed production catalog with ownership, route, deployment, health, and snapshot expectations for every TLC project.",
    sectionTitle: "Project Registry Coverage",
    sectionDescription: "Every project should have one owner, one production URL policy, one health endpoint, and one dashboard snapshot contract before Hermes treats it as a managed operating unit.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "8 registered dashboards",
    risk: "medium",
    cards: [
      { label: "Registered Projects", value: 8, detail: "root registry plus sibling manifests", tone: "success" },
      { label: "Snapshot Contracts", value: 2, detail: "Khashi VC and Media Engine expose project-owned snapshots", tone: "info" },
      { label: "Route Verification", value: "gated", detail: "needs production Caddy/DNS health sweep", tone: "warning" },
    ],
    rows: [
      { id: "registry-source", capability: "Single root registry plus sibling manifest merge", state: "built", owner: "Hermes", nextStep: "Keep root registry as the production launcher source." },
      { id: "ownership-map", capability: "Owner and business-unit map", state: "built", owner: "TLC Capital Group OS", nextStep: "Require owners for every future dashboard manifest." },
      { id: "production-verification", capability: "Production route verification", state: "gated", owner: "Operations", nextStep: "Run live DNS, Caddy, health, and snapshot checks before promotion." },
    ],
  },
  {
    version: "V32",
    route: "/telemetry-fabric",
    title: "Telemetry Fabric",
    eyebrow: "V32 observability layer",
    description: "Standardizes logs, health, API usage, storage growth, scheduler state, and queue pressure across dashboards so Hermes can reason from live operating signals.",
    sectionTitle: "Telemetry Signal Fabric",
    sectionDescription: "The fabric defines required telemetry families and freshness rules; project adapters fill the signals as each system matures.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 telemetry families",
    risk: "medium",
    cards: [
      { label: "Signal Families", value: 6, detail: "health, logs, cost, capacity, storage, queues", tone: "success" },
      { label: "Freshness Rules", value: "modeled", detail: "stale and missing signals are visible", tone: "info" },
      { label: "Live Adapters", value: "partial", detail: "project-owned telemetry still expanding", tone: "warning" },
    ],
    rows: [
      { id: "telemetry-contract", capability: "Telemetry contract and signal families", state: "built", owner: "Hermes", nextStep: "Use DashboardSnapshot as the first transport." },
      { id: "freshness-policy", capability: "Freshness and stale-signal handling", state: "built", owner: "Hermes", nextStep: "Display stale signals in executive and project pages." },
      { id: "project-instrumentation", capability: "Project-specific instrumentation", state: "gated", owner: "Project teams", nextStep: "Add usage, storage, and queue counters inside each production project." },
    ],
  },
  {
    version: "V33",
    route: "/incident-command",
    title: "Incident Command",
    eyebrow: "V33 response layer",
    description: "Creates a command view for production incidents, blocked dashboards, failed jobs, stale data, auth issues, and deployment regressions.",
    sectionTitle: "Incident Response Queue",
    sectionDescription: "Incident command keeps the system from silently drifting by making severity, owner, next step, and rollback path explicit.",
    status: "trackable",
    progress: 100,
    owner: "Operations",
    primaryMetric: "4 severity levels",
    risk: "high",
    cards: [
      { label: "Severity Levels", value: 4, detail: "info, warning, high, critical", tone: "success" },
      { label: "Owner Required", value: "yes", detail: "no orphan incidents", tone: "success" },
      { label: "Auto-Remediation", value: "locked", detail: "manual approval until V40 governance is live", tone: "critical" },
    ],
    rows: [
      { id: "incident-schema", capability: "Incident schema with severity and owner", state: "built", owner: "Operations", nextStep: "Route dashboard health failures into the queue." },
      { id: "rollback-path", capability: "Rollback and mitigation field", state: "built", owner: "Operations", nextStep: "Attach deployment version and rollback command evidence." },
      { id: "auto-remediation", capability: "Automated remediation", state: "gated", owner: "Operations", nextStep: "Require permission runtime, audit store, and kill switch before enabling." },
    ],
  },
  {
    version: "V34",
    route: "/deployment-promotion",
    title: "Deployment Promotion Rail",
    eyebrow: "V34 ship gate",
    description: "Defines one promotion rail for dashboard deployments: validate, build, test, migrate, deploy, health-check, screenshot, and rollback evidence.",
    sectionTitle: "Promotion Gate Checklist",
    sectionDescription: "Promotion becomes a governed workflow instead of a per-project habit, while still letting each project own its app-specific deploy details.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "7 promotion gates",
    risk: "high",
    cards: [
      { label: "Promotion Gates", value: 7, detail: "validate through rollback evidence", tone: "success" },
      { label: "Shared Rail", value: "ready", detail: "Hermes/Hetzner promotion pattern", tone: "info" },
      { label: "Auto Deploy", value: "gated", detail: "requires live secret and approval checks", tone: "critical" },
    ],
    rows: [
      { id: "promotion-checklist", capability: "Promotion checklist and evidence model", state: "built", owner: "Operations", nextStep: "Require checklist before marking deployments current." },
      { id: "migration-step", capability: "Migration-aware deploy step", state: "built", owner: "Project teams", nextStep: "Declare whether a Prisma/database migration is required." },
      { id: "auto-promotion", capability: "Automatic production promotion", state: "gated", owner: "Operations", nextStep: "Block unless V24, V29, and V30 gates pass." },
    ],
  },
  {
    version: "V35",
    route: "/secrets-posture",
    title: "Secrets And Access Posture",
    eyebrow: "V35 access governance",
    description: "Tracks required GitHub secrets, production environment variables, SSH keys, dashboard credentials, token rotation, and missing-secret blockers.",
    sectionTitle: "Secrets Posture Matrix",
    sectionDescription: "Hermes should know which projects are deployable without ever exposing secret values in the dashboard.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 secret classes",
    risk: "high",
    cards: [
      { label: "Secret Classes", value: 5, detail: "SSH, app env, dashboard auth, API keys, webhooks", tone: "success" },
      { label: "Value Exposure", value: "blocked", detail: "presence only, never raw values", tone: "success" },
      { label: "Rotation Cadence", value: "planned", detail: "manual until vault integration exists", tone: "warning" },
    ],
    rows: [
      { id: "presence-check", capability: "Secret presence and scope checks", state: "built", owner: "Operations", nextStep: "Report missing names without printing values." },
      { id: "access-map", capability: "Project access and deploy key map", state: "built", owner: "Operations", nextStep: "Track which projects share Hetzner deploy rails." },
      { id: "rotation-policy", capability: "Rotation and vault-backed secrets", state: "gated", owner: "Operations", nextStep: "Choose vault or managed secret backend before automation." },
    ],
  },
  {
    version: "V36",
    route: "/data-source-catalog",
    title: "Data Source Catalog",
    eyebrow: "V36 data governance",
    description: "Catalogs every external and internal data source, freshness rule, ingestion owner, retention rule, failure mode, and dashboard consumer.",
    sectionTitle: "Source Coverage Catalog",
    sectionDescription: "The catalog prevents hidden dependencies by making each dashboard's data lineage and freshness expectation visible.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 source fields",
    risk: "medium",
    cards: [
      { label: "Source Fields", value: 6, detail: "owner, cadence, freshness, cost, retention, consumers", tone: "success" },
      { label: "Retention Rules", value: "modeled", detail: "supports generated media and market data pressure", tone: "info" },
      { label: "Lineage Checks", value: "gated", detail: "requires project adapters to report dependencies", tone: "warning" },
    ],
    rows: [
      { id: "source-schema", capability: "Data source schema", state: "built", owner: "Hermes", nextStep: "Use for Khashi, Media Engine, and future dashboards." },
      { id: "retention-policy", capability: "Retention and storage pressure policy", state: "built", owner: "Operations", nextStep: "Attach generated image, market, and telemetry stores." },
      { id: "lineage-automation", capability: "Automated lineage discovery", state: "gated", owner: "Hermes", nextStep: "Scan project code and manifests after adapters stabilize." },
    ],
  },
  {
    version: "V37",
    route: "/finance-attribution",
    title: "Finance And Cost Attribution",
    eyebrow: "V37 operating economics",
    description: "Attributes model spend, API calls, storage, hosting, labor proxies, and business-unit costs so Hermes can explain what each operating loop costs.",
    sectionTitle: "Cost Attribution Model",
    sectionDescription: "This makes cost visible without pretending exact vendor invoices exist for every resource yet.",
    status: "trackable",
    progress: 100,
    owner: "TLC Capital Group OS",
    primaryMetric: "5 cost buckets",
    risk: "medium",
    cards: [
      { label: "Cost Buckets", value: 5, detail: "model, API, storage, hosting, labor proxy", tone: "success" },
      { label: "Exact Invoices", value: "partial", detail: "manual rates needed for some costs", tone: "warning" },
      { label: "Business Rollup", value: "ready", detail: "maps costs to project and business unit", tone: "info" },
    ],
    rows: [
      { id: "cost-schema", capability: "Cost attribution schema", state: "built", owner: "Hermes", nextStep: "Record cost bucket, project, and business unit." },
      { id: "manual-rate-table", capability: "Manual rate table for non-metered resources", state: "built", owner: "TLC Capital Group OS", nextStep: "Add monthly hosting and service assumptions when available." },
      { id: "invoice-reconciliation", capability: "Invoice reconciliation", state: "gated", owner: "Finance", nextStep: "Connect actual billing exports or manual monthly imports." },
    ],
  },
  {
    version: "V38",
    route: "/learning-engine",
    title: "Learning Engine",
    eyebrow: "V38 evidence loop",
    description: "Turns experiments, dashboard failures, generation outcomes, model choices, and project incidents into reusable learning records and strategy evidence.",
    sectionTitle: "Learning Evidence Loop",
    sectionDescription: "The learning engine separates useful findings from noise and shows what should change because of the evidence.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "4 learning loops",
    risk: "medium",
    cards: [
      { label: "Learning Loops", value: 4, detail: "experiments, content, deploys, model outcomes", tone: "success" },
      { label: "Evidence Promotion", value: "modeled", detail: "candidate to finding to policy", tone: "info" },
      { label: "Auto Policy Changes", value: "gated", detail: "requires approval and regression checks", tone: "critical" },
    ],
    rows: [
      { id: "learning-record", capability: "Learning record and evidence lifecycle", state: "built", owner: "Hermes", nextStep: "Collect evidence from Khashi and Media Engine outcomes." },
      { id: "promotion-rule", capability: "Finding promotion rules", state: "built", owner: "Hermes", nextStep: "Require sample size, confidence, and counterevidence." },
      { id: "auto-policy-update", capability: "Automated policy updates", state: "gated", owner: "Operations", nextStep: "Require explicit approval until evals prove reliability." },
    ],
  },
  {
    version: "V39",
    route: "/agent-eval-lab",
    title: "Agent Evaluation Lab",
    eyebrow: "V39 model proof layer",
    description: "Compares Codex, local agents, DeepSeek/Kimi/MiniMax-style providers, and premium fallback models on real TLC tasks before routing work automatically.",
    sectionTitle: "Provider Evaluation Matrix",
    sectionDescription: "This lab answers whether cheaper models are good enough for each task family with evidence instead of vibes.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "5 eval dimensions",
    risk: "medium",
    cards: [
      { label: "Eval Dimensions", value: 5, detail: "correctness, tests, design, cost, latency", tone: "success" },
      { label: "Task Families", value: "modeled", detail: "coding, research, dashboard, content, ops", tone: "info" },
      { label: "Auto Routing", value: "gated", detail: "requires outcome history", tone: "warning" },
    ],
    rows: [
      { id: "eval-matrix", capability: "Provider eval matrix", state: "built", owner: "Hermes", nextStep: "Score providers by task family and validation result." },
      { id: "golden-tasks", capability: "Golden task set", state: "built", owner: "Hermes", nextStep: "Capture representative TLC tasks and expected validation." },
      { id: "auto-provider-routing", capability: "Automatic provider routing", state: "gated", owner: "Hermes", nextStep: "Require enough scored outcomes before replacing manual choice." },
    ],
  },
  {
    version: "V40",
    route: "/executive-cockpit",
    title: "Executive Cockpit",
    eyebrow: "V40 CEO layer",
    description: "Combines project registry, telemetry, incidents, costs, learning, deployments, and autonomy readiness into one accountable executive operating view.",
    sectionTitle: "Executive Operating Cockpit",
    sectionDescription: "V40 is the CEO view for TLC Capital Group OS: what is healthy, what is expensive, what is blocked, what changed, and what needs approval.",
    status: "ready",
    progress: 100,
    owner: "TLC Capital Group OS",
    primaryMetric: "7 executive signals",
    risk: "high",
    cards: [
      { label: "Executive Signals", value: 7, detail: "health, cost, incidents, deploys, learning, revenue, autonomy", tone: "success" },
      { label: "Approval Surface", value: "explicit", detail: "high-risk actions require operator approval", tone: "critical" },
      { label: "Board Narrative", value: "ready", detail: "daily/weekly summaries can be generated", tone: "info" },
    ],
    rows: [
      { id: "executive-rollup", capability: "Cross-project executive rollup", state: "built", owner: "Hermes", nextStep: "Feed from V31-V39 runtime signals." },
      { id: "approval-agenda", capability: "Approval and decision agenda", state: "built", owner: "TLC Capital Group OS", nextStep: "Show actions awaiting explicit human approval." },
      { id: "ceo-autonomy-mode", capability: "CEO-level autonomy mode controls", state: "gated", owner: "Operations", nextStep: "Keep autonomous execution below approved project limits." },
    ],
  },
  {
    version: "V41",
    route: "/production-verification",
    title: "Live Production Verification Runner",
    eyebrow: "V41 live verification",
    description: "Runs governed production checks across dashboard URLs, health endpoints, snapshot endpoints, auth posture, and visual evidence before Hermes trusts a deployment.",
    sectionTitle: "Production Verification Checks",
    sectionDescription: "V41 turns production verification from a manual question into a repeatable evidence record.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 check classes",
    risk: "high",
    cards: [
      { label: "Check Classes", value: 5, detail: "DNS, Caddy, health, snapshot, screenshot", tone: "success" },
      { label: "Runtime Endpoint", value: "live", detail: "/api/operating-runtime/production-checks", tone: "info" },
      { label: "Network Sweep", value: "gated", detail: "operator-triggered only", tone: "warning" },
    ],
    rows: [
      { id: "prod-check-records", capability: "Production check evidence records", state: "built", owner: "Operations", nextStep: "Record health and screenshot outcomes per dashboard." },
      { id: "snapshot-check", capability: "Snapshot contract verification", state: "built", owner: "Hermes", nextStep: "Require a valid dashboard snapshot before marking a project current." },
      { id: "live-network-runner", capability: "Live DNS/Caddy/screenshot runner", state: "gated", owner: "Operations", nextStep: "Run only from approved production verification workflow." },
    ],
  },
  {
    version: "V42",
    route: "/command-gates",
    title: "Command Gate Runtime",
    eyebrow: "V42 permission enforcement",
    description: "Wraps production-affecting commands with permission decisions, explicit approvals, audit records, and deny-by-default behavior.",
    sectionTitle: "Command Gate Matrix",
    sectionDescription: "V42 is the enforcement layer between Hermes intent and live command execution.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "4 command risks",
    risk: "high",
    cards: [
      { label: "Command Risks", value: 4, detail: "deploy, secrets, scheduler, autonomy", tone: "critical" },
      { label: "Audit Records", value: "durable", detail: "/api/operating-runtime/permission-decision", tone: "success" },
      { label: "Bypass Policy", value: "blocked", detail: "high-risk commands require explicit approval", tone: "critical" },
    ],
    rows: [
      { id: "permission-decision", capability: "Permission decision middleware primitive", state: "built", owner: "Hermes", nextStep: "Call before every high-risk command handler." },
      { id: "audit-trail", capability: "Durable audit record", state: "built", owner: "Operations", nextStep: "Attach command payload, actor, and approval state." },
      { id: "full-command-wrap", capability: "Every live command wrapped", state: "gated", owner: "Project teams", nextStep: "Refactor deploy, scheduler, and secret routes through the gate." },
    ],
  },
  {
    version: "V43",
    route: "/telemetry-adapters",
    title: "Project Telemetry Adapter Kit",
    eyebrow: "V43 adapter kit",
    description: "Defines the adapter package each project uses to emit health, cost, storage, API, queue, deployment, and action-needed telemetry into Hermes.",
    sectionTitle: "Telemetry Adapter Contract",
    sectionDescription: "V43 makes project telemetry consistent enough for executive rollups and incident automation.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "7 signal fields",
    risk: "medium",
    cards: [
      { label: "Signal Fields", value: 7, detail: "health, cost, storage, API, queues, deploys, actions", tone: "success" },
      { label: "Transport", value: "snapshot", detail: "DashboardSnapshot plus runtime evidence", tone: "info" },
      { label: "Adoption", value: "partial", detail: "project teams still need adapters", tone: "warning" },
    ],
    rows: [
      { id: "adapter-contract", capability: "Telemetry adapter contract", state: "built", owner: "Hermes", nextStep: "Use for every dashboard snapshot producer." },
      { id: "standard-evidence", capability: "Standard runtime evidence ingestion", state: "built", owner: "Hermes", nextStep: "Record telemetry as evidence with freshness and owner." },
      { id: "project-adoption", capability: "All projects emitting rich telemetry", state: "gated", owner: "Project teams", nextStep: "Add adapters to Khashi, Media Engine, MBO, Business Mapper, and others." },
    ],
  },
  {
    version: "V44",
    route: "/incident-ingestion",
    title: "Incident Ingestion And Escalation",
    eyebrow: "V44 incident automation",
    description: "Turns failed health checks, stale snapshots, deployment failures, auth failures, and high-risk cost events into incident records.",
    sectionTitle: "Incident Ingestion Rules",
    sectionDescription: "V44 connects telemetry failures to a visible response queue without allowing unsupervised remediation.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 incident sources",
    risk: "high",
    cards: [
      { label: "Incident Sources", value: 5, detail: "health, stale, deploy, auth, cost", tone: "success" },
      { label: "Runtime Endpoint", value: "live", detail: "/api/operating-runtime/incidents", tone: "info" },
      { label: "Auto Remediate", value: "locked", detail: "explicit approval required", tone: "critical" },
    ],
    rows: [
      { id: "incident-recording", capability: "Runtime incident recording", state: "built", owner: "Operations", nextStep: "Create records from failed checks and operator reports." },
      { id: "severity-policy", capability: "Severity and owner policy", state: "built", owner: "Operations", nextStep: "Map critical failures to explicit owner and rollback path." },
      { id: "automatic-ingestion", capability: "Automatic incident ingestion from telemetry", state: "gated", owner: "Hermes", nextStep: "Enable after V41/V43 are producing reliable signals." },
    ],
  },
  {
    version: "V45",
    route: "/promotion-runner",
    title: "Shared Deployment Promotion Runner",
    eyebrow: "V45 deploy rail",
    description: "Standardizes production promotion through validation, build, migration, deployment, health checks, screenshots, evidence, and rollback records.",
    sectionTitle: "Promotion Runner Gates",
    sectionDescription: "V45 creates one deployment rail for dashboards instead of each project inventing its own deploy path.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "7 promotion gates",
    risk: "high",
    cards: [
      { label: "Promotion Gates", value: 7, detail: "validate, build, migrate, deploy, health, screenshot, rollback", tone: "success" },
      { label: "Runtime Endpoint", value: "live", detail: "/api/operating-runtime/deployments", tone: "info" },
      { label: "Actual Deploy", value: "gated", detail: "requires explicit command gate", tone: "critical" },
    ],
    rows: [
      { id: "deployment-evidence", capability: "Deployment evidence records", state: "built", owner: "Operations", nextStep: "Persist version, environment, migration, and rollback evidence." },
      { id: "promotion-checklist", capability: "Promotion checklist", state: "built", owner: "Hermes", nextStep: "Require V41 verification before marking current." },
      { id: "hetzner-runner", capability: "Live Hermes/Hetzner promotion runner", state: "gated", owner: "Operations", nextStep: "Wrap runner with V42 command gate." },
    ],
  },
  {
    version: "V46",
    route: "/secret-scanner",
    title: "Secrets Posture Scanner",
    eyebrow: "V46 secret safety",
    description: "Checks required secret names, scopes, deploy-key posture, and rotation status without exposing secret values.",
    sectionTitle: "Secret Presence Matrix",
    sectionDescription: "V46 makes missing secrets visible while keeping raw values out of dashboards, logs, and runtime payloads.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 secret classes",
    risk: "high",
    cards: [
      { label: "Secret Classes", value: 5, detail: "SSH, env, auth, API keys, webhooks", tone: "success" },
      { label: "Value Exposure", value: "never", detail: "presence and scope only", tone: "success" },
      { label: "Live Scanner", value: "gated", detail: "GitHub/Hetzner checks need approved tokens", tone: "warning" },
    ],
    rows: [
      { id: "presence-record", capability: "Secret presence evidence", state: "built", owner: "Operations", nextStep: "Record missing names without values." },
      { id: "scope-record", capability: "Scope and deploy-key posture", state: "built", owner: "Operations", nextStep: "Track which projects can deploy through shared rails." },
      { id: "live-secret-scan", capability: "Live GitHub/Hetzner scanner", state: "gated", owner: "Operations", nextStep: "Run only through approved admin credentials and audit." },
    ],
  },
  {
    version: "V47",
    route: "/cost-attribution-engine",
    title: "Cost Attribution Engine",
    eyebrow: "V47 cost runtime",
    description: "Turns model/API/storage/hosting/manual-rate inputs into project and business-unit cost attribution records.",
    sectionTitle: "Cost Attribution Inputs",
    sectionDescription: "V47 is the bridge from estimated operating cost to actual business-unit cost visibility.",
    status: "trackable",
    progress: 100,
    owner: "TLC Capital Group OS",
    primaryMetric: "5 cost inputs",
    risk: "medium",
    cards: [
      { label: "Cost Inputs", value: 5, detail: "model, API, storage, hosting, manual rates", tone: "success" },
      { label: "Runtime Endpoint", value: "live", detail: "/api/operating-runtime/costs", tone: "info" },
      { label: "Invoices", value: "manual", detail: "actual invoices still need import", tone: "warning" },
    ],
    rows: [
      { id: "cost-recording", capability: "Runtime cost records", state: "built", owner: "Hermes", nextStep: "Persist cost by project, unit, bucket, source, and period." },
      { id: "business-rollup", capability: "Business-unit attribution", state: "built", owner: "TLC Capital Group OS", nextStep: "Map every project to a business unit." },
      { id: "invoice-import", capability: "Invoice and rate import", state: "gated", owner: "Finance", nextStep: "Add billing exports or monthly manual rate sheets." },
    ],
  },
  {
    version: "V48",
    route: "/learning-ingestion",
    title: "Learning Ingestion Pipeline",
    eyebrow: "V48 evidence intake",
    description: "Ingests outcomes from experiments, generations, deployments, incidents, model choices, and operations into governed learning records.",
    sectionTitle: "Learning Ingestion Sources",
    sectionDescription: "V48 makes the system compound by turning operational outcomes into reusable evidence.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 learning sources",
    risk: "medium",
    cards: [
      { label: "Learning Sources", value: 6, detail: "experiments, content, deploys, incidents, models, ops", tone: "success" },
      { label: "Runtime Endpoint", value: "live", detail: "/api/operating-runtime/learning", tone: "info" },
      { label: "Auto Policy", value: "gated", detail: "approval required before policy changes", tone: "critical" },
    ],
    rows: [
      { id: "learning-records", capability: "Runtime learning records", state: "built", owner: "Hermes", nextStep: "Record candidates, findings, and recommendations." },
      { id: "source-linking", capability: "Outcome source linking", state: "built", owner: "Hermes", nextStep: "Attach source project and evidence count." },
      { id: "automatic-ingestion", capability: "Automatic learning ingestion", state: "gated", owner: "Project teams", nextStep: "Wire Khashi, Media Engine, deploys, and evals into the pipeline." },
    ],
  },
  {
    version: "V49",
    route: "/model-eval-harness",
    title: "Agent And Model Eval Harness",
    eyebrow: "V49 provider proof",
    description: "Runs repeatable golden tasks across local Codex, API models, and cheaper providers so routing decisions are evidence-backed.",
    sectionTitle: "Eval Harness Coverage",
    sectionDescription: "V49 protects quality and cost by proving which providers are good enough for each task family.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "5 eval scores",
    risk: "medium",
    cards: [
      { label: "Eval Scores", value: 5, detail: "correctness, tests, design, cost, latency", tone: "success" },
      { label: "Runtime Endpoint", value: "live", detail: "/api/operating-runtime/evals", tone: "info" },
      { label: "Auto Routing", value: "gated", detail: "needs enough scored outcomes", tone: "warning" },
    ],
    rows: [
      { id: "eval-records", capability: "Runtime provider eval records", state: "built", owner: "Hermes", nextStep: "Persist provider, task family, scores, and verdict." },
      { id: "golden-task-contract", capability: "Golden task contract", state: "built", owner: "Hermes", nextStep: "Define representative TLC coding, research, ops, and content tasks." },
      { id: "automatic-routing", capability: "Automatic model routing", state: "gated", owner: "Hermes", nextStep: "Require outcome history and approval thresholds." },
    ],
  },
  {
    version: "V50",
    route: "/circuit-breakers",
    title: "Runtime Circuit Breakers",
    eyebrow: "V50 autonomy brakes",
    description: "Implements the kill-switch, budget-breaker, provider-spend cap, loop-stop, and project autonomy limit records needed before serious autonomy.",
    sectionTitle: "Circuit Breaker Controls",
    sectionDescription: "V50 is the safety layer that lets Hermes become more autonomous without becoming uncontrolled.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 breaker classes",
    risk: "high",
    cards: [
      { label: "Breaker Classes", value: 5, detail: "kill, budget, provider, loop, project limit", tone: "critical" },
      { label: "Runtime Endpoint", value: "live", detail: "/api/operating-runtime/autonomy-controls", tone: "info" },
      { label: "Hard Enforcement", value: "gated", detail: "must wrap live execution paths", tone: "critical" },
    ],
    rows: [
      { id: "breaker-records", capability: "Runtime circuit breaker records", state: "built", owner: "Operations", nextStep: "Record enabled/disabled breaker controls per project." },
      { id: "budget-breakers", capability: "Budget and provider-spend breakers", state: "built", owner: "Operations", nextStep: "Attach to model and loop execution paths." },
      { id: "hard-enforcement", capability: "Hard execution-path enforcement", state: "gated", owner: "Operations", nextStep: "Block real commands when breakers are triggered." },
    ],
  },
  {
    version: "V51",
    route: "/production-sweep",
    title: "Production DNS And Health Sweep",
    eyebrow: "V51 live sweep",
    description: "Executes the V41 production verification contract against real dashboard URLs, DNS records, health endpoints, snapshot endpoints, and screenshot evidence.",
    sectionTitle: "Live Production Sweep",
    sectionDescription: "V51 turns the verification runner into a repeatable sweep that can prove a dashboard is reachable, healthy, and current.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "6 sweep checks",
    risk: "high",
    cards: [
      { label: "Sweep Checks", value: 6, detail: "DNS, TLS, Caddy, health, snapshot, screenshot", tone: "success" },
      { label: "Mode", value: "dry-run first", detail: "live network checks require approval", tone: "warning" },
      { label: "Evidence", value: "durable", detail: "results record to production-check evidence", tone: "info" },
    ],
    rows: [
      { id: "sweep-plan", capability: "Production sweep plan and payload", state: "built", owner: "Operations", nextStep: "Record target URLs and expected health routes." },
      { id: "dry-run-sweep", capability: "Dry-run sweep recording", state: "built", owner: "Hermes", nextStep: "Use before live network execution." },
      { id: "live-sweep", capability: "Approved live DNS/TLS/screenshot sweep", state: "gated", owner: "Operations", nextStep: "Run only with admin approval and V50 breaker checks." },
    ],
  },
  {
    version: "V52",
    route: "/hetzner-promotion-execution",
    title: "Hetzner Promotion Execution",
    eyebrow: "V52 deploy execution",
    description: "Runs the shared Hermes/Hetzner promotion rail with validation, migration awareness, deploy evidence, post-deploy health checks, and rollback records.",
    sectionTitle: "Promotion Execution Rail",
    sectionDescription: "V52 is the real deployment bridge from GitHub and local builds into the Hetzner production app directory.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "8 execution gates",
    risk: "high",
    cards: [
      { label: "Execution Gates", value: 8, detail: "validate, build, test, migrate, sync, restart, verify, rollback", tone: "critical" },
      { label: "Approval", value: "explicit", detail: "admin approval required for live deploy", tone: "critical" },
      { label: "Rollback", value: "recorded", detail: "rollback command or restore point required", tone: "info" },
    ],
    rows: [
      { id: "promotion-plan", capability: "Promotion execution plan", state: "built", owner: "Operations", nextStep: "Resolve project, app directory, branch, migration mode, and URL." },
      { id: "dry-run-promotion", capability: "Dry-run promotion evidence", state: "built", owner: "Hermes", nextStep: "Preview all gates before remote commands run." },
      { id: "live-promotion", capability: "Real Hetzner promotion execution", state: "gated", owner: "Operations", nextStep: "Require V42 approval and V51 post-deploy sweep." },
    ],
  },
  {
    version: "V53",
    route: "/command-gate-coverage",
    title: "Command Gate Coverage Auditor",
    eyebrow: "V53 gate coverage",
    description: "Audits live routes, CLI actions, schedulers, and deployment handlers to prove production-affecting commands pass through permission gates.",
    sectionTitle: "Command Gate Coverage",
    sectionDescription: "V53 closes the gap between having a permission primitive and knowing every risky command actually uses it.",
    status: "trackable",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 handler classes",
    risk: "high",
    cards: [
      { label: "Handler Classes", value: 5, detail: "deploy, secrets, scheduler, autonomy, spend", tone: "success" },
      { label: "Ungated Action", value: "incident", detail: "missing gate becomes an operations incident", tone: "critical" },
      { label: "Audit Trail", value: "required", detail: "approval and denial records persist", tone: "info" },
    ],
    rows: [
      { id: "gate-inventory", capability: "Production-affecting command inventory", state: "built", owner: "Hermes", nextStep: "List handlers and expected risk class." },
      { id: "coverage-record", capability: "Gate coverage evidence", state: "built", owner: "Operations", nextStep: "Record covered, partial, and missing gate states." },
      { id: "hard-wrap", capability: "All live handlers wrapped", state: "gated", owner: "Project teams", nextStep: "Patch remaining project-specific handlers through V42." },
    ],
  },
  {
    version: "V54",
    route: "/project-adapter-rollout",
    title: "Project Adapter Rollout",
    eyebrow: "V54 adapter adoption",
    description: "Rolls out the V43 telemetry adapter contract across Khashi VC, Media Engine, Media Business Operations, Business Mapper, Meal Assistant, and other Hermes projects.",
    sectionTitle: "Adapter Adoption Matrix",
    sectionDescription: "V54 makes cross-project telemetry real by tracking which production dashboards emit the standard snapshot fields.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "8 dashboard manifests",
    risk: "medium",
    cards: [
      { label: "Manifests", value: 8, detail: "registered Hermes dashboard projects", tone: "success" },
      { label: "Snapshot Fields", value: 7, detail: "health, cost, storage, API, queue, deploy, actions", tone: "info" },
      { label: "Adoption", value: "partial", detail: "project endpoints still need rollout", tone: "warning" },
    ],
    rows: [
      { id: "adapter-manifest-map", capability: "Manifest-to-adapter rollout matrix", state: "built", owner: "Hermes", nextStep: "Compare dashboard manifest URLs to snapshot support." },
      { id: "schema-validation", capability: "Snapshot schema validation evidence", state: "built", owner: "Hermes", nextStep: "Record missing fields without breaking the dashboard." },
      { id: "project-rollout", capability: "Project-owned adapter implementation", state: "gated", owner: "Project teams", nextStep: "Adopt in every production dashboard." },
    ],
  },
  {
    version: "V55",
    route: "/incident-automation",
    title: "Incident Automation Engine",
    eyebrow: "V55 incident engine",
    description: "Creates incident records from failed production sweeps, stale snapshots, missing gates, deployment failures, and circuit-breaker triggers.",
    sectionTitle: "Automated Incident Sources",
    sectionDescription: "V55 connects the operational signals to incident records while keeping remediation approval-based.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 auto sources",
    risk: "high",
    cards: [
      { label: "Sources", value: 5, detail: "sweep, snapshot, gates, deploy, breakers", tone: "success" },
      { label: "Remediation", value: "manual", detail: "auto-create incidents, do not auto-fix", tone: "critical" },
      { label: "Escalation", value: "owner mapped", detail: "every incident needs owner and next step", tone: "info" },
    ],
    rows: [
      { id: "incident-rules", capability: "Automation rules and severity mapping", state: "built", owner: "Operations", nextStep: "Map source failures to severity, owner, and rollback path." },
      { id: "incident-batch", capability: "Batch incident ingestion endpoint", state: "built", owner: "Hermes", nextStep: "Convert sweep and adapter failures into incidents." },
      { id: "auto-remediation", capability: "Approved remediation actions", state: "gated", owner: "Operations", nextStep: "Keep fixes behind V42 command gates." },
    ],
  },
  {
    version: "V56",
    route: "/live-secret-scan",
    title: "Live Secret Presence Scan",
    eyebrow: "V56 secret scanner",
    description: "Scans GitHub Actions variables, GitHub secrets presence, deployment key posture, and Hetzner env requirements without exposing secret values.",
    sectionTitle: "Secret Presence Sweep",
    sectionDescription: "V56 answers whether a deployment can run safely without printing or persisting sensitive values.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "6 secret classes",
    risk: "high",
    cards: [
      { label: "Secret Classes", value: 6, detail: "host, user, key, app dir, URL, app env", tone: "success" },
      { label: "Value Policy", value: "presence only", detail: "never store raw secrets", tone: "critical" },
      { label: "Credential Use", value: "approved", detail: "GitHub/Hetzner checks require admin approval", tone: "warning" },
    ],
    rows: [
      { id: "secret-contract", capability: "Required secret and variable contract", state: "built", owner: "Operations", nextStep: "Declare required names per dashboard." },
      { id: "presence-records", capability: "Presence-only scan records", state: "built", owner: "Hermes", nextStep: "Record present/missing/scope without values." },
      { id: "live-admin-scan", capability: "Live GitHub/Hetzner admin scan", state: "gated", owner: "Operations", nextStep: "Run with approved credentials and audit record." },
    ],
  },
  {
    version: "V57",
    route: "/cost-reconciliation",
    title: "Cost Reconciliation Import",
    eyebrow: "V57 cost reconcile",
    description: "Imports manual rate sheets, invoice totals, API hit counts, storage growth, and hosting costs into the V47 attribution engine.",
    sectionTitle: "Cost Reconciliation Inputs",
    sectionDescription: "V57 separates estimates from actual operating cost and shows where manual rates are still missing.",
    status: "trackable",
    progress: 100,
    owner: "Finance",
    primaryMetric: "5 import sources",
    risk: "medium",
    cards: [
      { label: "Import Sources", value: 5, detail: "rates, invoices, API hits, storage, hosting", tone: "success" },
      { label: "Actuals", value: "manual-ready", detail: "operator can enter monthly rates", tone: "info" },
      { label: "Auto Billing", value: "gated", detail: "provider invoice APIs require credentials", tone: "warning" },
    ],
    rows: [
      { id: "rate-sheet", capability: "Manual rate sheet ingestion", state: "built", owner: "Finance", nextStep: "Record monthly rate and source per cost bucket." },
      { id: "actual-vs-estimate", capability: "Actual versus estimate records", state: "built", owner: "TLC Capital Group OS", nextStep: "Tag records as actual, estimated, or blended." },
      { id: "provider-invoice-api", capability: "Provider invoice API imports", state: "gated", owner: "Finance", nextStep: "Add only after secret scan and cost approval policy exist." },
    ],
  },
  {
    version: "V58",
    route: "/outcome-learning-feeds",
    title: "Outcome Learning Feeds",
    eyebrow: "V58 learning feeds",
    description: "Connects Khashi, Media Engine, deployments, incidents, evals, and dashboard checks into automatic learning events.",
    sectionTitle: "Outcome Feed Sources",
    sectionDescription: "V58 is where Hermes starts learning from what happened instead of relying only on manual summaries.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 outcome feeds",
    risk: "medium",
    cards: [
      { label: "Outcome Feeds", value: 6, detail: "Khashi, Media, deploys, incidents, evals, checks", tone: "success" },
      { label: "Policy Changes", value: "approval", detail: "learning can recommend but not auto-change policy", tone: "critical" },
      { label: "Evidence Links", value: "required", detail: "events link back to source records", tone: "info" },
    ],
    rows: [
      { id: "learning-feed-contract", capability: "Outcome feed contract", state: "built", owner: "Hermes", nextStep: "Normalize source, evidence count, confidence, and recommendation." },
      { id: "batch-learning-ingest", capability: "Batch learning ingestion endpoint", state: "built", owner: "Hermes", nextStep: "Accept project-generated outcomes." },
      { id: "policy-promotion", capability: "Automatic policy promotion", state: "gated", owner: "Operations", nextStep: "Require enough evidence and explicit approval." },
    ],
  },
  {
    version: "V59",
    route: "/golden-eval-execution",
    title: "Golden Eval Execution",
    eyebrow: "V59 eval execution",
    description: "Runs and records golden tasks across model providers so Hermes can compare quality, cost, latency, and production suitability.",
    sectionTitle: "Golden Task Runs",
    sectionDescription: "V59 makes model routing decisions evidence-backed before cheaper providers are trusted with production-grade work.",
    status: "gated",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "5 scoring dimensions",
    risk: "medium",
    cards: [
      { label: "Scores", value: 5, detail: "correctness, tests, design, cost, latency", tone: "success" },
      { label: "Routing", value: "evidence-backed", detail: "recommendation requires scored history", tone: "info" },
      { label: "Provider Spend", value: "breaker-aware", detail: "runs must obey V50 caps", tone: "critical" },
    ],
    rows: [
      { id: "golden-task-batch", capability: "Golden task batch contract", state: "built", owner: "Hermes", nextStep: "Define task family, provider, and expected artifacts." },
      { id: "eval-run-records", capability: "Eval run evidence records", state: "built", owner: "Hermes", nextStep: "Persist score and verdict per provider/task family." },
      { id: "automatic-routing", capability: "Automatic provider routing", state: "gated", owner: "Hermes", nextStep: "Only enable after enough passing eval history." },
    ],
  },
  {
    version: "V60",
    route: "/hard-breaker-enforcement",
    title: "Hard Circuit Breaker Enforcement",
    eyebrow: "V60 hard brakes",
    description: "Applies kill switches, spend caps, provider caps, scheduler stops, and project autonomy limits before any live execution path proceeds.",
    sectionTitle: "Hard Enforcement Checks",
    sectionDescription: "V60 is the final boundary layer: live commands can be blocked even after permission approval if a breaker is active.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 breaker checks",
    risk: "high",
    cards: [
      { label: "Breaker Checks", value: 5, detail: "kill, budget, provider, scheduler, project", tone: "critical" },
      { label: "Default", value: "block on trip", detail: "execution stops when active breaker matches", tone: "critical" },
      { label: "Audit", value: "required", detail: "every block writes evidence", tone: "info" },
    ],
    rows: [
      { id: "breaker-check-endpoint", capability: "Execution-path breaker check", state: "built", owner: "Operations", nextStep: "Call before live deploy, scheduler, provider, and autonomy actions." },
      { id: "breaker-evidence", capability: "Durable block/allow evidence", state: "built", owner: "Hermes", nextStep: "Persist matched breaker and reason." },
      { id: "all-paths-enforced", capability: "Every live execution path enforced", state: "gated", owner: "Project teams", nextStep: "Patch remaining project-specific entry points." },
    ],
  },
  {
    version: "V61",
    route: "/network-runner-adapter",
    title: "Network Runner Adapter",
    eyebrow: "V61 network adapter",
    description: "Defines the approved adapter that can perform DNS, TLS, HTTP, snapshot, and screenshot checks for V51 production sweeps.",
    sectionTitle: "Network Adapter Contract",
    sectionDescription: "V61 turns production checking from evidence-only into an adapter contract that can safely run live network probes when approved.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 probe types",
    risk: "high",
    cards: [
      { label: "Probe Types", value: 5, detail: "DNS, TLS, HTTP, snapshot, screenshot", tone: "success" },
      { label: "Execution", value: "approved", detail: "no network probe runs without gate approval", tone: "critical" },
      { label: "Output", value: "evidence", detail: "normalized results feed V51/V55", tone: "info" },
    ],
    rows: [
      { id: "network-contract", capability: "Network probe adapter contract", state: "built", owner: "Operations", nextStep: "Normalize probe result shape." },
      { id: "network-dry-run", capability: "Dry-run probe plan records", state: "built", owner: "Hermes", nextStep: "Record target list before live network calls." },
      { id: "network-live-runner", capability: "Live network runner implementation", state: "gated", owner: "Operations", nextStep: "Enable only with approval and runtime breaker checks." },
    ],
  },
  {
    version: "V62",
    route: "/hetzner-ssh-adapter",
    title: "Hetzner SSH Adapter",
    eyebrow: "V62 SSH adapter",
    description: "Defines the remote execution adapter for Hetzner promotion commands, docker compose operations, migration commands, and rollback capture.",
    sectionTitle: "Hetzner SSH Execution Contract",
    sectionDescription: "V62 is the adapter between the promotion rail and real server actions, with command previews and approval records.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "6 command classes",
    risk: "high",
    cards: [
      { label: "Command Classes", value: 6, detail: "pull, build, migrate, compose, health, rollback", tone: "critical" },
      { label: "Preview", value: "required", detail: "commands are planned before execution", tone: "success" },
      { label: "SSH", value: "gated", detail: "requires approved server credentials", tone: "warning" },
    ],
    rows: [
      { id: "ssh-command-plan", capability: "Remote command plan records", state: "built", owner: "Operations", nextStep: "Resolve app directory and command sequence." },
      { id: "ssh-audit", capability: "SSH action audit evidence", state: "built", owner: "Hermes", nextStep: "Attach actor, approval, and rollback note." },
      { id: "ssh-live-exec", capability: "Live SSH execution", state: "gated", owner: "Operations", nextStep: "Run only through V42/V60 approved adapter." },
    ],
  },
  {
    version: "V63",
    route: "/secret-provider-adapter",
    title: "Secret Provider Adapter",
    eyebrow: "V63 secret adapter",
    description: "Defines provider adapters for GitHub Actions, deployment variables, deploy keys, and server env presence checks without leaking values.",
    sectionTitle: "Secret Provider Contracts",
    sectionDescription: "V63 separates secret discovery from secret exposure: only names, scope, presence, and rotation posture can be recorded.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "4 providers",
    risk: "high",
    cards: [
      { label: "Providers", value: 4, detail: "GitHub, Hetzner env, deploy keys, webhooks", tone: "success" },
      { label: "Values", value: "never", detail: "no raw values in logs or payloads", tone: "critical" },
      { label: "Rotation", value: "tracked", detail: "age and scope can become findings", tone: "info" },
    ],
    rows: [
      { id: "provider-contract", capability: "Presence-only provider contract", state: "built", owner: "Operations", nextStep: "Normalize provider, scope, and required name." },
      { id: "redaction-policy", capability: "Redaction and no-value policy", state: "built", owner: "Hermes", nextStep: "Keep payloads value-free." },
      { id: "provider-live-scan", capability: "Live provider scans", state: "gated", owner: "Operations", nextStep: "Run with approved credentials and audit only." },
    ],
  },
  {
    version: "V64",
    route: "/billing-provider-adapter",
    title: "Billing Provider Adapter",
    eyebrow: "V64 billing adapter",
    description: "Defines adapters for provider billing, usage counters, model spend, hosting invoices, and storage growth imports.",
    sectionTitle: "Billing Adapter Contract",
    sectionDescription: "V64 makes cost reconciliation less manual without forcing every provider to be integrated at once.",
    status: "trackable",
    progress: 100,
    owner: "Finance",
    primaryMetric: "5 billing feeds",
    risk: "medium",
    cards: [
      { label: "Billing Feeds", value: 5, detail: "OpenAI, other models, hosting, storage, APIs", tone: "success" },
      { label: "Fallback", value: "manual", detail: "rate sheets still work", tone: "info" },
      { label: "Live APIs", value: "gated", detail: "credentials and budget policy required", tone: "warning" },
    ],
    rows: [
      { id: "billing-contract", capability: "Billing provider import contract", state: "built", owner: "Finance", nextStep: "Normalize period, amount, bucket, and source." },
      { id: "manual-fallback", capability: "Manual rate sheet fallback", state: "built", owner: "Finance", nextStep: "Keep actuals available without API integrations." },
      { id: "provider-billing-api", capability: "Live billing provider imports", state: "gated", owner: "Finance", nextStep: "Enable per provider after secret and budget approval." },
    ],
  },
  {
    version: "V65",
    route: "/project-outcome-emitter",
    title: "Project Outcome Emitter",
    eyebrow: "V65 outcome SDK",
    description: "Defines the project-side emitter contract that lets Khashi, Media Engine, and other dashboards send outcomes into Hermes learning feeds.",
    sectionTitle: "Outcome Emitter Contract",
    sectionDescription: "V65 prevents learning from depending on manual copy/paste by giving projects a standard outcome payload.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 event types",
    risk: "medium",
    cards: [
      { label: "Event Types", value: 6, detail: "experiment, generation, deploy, incident, eval, ops", tone: "success" },
      { label: "Source Link", value: "required", detail: "each event points back to evidence", tone: "info" },
      { label: "Adoption", value: "project-owned", detail: "emitters must be added per app", tone: "warning" },
    ],
    rows: [
      { id: "outcome-contract", capability: "Outcome event contract", state: "built", owner: "Hermes", nextStep: "Normalize title, source, evidence count, confidence, and recommendation." },
      { id: "emitter-endpoint", capability: "Project outcome ingest endpoint", state: "built", owner: "Hermes", nextStep: "Accept project outcome events." },
      { id: "project-emitter-rollout", capability: "Emitters adopted by each project", state: "gated", owner: "Project teams", nextStep: "Add emitters to Khashi and Media Engine first." },
    ],
  },
  {
    version: "V66",
    route: "/provider-eval-runner",
    title: "Provider Eval Runner",
    eyebrow: "V66 eval runner",
    description: "Defines the approved runner that can execute golden tasks across local Codex, OpenAI API models, DeepSeek, Kimi, MiniMax, and other providers.",
    sectionTitle: "Provider Eval Runner Contract",
    sectionDescription: "V66 is the bridge from stored eval scores to real provider task execution with spend controls.",
    status: "gated",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 provider classes",
    risk: "medium",
    cards: [
      { label: "Providers", value: 6, detail: "local, OpenAI, DeepSeek, Kimi, MiniMax, fallback", tone: "success" },
      { label: "Spend", value: "breaker-aware", detail: "provider runs check budget caps", tone: "critical" },
      { label: "Artifacts", value: "scored", detail: "outputs become eval evidence", tone: "info" },
    ],
    rows: [
      { id: "eval-runner-contract", capability: "Provider eval runner contract", state: "built", owner: "Hermes", nextStep: "Normalize task, provider, artifact, score, and verdict." },
      { id: "spend-check", capability: "Provider spend breaker check", state: "built", owner: "Operations", nextStep: "Call V60 before paid runs." },
      { id: "live-provider-runs", capability: "Live provider execution", state: "gated", owner: "Hermes", nextStep: "Run only after cost and approval thresholds pass." },
    ],
  },
  {
    version: "V67",
    route: "/breaker-middleware",
    title: "Breaker Middleware SDK",
    eyebrow: "V67 breaker SDK",
    description: "Defines reusable middleware helpers that project routes and schedulers can call before live deploy, scheduler, provider, or autonomy actions.",
    sectionTitle: "Breaker Middleware Contract",
    sectionDescription: "V67 makes V60 enforceable by giving projects a small integration surface instead of bespoke breaker code.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "4 middleware hooks",
    risk: "high",
    cards: [
      { label: "Hooks", value: 4, detail: "deploy, scheduler, provider, autonomy", tone: "success" },
      { label: "Default", value: "block on trip", detail: "matched breakers stop execution", tone: "critical" },
      { label: "Integration", value: "project-owned", detail: "each app must call the middleware", tone: "warning" },
    ],
    rows: [
      { id: "middleware-contract", capability: "Reusable breaker middleware contract", state: "built", owner: "Operations", nextStep: "Normalize project, action, context, and result." },
      { id: "block-evidence", capability: "Block/allow evidence records", state: "built", owner: "Hermes", nextStep: "Persist result per execution attempt." },
      { id: "project-middleware-rollout", capability: "Middleware adopted by live projects", state: "gated", owner: "Project teams", nextStep: "Patch deploy, scheduler, provider, and autonomy entry points." },
    ],
  },
  {
    version: "V68",
    route: "/incident-subscriptions",
    title: "Incident Subscription Bus",
    eyebrow: "V68 incident bus",
    description: "Defines subscriptions that connect production sweeps, telemetry adapters, gate coverage, deployment execution, secret scans, and breakers to incident automation.",
    sectionTitle: "Incident Subscription Rules",
    sectionDescription: "V68 closes the gap between recording failures and routing them into the incident engine automatically.",
    status: "trackable",
    progress: 100,
    owner: "Operations",
    primaryMetric: "6 subscriptions",
    risk: "medium",
    cards: [
      { label: "Subscriptions", value: 6, detail: "sweep, telemetry, gate, deploy, secret, breaker", tone: "success" },
      { label: "Noise Control", value: "dedupe", detail: "same issue should not spam incidents", tone: "info" },
      { label: "Remediation", value: "approval", detail: "subscriptions create incidents, not fixes", tone: "critical" },
    ],
    rows: [
      { id: "subscription-contract", capability: "Incident subscription contract", state: "built", owner: "Operations", nextStep: "Normalize source, severity, dedupe key, and owner." },
      { id: "dedupe-records", capability: "Incident dedupe and suppression records", state: "built", owner: "Hermes", nextStep: "Avoid duplicate issue storms." },
      { id: "live-subscriptions", capability: "Live source subscriptions", state: "gated", owner: "Operations", nextStep: "Subscribe production sweep, telemetry, deploy, and breaker sources." },
    ],
  },
  {
    version: "V69",
    route: "/evidence-artifact-store",
    title: "Evidence Artifact Store",
    eyebrow: "V69 evidence store",
    description: "Defines the artifact index for screenshots, traces, logs, deploy receipts, invoices, eval outputs, and incident attachments.",
    sectionTitle: "Evidence Artifact Index",
    sectionDescription: "V69 keeps proof findable without stuffing large artifacts into the operating-runtime database.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "6 artifact types",
    risk: "medium",
    cards: [
      { label: "Artifact Types", value: 6, detail: "screenshots, logs, traces, receipts, invoices, evals", tone: "success" },
      { label: "Storage", value: "indexed", detail: "DB stores pointers, not blobs", tone: "info" },
      { label: "Retention", value: "policy", detail: "large files need pruning rules", tone: "warning" },
    ],
    rows: [
      { id: "artifact-index", capability: "Artifact pointer records", state: "built", owner: "Hermes", nextStep: "Store path, source, hash, retention, and related evidence id." },
      { id: "retention-policy", capability: "Evidence retention policy", state: "built", owner: "Operations", nextStep: "Set pruning rules per artifact type." },
      { id: "artifact-backend", capability: "Production artifact storage backend", state: "gated", owner: "Operations", nextStep: "Choose local disk, object storage, or project-owned storage." },
    ],
  },
  {
    version: "V70",
    route: "/release-train-orchestrator",
    title: "Release Train Orchestrator",
    eyebrow: "V70 release train",
    description: "Coordinates approvals, promotion execution, production sweeps, incident subscriptions, artifacts, rollback, and executive summaries into one release train.",
    sectionTitle: "Release Train Gates",
    sectionDescription: "V70 is the operating posture for safely pushing multiple dashboards through the same governed production rail.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "7 train gates",
    risk: "high",
    cards: [
      { label: "Train Gates", value: 7, detail: "approve, build, migrate, deploy, sweep, artifact, summary", tone: "critical" },
      { label: "Rollback", value: "planned", detail: "each release needs rollback note", tone: "info" },
      { label: "Autonomy", value: "manual-first", detail: "automatic trains remain gated", tone: "warning" },
    ],
    rows: [
      { id: "release-train-contract", capability: "Release train orchestration contract", state: "built", owner: "Operations", nextStep: "Normalize projects, versions, gates, evidence, and rollback." },
      { id: "train-summary", capability: "Executive release summary", state: "built", owner: "TLC Capital Group OS", nextStep: "Summarize what shipped, what failed, and what needs approval." },
      { id: "automatic-release-train", capability: "Automatic multi-project release trains", state: "gated", owner: "Operations", nextStep: "Keep manual until adapters prove reliable across projects." },
    ],
  },
  {
    version: "V71",
    route: "/production-screenshot-runner",
    title: "Production Screenshot Runner",
    eyebrow: "V71 screenshot runner",
    description: "Adds browser-based production screenshot checks to approved sweeps so Hermes can prove a route renders, not just that HTTP responds.",
    sectionTitle: "Screenshot Capture Gates",
    sectionDescription: "V71 turns screenshot evidence into a first-class production sweep artifact with viewport, route, hash, and blank-page failure handling.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "4 viewports",
    risk: "high",
    cards: [
      { label: "Viewports", value: 4, detail: "desktop, laptop, tablet, mobile", tone: "success" },
      { label: "Execution", value: "approved", detail: "runs after sweep permission and breaker checks", tone: "critical" },
      { label: "Artifacts", value: "linked", detail: "screenshots stored by pointer and hash", tone: "info" },
    ],
    rows: [
      { id: "screenshot-runner", capability: "Production browser screenshot runner", state: "built", owner: "Operations", nextStep: "Capture approved route screenshots." },
      { id: "blank-detection", capability: "Blank or route-mismatch detection", state: "built", owner: "Hermes", nextStep: "Fail sweeps when rendered evidence is invalid." },
      { id: "screenshot-storage", capability: "Screenshot artifact persistence", state: "gated", owner: "Operations", nextStep: "Attach durable artifact backend in V75." },
    ],
  },
  {
    version: "V72",
    route: "/hetzner-promotion-transport",
    title: "Hetzner Promotion Transport",
    eyebrow: "V72 SSH transport",
    description: "Connects the promotion rail to the shared Hetzner promotion script with approval, breaker checks, redacted command receipts, and rollback evidence.",
    sectionTitle: "Remote Promotion Execution",
    sectionDescription: "V72 is the live transport bridge: it does not replace gates, it executes only after they pass.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 execution receipts",
    risk: "high",
    cards: [
      { label: "Receipts", value: 5, detail: "plan, approval, command, health, rollback", tone: "success" },
      { label: "Transport", value: "SSH", detail: "shared Hetzner promotion script", tone: "warning" },
      { label: "Rollback", value: "required", detail: "no promotion without rollback note", tone: "critical" },
    ],
    rows: [
      { id: "ssh-transport", capability: "Shared Hetzner SSH transport wrapper", state: "built", owner: "Operations", nextStep: "Run approved promotion script over SSH." },
      { id: "command-receipts", capability: "Redacted command receipt artifacts", state: "built", owner: "Hermes", nextStep: "Attach stdout/stderr pointers without secrets." },
      { id: "server-credentials", capability: "Configured production SSH credentials", state: "gated", owner: "Operator", nextStep: "Verify GitHub Actions and production server secrets." },
    ],
  },
  {
    version: "V73",
    route: "/server-secret-posture-scanner",
    title: "Server Secret Posture Scanner",
    eyebrow: "V73 server secrets",
    description: "Verifies required production environment variable names on the server without exposing values.",
    sectionTitle: "Server Secret Presence",
    sectionDescription: "V73 checks whether production is configured enough to run safely while preserving the no-secret-values rule.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "0 values exposed",
    risk: "high",
    cards: [
      { label: "Values", value: "never", detail: "names and presence only", tone: "critical" },
      { label: "Scopes", value: 3, detail: "GitHub, server env, provider refs", tone: "success" },
      { label: "Incidents", value: "auto", detail: "missing required names create incidents", tone: "warning" },
    ],
    rows: [
      { id: "server-secret-manifest", capability: "Per-project required secret manifest", state: "built", owner: "Operations", nextStep: "Declare required env names per service." },
      { id: "server-secret-scan", capability: "Hetzner env-name scan plan", state: "built", owner: "Operations", nextStep: "Run approved presence checks without values." },
      { id: "rotation-metadata", capability: "Rotation age metadata", state: "gated", owner: "Operations", nextStep: "Add when a managed secret backend exists." },
    ],
  },
  {
    version: "V74",
    route: "/incident-notification-fanout",
    title: "Incident Notification Fanout",
    eyebrow: "V74 notification fanout",
    description: "Routes incident records to human channels by severity with dedupe, cooldown, acknowledgement, and resolution state.",
    sectionTitle: "Incident Fanout Rules",
    sectionDescription: "V74 makes failures visible without turning one broken service into a message storm.",
    status: "trackable",
    progress: 100,
    owner: "Operations",
    primaryMetric: "3 channels",
    risk: "medium",
    cards: [
      { label: "Channels", value: 3, detail: "Discord, Telegram, email-ready", tone: "success" },
      { label: "Noise", value: "deduped", detail: "cooldowns prevent repeat spam", tone: "info" },
      { label: "Critical", value: "immediate", detail: "critical routes notify now", tone: "critical" },
    ],
    rows: [
      { id: "fanout-targets", capability: "Incident fanout target records", state: "built", owner: "Operations", nextStep: "Record channel, severity, and cooldown." },
      { id: "notification-dispatch", capability: "Notification dispatch contract", state: "built", owner: "Hermes", nextStep: "Send enabled incidents to configured channels." },
      { id: "ack-resolution", capability: "Acknowledgement and resolution workflow", state: "gated", owner: "Operations", nextStep: "Add operator action buttons after channel auth is verified." },
    ],
  },
  {
    version: "V75",
    route: "/durable-artifact-backend",
    title: "Durable Artifact Backend",
    eyebrow: "V75 artifact backend",
    description: "Adds a concrete artifact backend for screenshots, logs, traces, receipts, invoices, eval outputs, and generated evidence.",
    sectionTitle: "Artifact Backend Policy",
    sectionDescription: "V75 converts artifact pointers into a storage posture with retention, checksums, and cleanup.",
    status: "trackable",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "7 artifact classes",
    risk: "medium",
    cards: [
      { label: "Classes", value: 7, detail: "screenshots, logs, traces, receipts, invoices, evals, media", tone: "success" },
      { label: "Storage", value: "pluggable", detail: "local volume now, object store later", tone: "info" },
      { label: "Retention", value: "required", detail: "cleanup prevents unbounded growth", tone: "warning" },
    ],
    rows: [
      { id: "artifact-backend-policy", capability: "Artifact backend selection records", state: "built", owner: "Hermes", nextStep: "Choose local, object-store, or hybrid." },
      { id: "artifact-write-adapter", capability: "Artifact write adapter contract", state: "built", owner: "Hermes", nextStep: "Normalize write result and pointer evidence." },
      { id: "artifact-cleanup", capability: "Retention cleanup job", state: "gated", owner: "Operations", nextStep: "Enable cleanup after backend path is configured." },
    ],
  },
  {
    version: "V76",
    route: "/remaining-project-outcome-adapters",
    title: "Remaining Project Outcome Adapters",
    eyebrow: "V76 adapter rollout",
    description: "Rolls the `/api/hermes/outcomes` contract beyond Khashi VC and Media Engine into every production dashboard.",
    sectionTitle: "Outcome Adapter Adoption",
    sectionDescription: "V76 makes Hermes learning work across the whole portfolio instead of only the first two production projects.",
    status: "trackable",
    progress: 100,
    owner: "Project teams",
    primaryMetric: "6 projects",
    risk: "medium",
    cards: [
      { label: "Projects", value: 6, detail: "Hermes OS, Mapper, MBO, Investing, Meal, future", tone: "success" },
      { label: "Contract", value: "/api/hermes/outcomes", detail: "standard outcome feed", tone: "info" },
      { label: "Validation", value: "registry", detail: "missing adapters are visible", tone: "warning" },
    ],
    rows: [
      { id: "adapter-template", capability: "Reusable project outcome adapter template", state: "built", owner: "Hermes", nextStep: "Use same payload contract everywhere." },
      { id: "adapter-validation", capability: "Registry validation for missing adapters", state: "built", owner: "Hermes", nextStep: "Flag dashboards without outcome endpoints." },
      { id: "remaining-rollout", capability: "Remaining project implementation", state: "gated", owner: "Project teams", nextStep: "Patch each project dashboard server." },
    ],
  },
  {
    version: "V77",
    route: "/breaker-middleware-rollout",
    title: "Breaker Middleware Rollout",
    eyebrow: "V77 breaker rollout",
    description: "Applies breaker middleware to scheduler, provider, deploy, and autopilot paths before autonomy increases.",
    sectionTitle: "Live Path Enforcement",
    sectionDescription: "V77 closes the gap between having breakers and actually using them in project-owned live paths.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "5 live path classes",
    risk: "high",
    cards: [
      { label: "Path Classes", value: 5, detail: "scheduler, sync, provider, autopilot, deploy", tone: "success" },
      { label: "Default", value: "block", detail: "breaker match stops execution", tone: "critical" },
      { label: "Tests", value: "required", detail: "each path proves block behavior", tone: "warning" },
    ],
    rows: [
      { id: "scheduler-breaker", capability: "Khashi scheduler and sync breaker wrapper", state: "built", owner: "Khashi VC", nextStep: "Block capacity changes when breakers trip." },
      { id: "media-provider-breaker", capability: "Media Engine autopilot/provider breaker wrapper", state: "built", owner: "Media Engine", nextStep: "Block paid generation when breakers trip." },
      { id: "remaining-breakers", capability: "Remaining project deploy/provider wrappers", state: "gated", owner: "Project teams", nextStep: "Patch project-owned live paths." },
    ],
  },
  {
    version: "V78",
    route: "/provider-eval-execution",
    title: "Provider Eval Execution",
    eyebrow: "V78 paid eval execution",
    description: "Runs approved golden tasks through provider adapters with budget breakers, scored artifacts, and routing recommendations.",
    sectionTitle: "Provider Eval Execution",
    sectionDescription: "V78 lets Hermes compare model providers using real outcomes without letting provider spend run loose.",
    status: "gated",
    progress: 100,
    owner: "Hermes",
    primaryMetric: "10 golden tasks",
    risk: "medium",
    cards: [
      { label: "Golden Tasks", value: 10, detail: "coding, design, research, ops", tone: "success" },
      { label: "Spend", value: "capped", detail: "budget breaker before each run", tone: "critical" },
      { label: "Routing", value: "earned", detail: "recommendations require scored history", tone: "info" },
    ],
    rows: [
      { id: "golden-task-catalog", capability: "Golden task catalog", state: "built", owner: "Hermes", nextStep: "Define representative paid-provider tasks." },
      { id: "provider-runner", capability: "Approved provider execution wrapper", state: "built", owner: "Hermes", nextStep: "Run only with spend and approval checks." },
      { id: "routing-promotion", capability: "Automatic provider-routing promotion", state: "gated", owner: "Hermes", nextStep: "Wait for enough scored runs." },
    ],
  },
  {
    version: "V79",
    route: "/billing-provider-integrations",
    title: "Billing Provider Integrations",
    eyebrow: "V79 billing APIs",
    description: "Imports provider billing and usage exports to reconcile real spend against manual invoice totals.",
    sectionTitle: "Billing Integration Sources",
    sectionDescription: "V79 keeps manual billing as the baseline while provider APIs fill in detail where available.",
    status: "trackable",
    progress: 100,
    owner: "Finance",
    primaryMetric: "4 provider feeds",
    risk: "medium",
    cards: [
      { label: "Feeds", value: 4, detail: "OpenAI, Firecrawl, hosting, storage", tone: "success" },
      { label: "Baseline", value: "manual", detail: "manual invoice import remains source of truth", tone: "info" },
      { label: "Variance", value: "reported", detail: "provider totals compared to invoices", tone: "warning" },
    ],
    rows: [
      { id: "billing-openai", capability: "OpenAI usage import contract", state: "built", owner: "Finance", nextStep: "Import usage when billing API access is available." },
      { id: "billing-hosting", capability: "Hosting and storage cost import contract", state: "built", owner: "Finance", nextStep: "Attach Hetzner/storage actuals." },
      { id: "billing-direct-apis", capability: "Provider-native billing APIs", state: "gated", owner: "Finance", nextStep: "Enable one provider at a time after credential approval." },
    ],
  },
  {
    version: "V80",
    route: "/release-train-execution",
    title: "Release Train Execution",
    eyebrow: "V80 train execution",
    description: "Executes multi-project release trains only after sweeps, secrets, artifacts, incidents, rollback, and breakers pass.",
    sectionTitle: "Release Train Execution Gates",
    sectionDescription: "V80 is the first safe posture for multi-project production promotion.",
    status: "gated",
    progress: 100,
    owner: "Operations",
    primaryMetric: "7 hard gates",
    risk: "high",
    cards: [
      { label: "Hard Gates", value: 7, detail: "sweep, secrets, breaker, rollback, artifact, incident, summary", tone: "critical" },
      { label: "Order", value: "controlled", detail: "projects promote one at a time", tone: "info" },
      { label: "Failure", value: "stop train", detail: "first critical failure halts execution", tone: "warning" },
    ],
    rows: [
      { id: "train-gate-checks", capability: "Preflight release train gate checks", state: "built", owner: "Operations", nextStep: "Require every hard gate before live train." },
      { id: "train-execution", capability: "Controlled release train execution", state: "built", owner: "Operations", nextStep: "Promote projects in approved order." },
      { id: "train-autonomy", capability: "Autonomous release trains", state: "gated", owner: "Operations", nextStep: "Keep human approval until enough successful trains exist." },
    ],
  },
];
