import { lazy, type ComponentType, type LazyExoticComponent } from "react";
import { Navigate } from "react-router-dom";
import {
  Activity,
  BarChart3,
  BookOpen,
  BriefcaseBusiness,
  Building2,
  Code2,
  Clock,
  Database,
  Cpu,
  FileText,
  FolderOpen,
  GalleryVerticalEnd,
  GitBranch,
  Globe,
  KeyRound,
  ListChecks,
  MessageSquare,
  Package,
  Palette,
  Plug,
  Puzzle,
  Radio,
  RotateCw,
  Scale,
  Settings,
  ShieldCheck,
  Siren,
  Sparkles,
  Terminal,
  Users,
  Webhook,
  Wrench,
  type LucideIcon,
} from "lucide-react";

const ConfigPage = lazy(() => import("@/pages/ConfigPage"));
const DocsPage = lazy(() => import("@/pages/DocsPage"));
const EnvPage = lazy(() => import("@/pages/EnvPage"));
const FilesPage = lazy(() => import("@/pages/FilesPage"));
const SessionsPage = lazy(() => import("@/pages/SessionsPage"));
const LogsPage = lazy(() => import("@/pages/LogsPage"));
const AnalyticsPage = lazy(() => import("@/pages/AnalyticsPage"));
const ModelsPage = lazy(() => import("@/pages/ModelsPage"));
const CronPage = lazy(() => import("@/pages/CronPage"));
const HermesOsPage = lazy(() => import("@/pages/HermesOsPage"));
const DesignSystemPage = lazy(() => import("@/pages/DesignSystemPage"));
const PackageNativeMigrationsPage = lazy(() => import("@/pages/PackageNativeMigrationsPage"));
const ExecutiveSummaryPage = lazy(() => import("@/pages/ExecutiveSummaryPage"));
const CentralCommandPage = lazy(() => import("@/pages/CentralCommandPage"));
const ThemeSystemPage = lazy(() => import("@/pages/ThemeSystemPage"));
const DashboardMarketplacePage = lazy(() => import("@/pages/DashboardMarketplacePage"));
const LiveSignalsPage = lazy(() => import("@/pages/LiveSignalsPage"));
const TaskRoutingPage = lazy(() => import("@/pages/TaskRoutingPage"));
const DecisionLedgerPage = lazy(() => import("@/pages/DecisionLedgerPage"));
const ModelRoutingPage = lazy(() => import("@/pages/ModelRoutingPage"));
const OperatingLoopsPage = lazy(() => import("@/pages/OperatingLoopsPage"));
const PermissionSecurityPage = lazy(() => import("@/pages/PermissionSecurityPage"));
const BusinessOSPage = lazy(() => import("@/pages/BusinessOSPage"));
const ProjectSnapshotsPage = lazy(() => import("@/pages/ProjectSnapshotsPage"));
const DurableMemoryPage = lazy(() => import("@/pages/DurableMemoryPage"));
const PermissionRuntimePage = lazy(() => import("@/pages/PermissionRuntimePage"));
const CostGovernorPage = lazy(() => import("@/pages/CostGovernorPage"));
const LoopRunnerPage = lazy(() => import("@/pages/LoopRunnerPage"));
const BusinessCommandPage = lazy(() => import("@/pages/BusinessCommandPage"));
const AgentWorkbenchPage = lazy(() => import("@/pages/AgentWorkbenchPage"));
const EvaluationGatesPage = lazy(() => import("@/pages/EvaluationGatesPage"));
const AutonomyReadinessPage = lazy(() => import("@/pages/AutonomyReadinessPage"));
const ProjectRegistryPage = lazy(() => import("@/pages/ProjectRegistryPage"));
const TelemetryFabricPage = lazy(() => import("@/pages/TelemetryFabricPage"));
const IncidentCommandPage = lazy(() => import("@/pages/IncidentCommandPage"));
const DeploymentPromotionPage = lazy(() => import("@/pages/DeploymentPromotionPage"));
const SecretsPosturePage = lazy(() => import("@/pages/SecretsPosturePage"));
const DataSourceCatalogPage = lazy(() => import("@/pages/DataSourceCatalogPage"));
const FinanceAttributionPage = lazy(() => import("@/pages/FinanceAttributionPage"));
const LearningEnginePage = lazy(() => import("@/pages/LearningEnginePage"));
const AgentEvalLabPage = lazy(() => import("@/pages/AgentEvalLabPage"));
const ExecutiveCockpitPage = lazy(() => import("@/pages/ExecutiveCockpitPage"));
const ProductionVerificationPage = lazy(() => import("@/pages/ProductionVerificationPage"));
const CommandGateRuntimePage = lazy(() => import("@/pages/CommandGateRuntimePage"));
const TelemetryAdapterKitPage = lazy(() => import("@/pages/TelemetryAdapterKitPage"));
const IncidentIngestionPage = lazy(() => import("@/pages/IncidentIngestionPage"));
const PromotionRunnerPage = lazy(() => import("@/pages/PromotionRunnerPage"));
const SecretScannerPage = lazy(() => import("@/pages/SecretScannerPage"));
const CostAttributionEnginePage = lazy(() => import("@/pages/CostAttributionEnginePage"));
const LearningIngestionPage = lazy(() => import("@/pages/LearningIngestionPage"));
const ModelEvalHarnessPage = lazy(() => import("@/pages/ModelEvalHarnessPage"));
const CircuitBreakersPage = lazy(() => import("@/pages/CircuitBreakersPage"));
const ProductionSweepPage = lazy(() => import("@/pages/ProductionSweepPage"));
const HetznerPromotionExecutionPage = lazy(() => import("@/pages/HetznerPromotionExecutionPage"));
const CommandGateCoveragePage = lazy(() => import("@/pages/CommandGateCoveragePage"));
const ProjectAdapterRolloutPage = lazy(() => import("@/pages/ProjectAdapterRolloutPage"));
const IncidentAutomationPage = lazy(() => import("@/pages/IncidentAutomationPage"));
const LiveSecretScanPage = lazy(() => import("@/pages/LiveSecretScanPage"));
const CostReconciliationPage = lazy(() => import("@/pages/CostReconciliationPage"));
const OutcomeLearningFeedsPage = lazy(() => import("@/pages/OutcomeLearningFeedsPage"));
const GoldenEvalExecutionPage = lazy(() => import("@/pages/GoldenEvalExecutionPage"));
const HardBreakerEnforcementPage = lazy(() => import("@/pages/HardBreakerEnforcementPage"));
const NetworkRunnerAdapterPage = lazy(() => import("@/pages/NetworkRunnerAdapterPage"));
const HetznerSshAdapterPage = lazy(() => import("@/pages/HetznerSshAdapterPage"));
const SecretProviderAdapterPage = lazy(() => import("@/pages/SecretProviderAdapterPage"));
const BillingProviderAdapterPage = lazy(() => import("@/pages/BillingProviderAdapterPage"));
const ProjectOutcomeEmitterPage = lazy(() => import("@/pages/ProjectOutcomeEmitterPage"));
const ProviderEvalRunnerPage = lazy(() => import("@/pages/ProviderEvalRunnerPage"));
const BreakerMiddlewarePage = lazy(() => import("@/pages/BreakerMiddlewarePage"));
const IncidentSubscriptionPage = lazy(() => import("@/pages/IncidentSubscriptionPage"));
const EvidenceArtifactStorePage = lazy(() => import("@/pages/EvidenceArtifactStorePage"));
const ReleaseTrainOrchestratorPage = lazy(() => import("@/pages/ReleaseTrainOrchestratorPage"));
const ProductionScreenshotRunnerPage = lazy(() => import("@/pages/ProductionScreenshotRunnerPage"));
const HetznerPromotionTransportPage = lazy(() => import("@/pages/HetznerPromotionTransportPage"));
const ServerSecretPostureScannerPage = lazy(() => import("@/pages/ServerSecretPostureScannerPage"));
const IncidentNotificationFanoutPage = lazy(() => import("@/pages/IncidentNotificationFanoutPage"));
const DurableArtifactBackendPage = lazy(() => import("@/pages/DurableArtifactBackendPage"));
const RemainingProjectOutcomeAdaptersPage = lazy(() => import("@/pages/RemainingProjectOutcomeAdaptersPage"));
const BreakerMiddlewareRolloutPage = lazy(() => import("@/pages/BreakerMiddlewareRolloutPage"));
const ProviderEvalExecutionPage = lazy(() => import("@/pages/ProviderEvalExecutionPage"));
const BillingProviderIntegrationsPage = lazy(() => import("@/pages/BillingProviderIntegrationsPage"));
const ReleaseTrainExecutionPage = lazy(() => import("@/pages/ReleaseTrainExecutionPage"));
const ProfilesPage = lazy(() => import("@/pages/ProfilesPage"));
const ProfileBuilderPage = lazy(() => import("@/pages/ProfileBuilderPage"));
const SkillsPage = lazy(() => import("@/pages/SkillsPage"));
const PluginsPage = lazy(() => import("@/pages/PluginsPage"));
const McpPage = lazy(() => import("@/pages/McpPage"));
const PairingPage = lazy(() => import("@/pages/PairingPage"));
const ChannelsPage = lazy(() => import("@/pages/ChannelsPage"));
const WebhooksPage = lazy(() => import("@/pages/WebhooksPage"));
const SystemPage = lazy(() => import("@/pages/SystemPage"));

export type RouteComponent = ComponentType | LazyExoticComponent<ComponentType>;

export interface BuiltinNavItem {
  path: string;
  label: string;
  icon: LucideIcon;
  labelKey?: string;
}

function RootRedirect() {
  return <Navigate to="/sessions" replace />;
}

export const CHAT_NAV_ITEM: BuiltinNavItem = {
  path: "/chat",
  labelKey: "chat",
  label: "Chat",
  icon: Terminal,
};

export const BUILTIN_ROUTES_CORE: Record<string, RouteComponent> = {
  "/": RootRedirect,
  "/sessions": SessionsPage,
  "/files": FilesPage,
  "/analytics": AnalyticsPage,
  "/hermes-os": HermesOsPage,
  "/central-command": CentralCommandPage,
  "/executive-summary": ExecutiveSummaryPage,
  "/dashboard-migrations": PackageNativeMigrationsPage,
  "/theme-system": ThemeSystemPage,
  "/dashboard-marketplace": DashboardMarketplacePage,
  "/live-signals": LiveSignalsPage,
  "/task-routing": TaskRoutingPage,
  "/decision-ledger": DecisionLedgerPage,
  "/model-routing": ModelRoutingPage,
  "/operating-loops": OperatingLoopsPage,
  "/permission-security": PermissionSecurityPage,
  "/business-os": BusinessOSPage,
  "/project-snapshots": ProjectSnapshotsPage,
  "/durable-memory": DurableMemoryPage,
  "/permission-runtime": PermissionRuntimePage,
  "/cost-governor": CostGovernorPage,
  "/loop-runner": LoopRunnerPage,
  "/business-command": BusinessCommandPage,
  "/agent-workbench": AgentWorkbenchPage,
  "/evaluation-gates": EvaluationGatesPage,
  "/autonomy-readiness": AutonomyReadinessPage,
  "/project-registry": ProjectRegistryPage,
  "/telemetry-fabric": TelemetryFabricPage,
  "/incident-command": IncidentCommandPage,
  "/deployment-promotion": DeploymentPromotionPage,
  "/secrets-posture": SecretsPosturePage,
  "/data-source-catalog": DataSourceCatalogPage,
  "/finance-attribution": FinanceAttributionPage,
  "/learning-engine": LearningEnginePage,
  "/agent-eval-lab": AgentEvalLabPage,
  "/executive-cockpit": ExecutiveCockpitPage,
  "/production-verification": ProductionVerificationPage,
  "/command-gates": CommandGateRuntimePage,
  "/telemetry-adapters": TelemetryAdapterKitPage,
  "/incident-ingestion": IncidentIngestionPage,
  "/promotion-runner": PromotionRunnerPage,
  "/secret-scanner": SecretScannerPage,
  "/cost-attribution-engine": CostAttributionEnginePage,
  "/learning-ingestion": LearningIngestionPage,
  "/model-eval-harness": ModelEvalHarnessPage,
  "/circuit-breakers": CircuitBreakersPage,
  "/production-sweep": ProductionSweepPage,
  "/hetzner-promotion-execution": HetznerPromotionExecutionPage,
  "/command-gate-coverage": CommandGateCoveragePage,
  "/project-adapter-rollout": ProjectAdapterRolloutPage,
  "/incident-automation": IncidentAutomationPage,
  "/live-secret-scan": LiveSecretScanPage,
  "/cost-reconciliation": CostReconciliationPage,
  "/outcome-learning-feeds": OutcomeLearningFeedsPage,
  "/golden-eval-execution": GoldenEvalExecutionPage,
  "/hard-breaker-enforcement": HardBreakerEnforcementPage,
  "/network-runner-adapter": NetworkRunnerAdapterPage,
  "/hetzner-ssh-adapter": HetznerSshAdapterPage,
  "/secret-provider-adapter": SecretProviderAdapterPage,
  "/billing-provider-adapter": BillingProviderAdapterPage,
  "/project-outcome-emitter": ProjectOutcomeEmitterPage,
  "/provider-eval-runner": ProviderEvalRunnerPage,
  "/breaker-middleware": BreakerMiddlewarePage,
  "/incident-subscriptions": IncidentSubscriptionPage,
  "/evidence-artifact-store": EvidenceArtifactStorePage,
  "/release-train-orchestrator": ReleaseTrainOrchestratorPage,
  "/production-screenshot-runner": ProductionScreenshotRunnerPage,
  "/hetzner-promotion-transport": HetznerPromotionTransportPage,
  "/server-secret-posture-scanner": ServerSecretPostureScannerPage,
  "/incident-notification-fanout": IncidentNotificationFanoutPage,
  "/durable-artifact-backend": DurableArtifactBackendPage,
  "/remaining-project-outcome-adapters": RemainingProjectOutcomeAdaptersPage,
  "/breaker-middleware-rollout": BreakerMiddlewareRolloutPage,
  "/provider-eval-execution": ProviderEvalExecutionPage,
  "/billing-provider-integrations": BillingProviderIntegrationsPage,
  "/release-train-execution": ReleaseTrainExecutionPage,
  "/design-system": DesignSystemPage,
  "/models": ModelsPage,
  "/logs": LogsPage,
  "/cron": CronPage,
  "/skills": SkillsPage,
  "/plugins": PluginsPage,
  "/mcp": McpPage,
  "/pairing": PairingPage,
  "/channels": ChannelsPage,
  "/webhooks": WebhooksPage,
  "/system": SystemPage,
  "/profiles": ProfilesPage,
  "/profiles/new": ProfileBuilderPage,
  "/config": ConfigPage,
  "/env": EnvPage,
  "/docs": DocsPage,
};

export const BUILTIN_NAV_REST: BuiltinNavItem[] = [
  { path: "/sessions", labelKey: "sessions", label: "Sessions", icon: MessageSquare },
  { path: "/files", label: "Files", icon: FolderOpen },
  { path: "/analytics", labelKey: "analytics", label: "Analytics", icon: BarChart3 },
  { path: "/hermes-os", label: "Hermes OS", icon: GitBranch },
  { path: "/central-command", label: "Central Command", icon: Building2 },
  { path: "/executive-summary", label: "Executive", icon: Building2 },
  { path: "/dashboard-migrations", label: "Dashboard Migrations", icon: Code2 },
  { path: "/theme-system", label: "Theme System", icon: Palette },
  { path: "/dashboard-marketplace", label: "Marketplace", icon: Plug },
  { path: "/live-signals", label: "Live Signals", icon: Radio },
  { path: "/task-routing", label: "Task Routing", icon: ListChecks },
  { path: "/decision-ledger", label: "Decision Ledger", icon: BookOpen },
  { path: "/model-routing", label: "Model Routing", icon: Cpu },
  { path: "/operating-loops", label: "Operating Loops", icon: RotateCw },
  { path: "/permission-security", label: "Permissions", icon: ShieldCheck },
  { path: "/business-os", label: "Business OS", icon: BriefcaseBusiness },
  { path: "/project-snapshots", label: "Snapshots", icon: Radio },
  { path: "/durable-memory", label: "Memory Store", icon: Database },
  { path: "/permission-runtime", label: "Permission Runtime", icon: ShieldCheck },
  { path: "/cost-governor", label: "Cost Governor", icon: Scale },
  { path: "/loop-runner", label: "Loop Runner", icon: RotateCw },
  { path: "/business-command", label: "Business Command", icon: BriefcaseBusiness },
  { path: "/agent-workbench", label: "Agent Workbench", icon: ListChecks },
  { path: "/evaluation-gates", label: "Evaluation Gates", icon: BookOpen },
  { path: "/autonomy-readiness", label: "Autonomy", icon: ShieldCheck },
  { path: "/project-registry", label: "Project Registry", icon: Building2 },
  { path: "/telemetry-fabric", label: "Telemetry Fabric", icon: Activity },
  { path: "/incident-command", label: "Incident Command", icon: Siren },
  { path: "/deployment-promotion", label: "Promotion", icon: GitBranch },
  { path: "/secrets-posture", label: "Secrets Posture", icon: KeyRound },
  { path: "/data-source-catalog", label: "Data Sources", icon: Database },
  { path: "/finance-attribution", label: "Attribution", icon: Scale },
  { path: "/learning-engine", label: "Learning Engine", icon: Sparkles },
  { path: "/agent-eval-lab", label: "Agent Eval Lab", icon: Code2 },
  { path: "/executive-cockpit", label: "Executive Cockpit", icon: Building2 },
  { path: "/production-verification", label: "Prod Verify", icon: Globe },
  { path: "/command-gates", label: "Command Gates", icon: ShieldCheck },
  { path: "/telemetry-adapters", label: "Telemetry Adapters", icon: Radio },
  { path: "/incident-ingestion", label: "Incident Ingest", icon: Siren },
  { path: "/promotion-runner", label: "Promotion Runner", icon: GitBranch },
  { path: "/secret-scanner", label: "Secret Scanner", icon: KeyRound },
  { path: "/cost-attribution-engine", label: "Cost Engine", icon: Scale },
  { path: "/learning-ingestion", label: "Learning Ingest", icon: Sparkles },
  { path: "/model-eval-harness", label: "Eval Harness", icon: Code2 },
  { path: "/circuit-breakers", label: "Circuit Breakers", icon: ShieldCheck },
  { path: "/production-sweep", label: "Prod Sweep", icon: Globe },
  { path: "/hetzner-promotion-execution", label: "Hetzner Promote", icon: GitBranch },
  { path: "/command-gate-coverage", label: "Gate Coverage", icon: ShieldCheck },
  { path: "/project-adapter-rollout", label: "Adapter Rollout", icon: Radio },
  { path: "/incident-automation", label: "Incident Auto", icon: Siren },
  { path: "/live-secret-scan", label: "Live Secret Scan", icon: KeyRound },
  { path: "/cost-reconciliation", label: "Cost Reconcile", icon: Scale },
  { path: "/outcome-learning-feeds", label: "Outcome Feeds", icon: Sparkles },
  { path: "/golden-eval-execution", label: "Eval Execution", icon: Code2 },
  { path: "/hard-breaker-enforcement", label: "Hard Breakers", icon: ShieldCheck },
  { path: "/network-runner-adapter", label: "Network Runner", icon: Globe },
  { path: "/hetzner-ssh-adapter", label: "Hetzner SSH", icon: Terminal },
  { path: "/secret-provider-adapter", label: "Secret Adapter", icon: KeyRound },
  { path: "/billing-provider-adapter", label: "Billing Adapter", icon: Scale },
  { path: "/project-outcome-emitter", label: "Outcome Emitter", icon: Radio },
  { path: "/provider-eval-runner", label: "Provider Runner", icon: Code2 },
  { path: "/breaker-middleware", label: "Breaker Middleware", icon: ShieldCheck },
  { path: "/incident-subscriptions", label: "Incident Subs", icon: Siren },
  { path: "/evidence-artifact-store", label: "Evidence Store", icon: Database },
  { path: "/release-train-orchestrator", label: "Release Train", icon: GitBranch },
  { path: "/production-screenshot-runner", label: "Screenshots", icon: GalleryVerticalEnd },
  { path: "/hetzner-promotion-transport", label: "SSH Promote", icon: Terminal },
  { path: "/server-secret-posture-scanner", label: "Server Secrets", icon: KeyRound },
  { path: "/incident-notification-fanout", label: "Incident Fanout", icon: Siren },
  { path: "/durable-artifact-backend", label: "Artifact Backend", icon: Database },
  { path: "/remaining-project-outcome-adapters", label: "Outcome Adapters", icon: Radio },
  { path: "/breaker-middleware-rollout", label: "Breaker Rollout", icon: ShieldCheck },
  { path: "/provider-eval-execution", label: "Provider Evals", icon: Code2 },
  { path: "/billing-provider-integrations", label: "Billing APIs", icon: Scale },
  { path: "/release-train-execution", label: "Train Execute", icon: GitBranch },
  { path: "/design-system", label: "Design System", icon: GalleryVerticalEnd },
  { path: "/models", labelKey: "models", label: "Models", icon: Cpu },
  { path: "/logs", labelKey: "logs", label: "Logs", icon: FileText },
  { path: "/cron", labelKey: "cron", label: "Cron", icon: Clock },
  { path: "/skills", labelKey: "skills", label: "Skills", icon: Package },
  { path: "/plugins", labelKey: "plugins", label: "Plugins", icon: Puzzle },
  { path: "/mcp", label: "MCP", icon: Plug },
  { path: "/channels", label: "Channels", icon: Radio },
  { path: "/webhooks", label: "Webhooks", icon: Webhook },
  { path: "/pairing", label: "Pairing", icon: ShieldCheck },
  { path: "/profiles", labelKey: "profiles", label: "Profiles", icon: Users },
  { path: "/config", labelKey: "config", label: "Config", icon: Settings },
  { path: "/env", labelKey: "keys", label: "Keys", icon: KeyRound },
  { path: "/system", label: "System", icon: Wrench },
  { path: "/docs", labelKey: "documentation", label: "Documentation", icon: BookOpen },
];
