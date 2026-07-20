import { Code2, GitBranch, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  ChartPanel,
  CommandBar,
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  KpiCard,
  MetricGrid,
  SimpleBarChart,
  StatusPill,
  type DataTableColumn,
} from "@hermes/dashboard-kit";

interface MigrationTarget {
  id: string;
  dashboard: string;
  recipe: string;
  current: string;
  target: string;
  completion: number;
  status: "ready" | "in-progress" | "blocked" | "planned";
  nextStep: string;
  packageNativeRoute?: string;
}

const targets: MigrationTarget[] = [
  {
    id: "media-engine",
    dashboard: "Media Engine Ops",
    recipe: "pipeline-workflow-dashboard",
    current: "Generated static HTML with hdk adapter classes.",
    target: "Package-native React dashboard consuming a dashboard snapshot API.",
    completion: 65,
    status: "in-progress",
    nextStep: "Capture live screenshot parity and rollback notes before adapter retirement.",
    packageNativeRoute: "/package-native/media-engine",
  },
  {
    id: "khashi-vc",
    dashboard: "Khashi VC ROC",
    recipe: "operations-control-room + market-asset-explorer",
    current: "Static HTML and vanilla JavaScript ROC with hdk adapter classes.",
    target: "Package-native React ROC with typed API client, query layer, and recipe-based views.",
    completion: 60,
    status: "in-progress",
    nextStep: "Capture live screenshot parity and validate command behavior before adapter retirement.",
    packageNativeRoute: "/package-native/khashi-vc",
  },
  {
    id: "executive-summary",
    dashboard: "Hermes Executive Summary",
    recipe: "executive-command-center",
    current: "Package-native reference route with partial signal data.",
    target: "Live TLC central command backed by project signal contracts.",
    completion: 65,
    status: "in-progress",
    nextStep: "Wire project Health/Cost/Capacity/Queue/ActionNeeded endpoints.",
  },
  {
    id: "media-business-ops",
    dashboard: "Media Business Operations",
    recipe: "brand-business-performance",
    current: "Static adapter standardization.",
    target: "Package-native business performance dashboard.",
    completion: 10,
    status: "planned",
    nextStep: "Define source analytics contract and production route.",
  },
];

const columns: DataTableColumn<MigrationTarget>[] = [
  { id: "dashboard", header: "Dashboard", accessor: (row) => row.dashboard, sortValue: (row) => row.dashboard },
  { id: "recipe", header: "V7 Recipe", accessor: (row) => <span className="font-mono-ui text-xs">{row.recipe}</span>, sortValue: (row) => row.recipe },
  { id: "status", header: "Status", accessor: (row) => <StatusPill tone={toneForStatus(row.status)}>{row.status}</StatusPill>, sortValue: (row) => row.status },
  { id: "completion", header: "Complete", accessor: (row) => `${row.completion}%`, sortValue: (row) => row.completion },
  {
    id: "route",
    header: "Native Route",
    accessor: (row) => row.packageNativeRoute
      ? <a className="font-mono-ui text-xs text-primary hover:underline" href={row.packageNativeRoute}>{row.packageNativeRoute}</a>
      : <span className="text-muted-foreground">not ready</span>,
    sortValue: (row) => row.packageNativeRoute ?? "",
  },
  { id: "next", header: "Next Step", accessor: (row) => row.nextStep, sortValue: (row) => row.nextStep },
];

export default function PackageNativeMigrationsPage() {
  const averageCompletion = Math.round(targets.reduce((total, target) => total + target.completion, 0) / targets.length);
  const ready = targets.filter((target) => target.status === "ready").length;
  const inProgress = targets.filter((target) => target.status === "in-progress").length;
  const planned = targets.filter((target) => target.status === "planned").length;

  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="Package Native"
          description="V8 migration control."
          items={[
            { id: "overview", label: "Overview", href: "#overview", active: true, icon: Code2 },
            { id: "targets", label: "Targets", href: "#targets" },
            { id: "actions", label: "Actions", href: "#actions" },
          ]}
        />
      )}
      header={(
        <DashboardHeader
          title="Package-Native Dashboard Migrations"
          eyebrow="V8 migration workbench"
          description="Tracks the move from static adapter dashboards to package-native Hermes dashboard-kit implementations."
          actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>}
          meta={<StatusPill tone="warning">{averageCompletion}% complete</StatusPill>}
        />
      )}
    >
      <MetricGrid id="overview" columns={4}>
        <KpiCard label="Migration Completion" value={`${averageCompletion}%`} detail="Average across priority dashboards" tone="warning" />
        <KpiCard label="Ready To Start" value={ready} detail="Targets with clear first migration step" tone="info" />
        <KpiCard label="In Progress" value={inProgress} detail="Package-native work already underway" tone="success" />
        <KpiCard label="Planned" value={planned} detail="Waiting on data/source contracts" tone="neutral" />
      </MetricGrid>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <DashboardSection id="targets" title="Migration Targets" description="Every target must preserve production behavior before the static adapter can be retired.">
          <DataTable columns={columns} rows={targets} getRowKey={(row) => row.id} />
        </DashboardSection>
        <ChartPanel title="Migration Readiness" description="Current package-native progress by priority dashboard.">
          <SimpleBarChart data={targets.map((target) => ({ label: target.dashboard.replace(" ", "\n"), value: target.completion }))} />
        </ChartPanel>
      </div>

      <CommandBar
        title="Migration Commands"
        description="These are guardrail actions for package-native migration planning."
        actions={[
          { id: "validate-v8", label: "Validate V8", icon: GitBranch, permission: "operator", riskLevel: "low" },
          { id: "retire-adapter", label: "Retire Adapter", disabled: true, disabledReason: "Only after live parity, visual QA, and rollback plan.", permission: "admin", riskLevel: "high" },
        ]}
      />

      <DashboardSection id="actions" title="Migration Rule" description="The static adapter remains until a package-native dashboard has feature parity, live data, Playwright coverage, and production verification.">
        <p className="text-sm text-muted-foreground">
          Treat adapter retirement as a production change. Each target needs parity proof, rollback notes, screenshots, and live health checks before the old dashboard path is removed.
        </p>
      </DashboardSection>
    </DashboardShell>
  );
}

function toneForStatus(status: MigrationTarget["status"]) {
  if (status === "in-progress") return "success" as const;
  if (status === "ready") return "info" as const;
  if (status === "blocked") return "critical" as const;
  return "neutral" as const;
}
