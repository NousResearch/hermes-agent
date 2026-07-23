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
import { packageNativeMigrationTargets, type PackageNativeMigrationTarget } from "./package-native-migration-data";

const targets = packageNativeMigrationTargets;

const columns: DataTableColumn<PackageNativeMigrationTarget>[] = [
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
  { id: "retirement", header: "Retirement", accessor: (row) => <StatusPill tone={row.retirementAllowed ? "success" : "warning"}>{row.retirementAllowed ? "allowed" : "blocked"}</StatusPill>, sortValue: (row) => row.retirementAllowed ? 1 : 0 },
  { id: "next", header: "Next Step", accessor: (row) => row.nextStep, sortValue: (row) => row.nextStep },
];

export default function PackageNativeMigrationsPage() {
  const averageCompletion = Math.round(targets.reduce((total, target) => total + target.completion, 0) / targets.length);
  const ready = targets.filter((target) => target.status === "ready").length;
  const inProgress = targets.filter((target) => target.status === "in-progress").length;
  const blocked = targets.filter((target) => target.status === "blocked").length;
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
        <KpiCard label="Blocked / Planned" value={`${blocked} / ${planned}`} detail="Blocked means retirement evidence is missing" tone={blocked > 0 ? "warning" : "neutral"} />
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

function toneForStatus(status: PackageNativeMigrationTarget["status"]) {
  if (status === "in-progress") return "success" as const;
  if (status === "ready") return "info" as const;
  if (status === "blocked") return "critical" as const;
  return "neutral" as const;
}
