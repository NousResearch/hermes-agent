import { Activity, AlertTriangle, CircleDollarSign, Database, ExternalLink, Gauge, GitBranch, ListChecks } from "lucide-react";
import {
  CommandBar,
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  KpiCard,
  MetricGrid,
  StatusPill,
  dashboardToneForHealth,
  dashboardToneForSeverity,
  type ActionNeeded,
  type DashboardSnapshot,
  type DataTableColumn,
  type DashboardTone,
} from "@hermes/dashboard-kit";
import { buildKnownDashboardSnapshots } from "./dashboard-signals";

type ProjectSnapshotKind = "media-engine" | "khashi-vc";

interface ProjectSnapshotConfig {
  kind: ProjectSnapshotKind;
  sourceId: string;
  title: string;
  eyebrow: string;
  description: string;
  recipe: string;
  productionUrl: string;
  migrationTarget: string;
}

interface SignalRow {
  id: string;
  signal: string;
  state: string;
  detail: string;
  tone: DashboardTone;
}

const configs: Record<ProjectSnapshotKind, ProjectSnapshotConfig> = {
  "media-engine": {
    kind: "media-engine",
    sourceId: "media-engine.ops",
    title: "Media Engine Package-Native Ops",
    eyebrow: "V8 shadow dashboard",
    description: "Package-native Media Engine operations surface consuming the standard DashboardSnapshot contract.",
    recipe: "pipeline-workflow-dashboard",
    productionUrl: "https://media.tlccapitalgroup.com/dashboard",
    migrationTarget: "Generated static HTML with hdk adapter classes.",
  },
  "khashi-vc": {
    kind: "khashi-vc",
    sourceId: "khashi-vc.roc",
    title: "Khashi VC Package-Native ROC",
    eyebrow: "V8 shadow dashboard",
    description: "Package-native Khashi VC research operations surface consuming the standard DashboardSnapshot contract.",
    recipe: "operations-control-room + market-asset-explorer",
    productionUrl: "https://roc.tlccapitalgroup.com/",
    migrationTarget: "Static HTML and vanilla JavaScript ROC with hdk adapter classes.",
  },
};

const actionColumns: DataTableColumn<ActionNeeded>[] = [
  { id: "title", header: "Action", accessor: (row) => row.title, sortValue: (row) => row.title },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  {
    id: "severity",
    header: "Severity",
    accessor: (row) => <StatusPill tone={dashboardToneForSeverity(row.severity)}>{row.severity}</StatusPill>,
    sortValue: (row) => row.severity,
  },
  { id: "next", header: "Next Step", accessor: (row) => row.nextStep ?? "No next step recorded.", sortValue: (row) => row.nextStep ?? "" },
];

const signalColumns: DataTableColumn<SignalRow>[] = [
  { id: "signal", header: "Signal", accessor: (row) => row.signal, sortValue: (row) => row.signal },
  { id: "state", header: "State", accessor: (row) => <StatusPill tone={row.tone}>{row.state}</StatusPill>, sortValue: (row) => row.state },
  { id: "detail", header: "Detail", accessor: (row) => row.detail, sortValue: (row) => row.detail },
];

export function MediaEnginePackageNativePage() {
  return <ProjectSnapshotDashboardPage kind="media-engine" />;
}

export function KhashiVcPackageNativePage() {
  return <ProjectSnapshotDashboardPage kind="khashi-vc" />;
}

export default function ProjectSnapshotDashboardPage({ kind }: { kind: ProjectSnapshotKind }) {
  const config = configs[kind];
  const snapshot = findSnapshot(config.sourceId);
  const healthTone = dashboardToneForHealth(snapshot.health.state);
  const capacityUsed = snapshot.capacity?.used ?? snapshot.queue?.running ?? 0;
  const capacityLimit = snapshot.capacity?.limit ?? 0;
  const costValue = snapshot.cost?.known && typeof snapshot.cost.amountUsd === "number"
    ? `$${snapshot.cost.amountUsd.toFixed(2)}`
    : "Unknown";
  const signalRows = buildSignalRows(snapshot);

  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title={snapshot.source.label}
          description="Package-native migration surface."
          items={[
            { id: "overview", label: "Overview", href: "#overview", active: true, icon: Activity },
            { id: "signals", label: "Signals", href: "#signals", icon: Database },
            { id: "actions", label: "Actions", href: "#actions", icon: ListChecks },
            { id: "parity", label: "Parity Gates", href: "#parity", icon: GitBranch },
          ]}
          footer={<StatusPill tone={healthTone}>{snapshot.health.state}</StatusPill>}
        />
      )}
      header={(
        <DashboardHeader
          title={config.title}
          eyebrow={config.eyebrow}
          description={config.description}
          actions={(
            <a
              className="inline-flex h-10 items-center gap-2 rounded-md border border-border bg-background px-3 text-sm font-medium text-foreground hover:bg-muted"
              href={config.productionUrl}
              rel="noreferrer"
              target="_blank"
            >
              <ExternalLink className="h-4 w-4" aria-hidden="true" />
              Production
            </a>
          )}
          meta={(
            <>
              <StatusPill tone="info">{config.recipe}</StatusPill>
              <StatusPill tone={healthTone}>{snapshot.health.freshness ?? "unknown"} signal</StatusPill>
            </>
          )}
        />
      )}
    >
      <MetricGrid id="overview" columns={4}>
        <KpiCard
          label="Health"
          value={snapshot.health.score ?? "Unknown"}
          detail={snapshot.health.message ?? "No health detail exposed."}
          tone={healthTone}
          icon={Activity}
        />
        <KpiCard
          label="Queue"
          value={snapshot.queue?.running ?? 0}
          detail={`${snapshot.queue?.queued ?? 0} queued, ${snapshot.queue?.failed ?? 0} failed`}
          tone={(snapshot.queue?.failed ?? 0) > 0 ? "warning" : "info"}
          icon={ListChecks}
        />
        <KpiCard
          label="Cost"
          value={costValue}
          detail={`${snapshot.cost?.tokenCount ?? 0} tokens, ${snapshot.cost?.apiCalls ?? 0} API calls`}
          tone={snapshot.cost?.known ? "success" : "warning"}
          icon={CircleDollarSign}
        />
        <KpiCard
          label="Capacity"
          value={capacityLimit > 0 ? `${capacityUsed} / ${capacityLimit}` : capacityUsed}
          detail={snapshot.capacity?.message ?? "Capacity inferred from active queue signal."}
          tone={toneForCapacity(snapshot)}
          icon={Gauge}
        />
      </MetricGrid>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <DashboardSection
          id="signals"
          title="Dashboard Snapshot Signals"
          description="Canonical signals exposed for central command and downstream business-unit dashboards."
        >
          <DataTable columns={signalColumns} rows={signalRows} getRowKey={(row) => row.id} />
        </DashboardSection>

        <DashboardSection
          id="parity"
          title="V8 Parity Position"
          description="This route proves the package-native shell and snapshot contract before adapter retirement."
        >
          <div className="space-y-3 text-sm text-muted-foreground">
            <div>
              <div className="font-medium text-foreground">Current surface</div>
              <div>{config.migrationTarget}</div>
            </div>
            <div>
              <div className="font-medium text-foreground">Target surface</div>
              <div>Package-native React dashboard using `@hermes/dashboard-kit` components.</div>
            </div>
            <div>
              <div className="font-medium text-foreground">Retirement status</div>
              <div>Static adapter stays until live screenshots, rollback notes, and production parity are captured.</div>
            </div>
          </div>
        </DashboardSection>
      </div>

      <DashboardSection
        id="actions"
        title="Actions Needed"
        description="Migration and signal work required before this can replace the current production dashboard."
      >
        <DataTable
          columns={actionColumns}
          rows={snapshot.actions ?? []}
          getRowKey={(row) => row.id}
          emptyTitle="No actions recorded"
          emptyDescription="This snapshot does not currently report required operator actions."
        />
      </DashboardSection>

      <CommandBar
        title="Adapter Retirement Gate"
        description="Retiring static adapters is intentionally blocked until live parity evidence exists."
        actions={[
          { id: "validate-v8", label: "Validate V8", icon: GitBranch, permission: "operator", riskLevel: "low" },
          {
            id: "retire-static-adapter",
            label: "Retire Static Adapter",
            disabled: true,
            disabledReason: "Requires production screenshot evidence, rollback path, and full API parity.",
            icon: AlertTriangle,
            permission: "admin",
            riskLevel: "high",
          },
        ]}
      />
    </DashboardShell>
  );
}

function findSnapshot(sourceId: string): DashboardSnapshot {
  const snapshot = buildKnownDashboardSnapshots().find((candidate) => candidate.source.id === sourceId);
  if (!snapshot) throw new Error(`Unknown dashboard source: ${sourceId}`);
  return snapshot;
}

function buildSignalRows(snapshot: DashboardSnapshot): SignalRow[] {
  return [
    {
      id: "health",
      signal: "HealthSnapshot",
      state: snapshot.health.state,
      detail: snapshot.health.message ?? "No health message.",
      tone: dashboardToneForHealth(snapshot.health.state),
    },
    {
      id: "cost",
      signal: "CostSnapshot",
      state: snapshot.cost?.known ? snapshot.cost.period : "missing",
      detail: snapshot.cost?.message ?? `${snapshot.cost?.amountUsd ?? 0} USD tracked.`,
      tone: snapshot.cost?.known ? "success" : "warning",
    },
    {
      id: "capacity",
      signal: "CapacitySnapshot",
      state: snapshot.capacity?.pressure ?? "unknown",
      detail: snapshot.capacity?.message ?? "No capacity contract exposed.",
      tone: toneForCapacity(snapshot),
    },
    {
      id: "queue",
      signal: "QueueSnapshot",
      state: `${snapshot.queue?.running ?? 0} running`,
      detail: `${snapshot.queue?.queued ?? 0} queued, ${snapshot.queue?.blocked ?? 0} blocked, ${snapshot.queue?.stale ?? 0} stale.`,
      tone: (snapshot.queue?.failed ?? 0) > 0 ? "warning" : "info",
    },
    {
      id: "deployment",
      signal: "DeploymentSignal",
      state: snapshot.deployment?.status ?? "unknown",
      detail: snapshot.deployment?.message ?? "No deployment signal exposed.",
      tone: snapshot.deployment?.status === "current" ? "success" : "warning",
    },
  ];
}

function toneForCapacity(snapshot: DashboardSnapshot): DashboardTone {
  if (!snapshot.capacity?.known) return "warning";
  if (snapshot.capacity.pressure === "high") return "critical";
  if (snapshot.capacity.pressure === "medium") return "warning";
  if (snapshot.capacity.pressure === "low") return "success";
  return "unknown";
}
