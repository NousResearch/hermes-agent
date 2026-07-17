import { Building2, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  ExecutiveActionQueue,
  ExecutiveHealthRollup,
  InsightPanel,
  KpiCard,
  MetricGrid,
  StatusPill,
  dashboardToneForSeverity,
  type DataTableColumn,
} from "@hermes/dashboard-kit";
import { buildCentralCommandData, type DailyBriefItem } from "./central-command-data";

const data = buildCentralCommandData();

const columns: DataTableColumn<DailyBriefItem>[] = [
  { id: "title", header: "Brief Item", accessor: (row) => row.title, sortValue: (row) => row.title },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "severity", header: "Severity", accessor: (row) => <StatusPill tone={dashboardToneForSeverity(row.severity)}>{row.severity}</StatusPill>, sortValue: (row) => row.severity },
  { id: "summary", header: "Read", accessor: (row) => row.summary, sortValue: (row) => row.summary },
];

export default function CentralCommandPage() {
  const healthy = data.snapshots.filter((snapshot) => snapshot.health.state === "healthy").length;
  const missingCost = data.snapshots.filter((snapshot) => !snapshot.cost?.known).length;
  const running = data.snapshots.reduce((total, snapshot) => total + (snapshot.queue?.running ?? 0), 0);

  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="Central Command"
          description="Hermes CEO layer."
          items={[
            { id: "brief", label: "Daily Brief", href: "#brief", active: true, icon: Building2 },
            { id: "actions", label: "Actions", href: "#actions" },
            { id: "rollups", label: "Rollups", href: "#rollups" },
          ]}
        />
      )}
      header={(
        <DashboardHeader
          title="Hermes Central Command"
          eyebrow="V12 command layer"
          description="One operating read across dashboard readiness, missing signals, migration priorities, and action-needed items."
          actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>}
          meta={<StatusPill tone="info">fallback signals</StatusPill>}
        />
      )}
    >
      <ExecutiveHealthRollup
        metrics={[
          { label: "Tracked Dashboards", value: data.snapshots.length, detail: "standard snapshot sources", tone: "info" },
          { label: "Healthy", value: healthy, detail: "healthy standard signals", tone: healthy ? "success" : "warning" },
          { label: "Running Work", value: running, detail: "queue rollup across snapshots", tone: running ? "success" : "neutral" },
          { label: "Missing Cost Signals", value: missingCost, detail: "V9 endpoint work remaining", tone: missingCost ? "warning" : "success" },
        ]}
      />

      <DashboardSection id="brief" title="Daily Cross-Project Brief" description="What changed, what matters, and what needs attention.">
        <DataTable columns={columns} rows={data.brief} getRowKey={(row) => row.id} />
      </DashboardSection>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <DashboardSection id="rollups" title="Business Impact Read" description="Executive interpretation generated from dashboard signals.">
          <InsightPanel title="System Read" tone="info">
            <p className="text-sm text-muted-foreground">{data.systemRead}</p>
          </InsightPanel>
          <MetricGrid columns={3} className="mt-4">
            <KpiCard label="Action Items" value={data.actions.length} detail="cross-project follow-ups" tone="warning" />
            <KpiCard label="Signal Quality" value="partial" detail="health exists before cost/capacity" tone="warning" />
            <KpiCard label="Migration Readiness" value="ready" detail="V8 workbench active" tone="success" />
          </MetricGrid>
        </DashboardSection>
        <ExecutiveActionQueue id="actions" items={data.actions.map((action) => ({
          id: action.id,
          title: action.title,
          owner: action.owner,
          urgency: action.severity,
          source: action.source ?? action.sourceDashboardId,
          due: action.due,
        }))} />
      </div>
    </DashboardShell>
  );
}
