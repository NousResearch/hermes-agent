import { Radio, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { DashboardHeader, DashboardSection, DashboardShell, DashboardSidebar, DataTable, KpiCard, MetricGrid, StatusPill, type DataTableColumn } from "@hermes/dashboard-kit";
import { liveSignalIntegrations, type LiveSignalIntegration } from "./operating-system-data";

const columns: DataTableColumn<LiveSignalIntegration>[] = [
  { id: "project", header: "Project", accessor: (row) => row.project, sortValue: (row) => row.project },
  { id: "endpoint", header: "Endpoint", accessor: (row) => <span className="font-mono-ui text-xs">{row.endpoint}</span>, sortValue: (row) => row.endpoint },
  { id: "signals", header: "Signals", accessor: (row) => row.signals.length, sortValue: (row) => row.signals.length },
  { id: "status", header: "Status", accessor: (row) => <StatusPill tone={row.status === "ready" ? "success" : "warning"}>{row.status}</StatusPill>, sortValue: (row) => row.status },
  { id: "next", header: "Next Step", accessor: (row) => row.nextStep, sortValue: (row) => row.nextStep },
];

export default function LiveSignalsPage() {
  const ready = liveSignalIntegrations.filter((item) => item.status === "ready").length;
  return (
    <DashboardShell
      sidebar={<DashboardSidebar title="Live Signals" description="V15 integrations." items={[{ id: "signals", label: "Signals", href: "#signals", active: true, icon: Radio }]} />}
      header={<DashboardHeader title="Live Project Signal Integration" eyebrow="V15 signal layer" description="Tracks which projects can feed Hermes with standard dashboard snapshots." actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>} meta={<StatusPill tone="warning">{ready}/{liveSignalIntegrations.length} ready</StatusPill>} />}
    >
      <MetricGrid columns={3}>
        <KpiCard label="Projects" value={liveSignalIntegrations.length} detail="known signal sources" tone="info" />
        <KpiCard label="Ready" value={ready} detail="standard snapshots available" tone={ready ? "success" : "warning"} />
        <KpiCard label="Partial" value={liveSignalIntegrations.length - ready} detail="requires project endpoint work" tone="warning" />
      </MetricGrid>
      <DashboardSection id="signals" title="Signal Integrations" description="V15 is complete as the integration registry; live project endpoint implementation stays parked until migrations resume.">
        <DataTable columns={columns} rows={liveSignalIntegrations} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}
