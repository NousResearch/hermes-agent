import { RotateCw, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { DashboardHeader, DashboardSection, DashboardShell, DashboardSidebar, DataTable, KpiCard, MetricGrid, StatusPill, type DataTableColumn } from "@hermes/dashboard-kit";
import { operatingLoops, type OperatingLoop } from "./operating-system-data";

const columns: DataTableColumn<OperatingLoop>[] = [
  { id: "name", header: "Loop", accessor: (row) => row.name, sortValue: (row) => row.name },
  { id: "cadence", header: "Cadence", accessor: (row) => row.cadence, sortValue: (row) => row.cadence },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "status", header: "Status", accessor: (row) => <StatusPill tone={row.status === "ready" ? "success" : "warning"}>{row.status}</StatusPill>, sortValue: (row) => row.status },
  { id: "output", header: "Output", accessor: (row) => row.output, sortValue: (row) => row.output },
];

export default function OperatingLoopsPage() {
  return (
    <DashboardShell
      sidebar={<DashboardSidebar title="Operating Loops" description="V19 autonomy." items={[{ id: "loops", label: "Loops", href: "#loops", active: true, icon: RotateCw }]} />}
      header={<DashboardHeader title="Autonomous Operating Loops" eyebrow="V19 recurring operations" description="Defines recurring reviews Hermes can run once permissions, model routing, and signals are mature." actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>} meta={<StatusPill tone="info">{operatingLoops.length} loops</StatusPill>} />}
    >
      <MetricGrid columns={3}>
        <KpiCard label="Loops" value={operatingLoops.length} detail="recurring operating cadences" tone="info" />
        <KpiCard label="Ready" value={operatingLoops.filter((loop) => loop.status === "ready").length} detail="can be run manually" tone="success" />
        <KpiCard label="Draft" value={operatingLoops.filter((loop) => loop.status === "draft").length} detail="needs signal/permission wiring" tone="warning" />
      </MetricGrid>
      <DashboardSection id="loops" title="Operating Loop Registry" description="V19 is complete as a loop registry; autonomous execution should wait on V20 permission rails.">
        <DataTable columns={columns} rows={operatingLoops} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}
