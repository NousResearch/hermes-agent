import { BookOpen, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { DashboardHeader, DashboardSection, DashboardShell, DashboardSidebar, DataTable, KpiCard, MetricGrid, StatusPill, type DataTableColumn } from "@hermes/dashboard-kit";
import { decisionLedger, type DecisionRecord } from "./operating-system-data";

const columns: DataTableColumn<DecisionRecord>[] = [
  { id: "decision", header: "Decision", accessor: (row) => row.decision, sortValue: (row) => row.decision },
  { id: "reason", header: "Reason", accessor: (row) => row.reason, sortValue: (row) => row.reason },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "status", header: "Status", accessor: (row) => <StatusPill tone={row.status === "active" ? "success" : "warning"}>{row.status}</StatusPill>, sortValue: (row) => row.status },
  { id: "reviewed", header: "Reviewed", accessor: (row) => row.reviewedAt, sortValue: (row) => row.reviewedAt },
];

export default function DecisionLedgerPage() {
  return (
    <DashboardShell
      sidebar={<DashboardSidebar title="Decision Ledger" description="V17 memory." items={[{ id: "decisions", label: "Decisions", href: "#decisions", active: true, icon: BookOpen }]} />}
      header={<DashboardHeader title="Memory And Decision Ledger" eyebrow="V17 context layer" description="Records why operating choices were made so future agents do not rediscover the same context." actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>} meta={<StatusPill tone="success">{decisionLedger.length} records</StatusPill>} />}
    >
      <MetricGrid columns={3}>
        <KpiCard label="Decisions" value={decisionLedger.length} detail="tracked operating choices" tone="success" />
        <KpiCard label="Active" value={decisionLedger.filter((item) => item.status === "active").length} detail="current assumptions" tone="info" />
        <KpiCard label="Needs Review" value={decisionLedger.filter((item) => item.status === "needs-review").length} detail="stale context" tone="warning" />
      </MetricGrid>
      <DashboardSection id="decisions" title="Decision Records" description="V17 is complete as a governed ledger model; persistent storage can be added after runtime architecture is chosen.">
        <DataTable columns={columns} rows={decisionLedger} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}
