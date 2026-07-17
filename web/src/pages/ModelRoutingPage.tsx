import { Cpu, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { DashboardHeader, DashboardSection, DashboardShell, DashboardSidebar, DataTable, KpiCard, MetricGrid, StatusPill, type DataTableColumn } from "@hermes/dashboard-kit";
import { modelRoutingPolicies, type ModelRoutingPolicy } from "./operating-system-data";

const columns: DataTableColumn<ModelRoutingPolicy>[] = [
  { id: "task", header: "Task Type", accessor: (row) => row.taskType, sortValue: (row) => row.taskType },
  { id: "preferred", header: "Preferred", accessor: (row) => row.preferred, sortValue: (row) => row.preferred },
  { id: "fallback", header: "Fallback", accessor: (row) => row.fallback, sortValue: (row) => row.fallback },
  { id: "cost", header: "Cost Mode", accessor: (row) => <StatusPill tone={row.costMode === "premium-approval" ? "warning" : "info"}>{row.costMode}</StatusPill>, sortValue: (row) => row.costMode },
  { id: "approval", header: "Approval", accessor: (row) => row.approvalRequired ? "required" : "not required", sortValue: (row) => Number(row.approvalRequired) },
];

export default function ModelRoutingPage() {
  return (
    <DashboardShell
      sidebar={<DashboardSidebar title="Model Routing" description="V18 cost policy." items={[{ id: "policies", label: "Policies", href: "#policies", active: true, icon: Cpu }]} />}
      header={<DashboardHeader title="Model And Cost Routing" eyebrow="V18 model router" description="Defines when Hermes should use local Codex, cheaper models, or premium API fallback with approval." actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>} meta={<StatusPill tone="warning">{modelRoutingPolicies.filter((policy) => policy.approvalRequired).length} approval gates</StatusPill>} />}
    >
      <MetricGrid columns={3}>
        <KpiCard label="Policies" value={modelRoutingPolicies.length} detail="task routing modes" tone="info" />
        <KpiCard label="Premium Gates" value={modelRoutingPolicies.filter((policy) => policy.approvalRequired).length} detail="approval before spend" tone="warning" />
        <KpiCard label="Local First" value={modelRoutingPolicies.filter((policy) => policy.costMode === "free-local-first").length} detail="no API default" tone="success" />
      </MetricGrid>
      <DashboardSection id="policies" title="Routing Policies" description="V18 is complete as policy infrastructure; model performance logging can follow after providers are wired.">
        <DataTable columns={columns} rows={modelRoutingPolicies} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}
