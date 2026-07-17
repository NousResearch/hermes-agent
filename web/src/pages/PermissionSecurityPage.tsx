import { ShieldCheck, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { DashboardHeader, DashboardSection, DashboardShell, DashboardSidebar, DataTable, KpiCard, MetricGrid, StatusPill, type DataTableColumn } from "@hermes/dashboard-kit";
import { permissionPolicies, type PermissionPolicy } from "./operating-system-data";

const columns: DataTableColumn<PermissionPolicy>[] = [
  { id: "action", header: "Action", accessor: (row) => row.action, sortValue: (row) => row.action },
  { id: "level", header: "Level", accessor: (row) => <StatusPill tone={row.level === "admin" ? "critical" : row.level === "operator" ? "warning" : "info"}>{row.level}</StatusPill>, sortValue: (row) => row.level },
  { id: "approval", header: "Approval", accessor: (row) => row.approval, sortValue: (row) => row.approval },
  { id: "audit", header: "Audit", accessor: (row) => row.audit ? "yes" : "no", sortValue: (row) => Number(row.audit) },
];

export default function PermissionSecurityPage() {
  return (
    <DashboardShell
      sidebar={<DashboardSidebar title="Permissions" description="V20 safety." items={[{ id: "policies", label: "Policies", href: "#policies", active: true, icon: ShieldCheck }]} />}
      header={<DashboardHeader title="Secure Tool And Permission Layer" eyebrow="V20 safety rails" description="Defines what Hermes can view, operate, deploy, or change before autonomous loops run." actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>} meta={<StatusPill tone="critical">{permissionPolicies.filter((policy) => policy.level === "admin").length} admin actions</StatusPill>} />}
    >
      <MetricGrid columns={3}>
        <KpiCard label="Policies" value={permissionPolicies.length} detail="permission rules" tone="info" />
        <KpiCard label="Explicit Approval" value={permissionPolicies.filter((policy) => policy.approval === "explicit").length} detail="highest risk actions" tone="critical" />
        <KpiCard label="Audited" value={permissionPolicies.filter((policy) => policy.audit).length} detail="recorded actions" tone="success" />
      </MetricGrid>
      <DashboardSection id="policies" title="Permission Policies" description="V20 is complete as policy infrastructure; runtime enforcement hooks should be wired before autonomous execution.">
        <DataTable columns={columns} rows={permissionPolicies} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}
