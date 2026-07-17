import { ListChecks, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { DashboardHeader, DashboardSection, DashboardShell, DashboardSidebar, DataTable, KpiCard, MetricGrid, StatusPill, dashboardToneForSeverity, type DataTableColumn } from "@hermes/dashboard-kit";
import { routedTasks, type RoutedTask } from "./operating-system-data";

const columns: DataTableColumn<RoutedTask>[] = [
  { id: "title", header: "Task", accessor: (row) => row.title, sortValue: (row) => row.title },
  { id: "source", header: "Source", accessor: (row) => row.source, sortValue: (row) => row.source },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "priority", header: "Priority", accessor: (row) => <StatusPill tone={dashboardToneForSeverity(row.priority)}>{row.priority}</StatusPill>, sortValue: (row) => row.priority },
  { id: "status", header: "Status", accessor: (row) => row.status, sortValue: (row) => row.status },
];

export default function TaskRoutingPage() {
  const critical = routedTasks.filter((task) => task.priority === "critical").length;
  return (
    <DashboardShell
      sidebar={<DashboardSidebar title="Task Routing" description="V16 work intake." items={[{ id: "tasks", label: "Tasks", href: "#tasks", active: true, icon: ListChecks }]} />}
      header={<DashboardHeader title="Agent Task Routing" eyebrow="V16 work intake" description="Turns dashboard signals into prioritized work with owner, source, status, and next step." actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>} meta={<StatusPill tone={critical ? "critical" : "info"}>{critical} critical</StatusPill>} />}
    >
      <MetricGrid columns={3}>
        <KpiCard label="Routed Tasks" value={routedTasks.length} detail="signal-derived work items" tone="info" />
        <KpiCard label="Assigned" value={routedTasks.filter((task) => task.status === "assigned").length} detail="owned now" tone="success" />
        <KpiCard label="Queued" value={routedTasks.filter((task) => task.status === "queued").length} detail="waiting for next cycle" tone="warning" />
      </MetricGrid>
      <DashboardSection id="tasks" title="Work Intake Queue" description="V16 is complete as a routing model; runtime task execution comes after permission rails and operating loops mature.">
        <DataTable columns={columns} rows={routedTasks} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}
