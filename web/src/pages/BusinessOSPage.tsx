import { BriefcaseBusiness, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { DashboardHeader, DashboardSection, DashboardShell, DashboardSidebar, DataTable, KpiCard, MetricGrid, ProgressMetric, type DataTableColumn } from "@hermes/dashboard-kit";
import { businessScorecards, type BusinessScorecard } from "./operating-system-data";

const columns: DataTableColumn<BusinessScorecard>[] = [
  { id: "business", header: "Business", accessor: (row) => row.business, sortValue: (row) => row.business },
  { id: "health", header: "Health", accessor: (row) => <ProgressMetric label="Health" value={row.health} tone={row.health >= 75 ? "success" : "warning"} />, sortValue: (row) => row.health },
  { id: "revenue", header: "Revenue Signal", accessor: (row) => row.revenueSignal, sortValue: (row) => row.revenueSignal },
  { id: "cost", header: "Cost Signal", accessor: (row) => row.costSignal, sortValue: (row) => row.costSignal },
  { id: "focus", header: "Operating Focus", accessor: (row) => row.operatingFocus, sortValue: (row) => row.operatingFocus },
];

export default function BusinessOSPage() {
  const average = Math.round(businessScorecards.reduce((total, scorecard) => total + scorecard.health, 0) / businessScorecards.length);
  return (
    <DashboardShell
      sidebar={<DashboardSidebar title="Business OS" description="V21 business layer." items={[{ id: "scorecards", label: "Scorecards", href: "#scorecards", active: true, icon: BriefcaseBusiness }]} />}
      header={<DashboardHeader title="TLC Business Operating System" eyebrow="V21 business layer" description="Connects project dashboards into business-unit health, revenue/cost signal maturity, and operating focus." actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>} />}
    >
      <MetricGrid columns={3}>
        <KpiCard label="Business Health" value={`${average}/100`} detail="current scorecard average" tone={average >= 75 ? "success" : "warning"} />
        <KpiCard label="Businesses" value={businessScorecards.length} detail="tracked operating units" tone="info" />
        <KpiCard label="Revenue Signals" value="early" detail="not yet connected to finance systems" tone="warning" />
      </MetricGrid>
      <DashboardSection id="scorecards" title="Business Unit Scorecards" description="V21 is complete as a business OS model; live finance/revenue integrations can come after project signals mature.">
        <DataTable columns={columns} rows={businessScorecards} getRowKey={(row) => row.id} />
      </DashboardSection>
    </DashboardShell>
  );
}
