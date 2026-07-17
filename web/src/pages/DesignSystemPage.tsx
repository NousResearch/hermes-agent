import { useState } from "react";
import {
  Activity,
  AlertTriangle,
  Database,
  Pause,
  Play,
  RefreshCw,
  ShieldCheck,
  Square,
} from "lucide-react";
import {
  ActivityTimeline,
  CapacityMeter,
  ChartPanel,
  CommandBar,
  DashboardEmptyState,
  DashboardErrorState,
  DashboardHeader,
  DashboardLauncher,
  DashboardLoadingState,
  DashboardSection,
  DataTable,
  DateRangeToggle,
  FilterBar,
  FindingCard,
  HeatmapGrid,
  InsightPanel,
  KpiCard,
  MetricGrid,
  ProjectSwitcher,
  QueuePanel,
  RecommendationCard,
  RunStatusPanel,
  SearchInput,
  SegmentedControl,
  SimpleBarChart,
  SimpleLineChart,
  StatusPill,
  type DataTableColumn,
  type DashboardRegistryEntry,
  type QueueItem,
} from "@hermes/dashboard-kit";

interface ExampleRun {
  id: string;
  name: string;
  status: string;
  owner: string;
  age: string;
}

const exampleRows: ExampleRun[] = [
  { id: "run-001", name: "Market scan", status: "running", owner: "Khashi", age: "4m" },
  { id: "run-002", name: "Brand generation", status: "completed", owner: "Media Engine", age: "21m" },
  { id: "run-003", name: "Dashboard sync", status: "blocked", owner: "Hermes OS", age: "1h" },
];

const columns: DataTableColumn<ExampleRun>[] = [
  { id: "id", header: "ID", accessor: (row) => <span className="font-mono-ui text-xs text-muted-foreground">{row.id}</span>, sortValue: (row) => row.id },
  { id: "name", header: "Name", accessor: (row) => row.name, sortValue: (row) => row.name },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "status", header: "Status", accessor: (row) => <StatusPill tone={row.status === "blocked" ? "critical" : row.status === "completed" ? "success" : "info"}>{row.status}</StatusPill>, sortValue: (row) => row.status },
  { id: "age", header: "Age", accessor: (row) => row.age, sortValue: (row) => row.age },
];

const dashboards: DashboardRegistryEntry[] = [
  {
    id: "nous-hermes-agent.dashboard",
    label: "Nous Hermes Agent",
    description: "Official agent dashboard for chat, profiles, models, sessions, tools, and setup.",
    url: "/",
    healthUrl: "/api/status",
    status: "current",
    category: "control plane",
    owner: "Hermes",
  },
  {
    id: "khashi-vc.roc",
    label: "Khashi VC ROC",
    description: "Research operations center for markets, scheduler capacity, experiments, cost, and findings.",
    productionUrl: "https://roc.tlccapitalgroup.com",
    status: "online",
    category: "research",
    owner: "Khashi",
  },
  {
    id: "missing.dashboard",
    label: "Missing Manifest Example",
    description: "Shows how unavailable dashboards should render in the launcher.",
    status: "missing",
    category: "example",
  },
];

const queueItems: QueueItem[] = [
  { id: "q-1", label: "Generate morning posts", status: "running", detail: "2 brands active" },
  { id: "q-2", label: "Refresh dashboard registry", status: "queued" },
  { id: "q-3", label: "Promote production build", status: "blocked", detail: "Awaiting approval" },
];

export default function DesignSystemPage() {
  const [query, setQuery] = useState("");
  const [range, setRange] = useState<"7d" | "30d" | "90d">("30d");
  const [segment, setSegment] = useState<"all" | "active" | "blocked">("all");

  const filteredRows = exampleRows.filter((row) => {
    const matchesQuery = query
      ? `${row.id} ${row.name} ${row.owner} ${row.status}`.toLowerCase().includes(query.toLowerCase())
      : true;
    const matchesSegment = segment === "all" ? true : segment === "active" ? row.status === "running" : row.status === "blocked";
    return matchesQuery && matchesSegment;
  });

  return (
    <div className="space-y-5">
      <DashboardHeader
        title="Hermes Dashboard Design System"
        eyebrow="Component gallery"
        description="Approved dashboard primitives for Hermes/TLC operational dashboards."
        meta={(
          <>
            <StatusPill tone="success">V1 complete</StatusPill>
            <StatusPill tone="info">V2 gallery</StatusPill>
          </>
        )}
      />

      <MetricGrid columns={4}>
        <KpiCard label="Design System" value="55%" detail="Current maturity estimate" icon={ShieldCheck} tone="info" />
        <KpiCard label="Components" value="28" detail="Reusable primitives" icon={Database} />
        <KpiCard label="Dashboards" value="3" detail="Example registry entries" icon={Activity} tone="success" />
        <KpiCard label="Risk" value="Medium" detail="Cross-project adoption still pending" icon={AlertTriangle} tone="warning" />
      </MetricGrid>

      <DashboardSection title="Status And Capacity" description="Use these patterns for operational state, limits, and health.">
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              <StatusPill tone="success">healthy</StatusPill>
              <StatusPill tone="warning">warning</StatusPill>
              <StatusPill tone="critical">critical</StatusPill>
              <StatusPill tone="unknown">unknown</StatusPill>
            </div>
            <CapacityMeter used={68} limit={100} label="Scheduler capacity" />
            <RunStatusPanel running={12} queued={8} failed={1} completed={84} />
          </div>
          <QueuePanel title="Queue Pattern" items={queueItems} />
        </div>
      </DashboardSection>

      <CommandBar
        title="Command Pattern"
        description="Operator actions should be grouped, labeled, and visually separated from read-only analytics."
        actions={[
          { id: "start", label: "Start", icon: Play, tone: "success" },
          { id: "pause", label: "Pause", icon: Pause, tone: "warning" },
          { id: "stop", label: "Stop", icon: Square, tone: "critical" },
          { id: "refresh", label: "Refresh", icon: RefreshCw },
        ]}
      />

      <DashboardSection title="Tables And Filters" description="Use DataTable and shared filter controls for list-heavy dashboards.">
        <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
          <FilterBar>
            <SearchInput value={query} onChange={setQuery} placeholder="Search runs" />
            <SegmentedControl
              value={segment}
              onChange={setSegment}
              options={[
                { value: "all", label: "All" },
                { value: "active", label: "Active" },
                { value: "blocked", label: "Blocked" },
              ]}
            />
          </FilterBar>
          <DateRangeToggle value={range} onChange={setRange} />
        </div>
        <DataTable
          rows={filteredRows}
          columns={columns}
          getRowKey={(row) => row.id}
          emptyTitle="No matching runs"
          emptyDescription="Adjust the search or filter."
        />
      </DashboardSection>

      <div className="grid gap-4 lg:grid-cols-2">
        <ChartPanel title="Chart Panel" description="Charts should live inside shared panel chrome.">
          <SimpleBarChart
            data={[
              { label: "Khashi", value: 42 },
              { label: "Media", value: 31 },
              { label: "Hermes", value: 18 },
            ]}
          />
        </ChartPanel>
        <ChartPanel title="Trend Pattern" description="Simple line charts can be used until the chart library layer is standardized.">
          <SimpleLineChart
            data={[
              { label: "Mon", value: 12 },
              { label: "Tue", value: 19 },
              { label: "Wed", value: 14 },
              { label: "Thu", value: 28 },
              { label: "Fri", value: 35 },
            ]}
          />
        </ChartPanel>
      </div>

      <ChartPanel title="Heatmap Pattern" description="Useful for coverage, scheduler, and market/category density.">
        <HeatmapGrid
          rows={["Liquidity", "Spread", "Catalyst"]}
          columns={["0-6h", "6-24h", "1-2d", "2d+"]}
          values={{
            "Liquidity:0-6h": 4,
            "Liquidity:6-24h": 8,
            "Liquidity:1-2d": 13,
            "Spread:6-24h": 2,
            "Catalyst:0-6h": 5,
          }}
        />
      </ChartPanel>

      <div className="grid gap-4 lg:grid-cols-2">
        <InsightPanel title="Insight Pattern" tone="info">
          <FindingCard
            title="Evidence is still forming"
            description="Use finding cards when the system has observed a repeated pattern but has not promoted it to a recommendation."
            evidence="Example: 51 evidence points, 0 promoted closures."
            tone="info"
          />
          <RecommendationCard
            title="Raise dashboard adoption first"
            action="Migrate the most complex dashboard after the gallery is stable."
            confidence={0.74}
          />
        </InsightPanel>

        <DashboardSection title="Activity Pattern" description="Use timelines for audit, scheduler, and generation events.">
          <ActivityTimeline
            events={[
              { id: "a1", title: "Gallery generated", timestamp: "now", tone: "success" },
              { id: "a2", title: "Registry sync queued", timestamp: "2m", tone: "info" },
              { id: "a3", title: "Manifest missing", timestamp: "8m", tone: "warning", description: "One dashboard has no production URL." },
            ]}
          />
        </DashboardSection>
      </div>

      <DashboardLauncher dashboards={dashboards} currentId="nous-hermes-agent.dashboard" />

      <DashboardSection title="State Patterns" description="Every data surface needs explicit loading, empty, and error states.">
        <div className="grid gap-3 lg:grid-cols-3">
          <DashboardLoadingState label="Loading example" />
          <DashboardEmptyState title="Empty example" description="No records matched the selected filters." />
          <DashboardErrorState title="Error example" message="The request failed or was aborted." />
        </div>
      </DashboardSection>

      <DashboardSection title="Project Switcher" description="Use this pattern when one dashboard can switch context across registered projects.">
        <ProjectSwitcher
          currentId="nous-hermes-agent.dashboard"
          projects={dashboards.map((dashboard) => ({ id: dashboard.id, label: dashboard.label, status: dashboard.status }))}
        />
      </DashboardSection>
    </div>
  );
}
