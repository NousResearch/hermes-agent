import { Plug, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  DataTable,
  KpiCard,
  MetricGrid,
  StatusPill,
  dashboardPluginRequiresAdmin,
  type DashboardPluginManifest,
  type DataTableColumn,
} from "@hermes/dashboard-kit";
import { dashboardMarketplacePlugins } from "./dashboard-marketplace-data";

const columns: DataTableColumn<DashboardPluginManifest>[] = [
  { id: "label", header: "Plugin", accessor: (row) => row.label, sortValue: (row) => row.label },
  { id: "owner", header: "Owner", accessor: (row) => row.owner, sortValue: (row) => row.owner },
  { id: "category", header: "Category", accessor: (row) => row.category, sortValue: (row) => row.category },
  { id: "signals", header: "Signals", accessor: (row) => row.signals.length, sortValue: (row) => row.signals.length },
  { id: "commands", header: "Commands", accessor: (row) => <StatusPill tone={dashboardPluginRequiresAdmin(row) ? "warning" : "info"}>{row.commands.length}</StatusPill>, sortValue: (row) => row.commands.length },
];

export default function DashboardMarketplacePage() {
  const commands = dashboardMarketplacePlugins.reduce((total, plugin) => total + plugin.commands.length, 0);
  const signals = dashboardMarketplacePlugins.reduce((total, plugin) => total + plugin.signals.length, 0);
  const highRisk = dashboardMarketplacePlugins.filter(dashboardPluginRequiresAdmin).length;

  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="Marketplace"
          description="V14 plugin registry."
          items={[
            { id: "plugins", label: "Plugins", href: "#plugins", active: true, icon: Plug },
            { id: "commands", label: "Commands", href: "#commands" },
          ]}
        />
      )}
      header={(
        <DashboardHeader
          title="Dashboard Marketplace"
          eyebrow="V14 plugin layer"
          description="Projects register dashboards, panels, commands, signals, and permission requirements through a standard manifest."
          actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>}
          meta={<StatusPill tone="info">{dashboardMarketplacePlugins.length} plugins</StatusPill>}
        />
      )}
    >
      <MetricGrid columns={4}>
        <KpiCard label="Plugins" value={dashboardMarketplacePlugins.length} detail="registered manifests" tone="success" />
        <KpiCard label="Signals" value={signals} detail="declared signal contracts" tone="info" />
        <KpiCard label="Commands" value={commands} detail="permission-aware actions" tone="warning" />
        <KpiCard label="Admin Risk" value={highRisk} detail="plugins with high-risk commands" tone={highRisk ? "warning" : "success"} />
      </MetricGrid>

      <DashboardSection id="plugins" title="Registered Dashboard Plugins" description="This is the discovery layer Hermes should eventually load dynamically.">
        <DataTable columns={columns} rows={dashboardMarketplacePlugins} getRowKey={(row) => row.id} />
      </DashboardSection>

      <DashboardSection id="commands" title="Permission-Aware Commands" description="Commands declare permission and risk before Hermes can execute them.">
        <div className="grid gap-3 lg:grid-cols-2">
          {dashboardMarketplacePlugins.map((plugin) => (
            <article key={plugin.id} className="rounded-lg border border-border bg-card p-4">
              <h3 className="text-sm font-semibold text-foreground">{plugin.label}</h3>
              <div className="mt-3 space-y-2">
                {plugin.commands.map((command) => (
                  <div key={command.id} className="flex items-center justify-between gap-3 rounded-md border border-border bg-background p-2">
                    <span className="text-sm text-foreground">{command.label}</span>
                    <StatusPill tone={command.riskLevel === "high" ? "critical" : command.riskLevel === "medium" ? "warning" : "info"}>{command.permission}</StatusPill>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </div>
      </DashboardSection>
    </DashboardShell>
  );
}
