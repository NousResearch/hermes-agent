import { Palette, RefreshCw } from "lucide-react";
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
  dashboardThemeProfiles,
  type DashboardThemeProfile,
  type DataTableColumn,
} from "@hermes/dashboard-kit";

const columns: DataTableColumn<DashboardThemeProfile>[] = [
  { id: "label", header: "Theme", accessor: (row) => row.label, sortValue: (row) => row.label },
  { id: "domain", header: "Domain", accessor: (row) => row.domain, sortValue: (row) => row.domain },
  { id: "density", header: "Density", accessor: (row) => <StatusPill tone="info">{row.density}</StatusPill>, sortValue: (row) => row.density },
  { id: "tone", header: "Tone", accessor: (row) => row.tone, sortValue: (row) => row.tone },
];

export default function ThemeSystemPage() {
  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="Theme System"
          description="V13 product polish."
          items={[
            { id: "themes", label: "Themes", href: "#themes", active: true, icon: Palette },
            { id: "tokens", label: "Tokens", href: "#tokens" },
          ]}
        />
      )}
      header={(
        <DashboardHeader
          title="Multi-Brand Dashboard Themes"
          eyebrow="V13 theme layer"
          description="One dashboard system with domain-specific identity for TLC, Khashi, Media Engine, and business analytics."
          actions={<Button><RefreshCw className="h-4 w-4" />Refresh</Button>}
          meta={<StatusPill tone="success">{dashboardThemeProfiles.length} themes</StatusPill>}
        />
      )}
    >
      <MetricGrid columns={4}>
        <KpiCard label="Theme Profiles" value={dashboardThemeProfiles.length} detail="governed domain themes" tone="success" />
        <KpiCard label="Density Modes" value="3" detail="compact, balanced, spacious" tone="info" />
        <KpiCard label="Contrast" value="tracked" detail="validated in future V13 hardening" tone="warning" />
        <KpiCard label="Brand Drift" value="guarded" detail="tokens instead of ad hoc colors" tone="info" />
      </MetricGrid>

      <DashboardSection id="themes" title="Theme Profiles" description="Themes tune identity without changing the underlying dashboard structure.">
        <DataTable columns={columns} rows={dashboardThemeProfiles} getRowKey={(row) => row.id} />
      </DashboardSection>

      <DashboardSection id="tokens" title="Token Swatches" description="Primary/accent/status colors for each dashboard theme.">
        <div className="grid gap-3 lg:grid-cols-2">
          {dashboardThemeProfiles.map((theme) => (
            <article key={theme.id} className="rounded-lg border border-border bg-card p-4">
              <h3 className="text-sm font-semibold text-foreground">{theme.label}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{theme.notes.join(" ")}</p>
              <div className="mt-3 grid grid-cols-5 gap-2">
                {(["primary", "accent", "success", "warning", "critical"] as const).map((key) => (
                  <div key={key} className="space-y-1">
                    <div className="h-10 rounded-md border border-border" style={{ background: theme.tokens[key] }} />
                    <div className="truncate text-xs text-muted-foreground">{key}</div>
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
