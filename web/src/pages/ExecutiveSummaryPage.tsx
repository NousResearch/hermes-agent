import { useMemo, useState } from "react";
import { Building2, RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import {
  DashboardHeader,
  DashboardSection,
  DashboardShell,
  DashboardSidebar,
  ExecutiveActionQueue,
  ExecutiveCostCapacityRollup,
  ExecutiveDomainTabs,
  ExecutiveHealthRollup,
  ExecutiveProjectScorecard,
  InsightPanel,
  StatusPill,
  type DashboardTone,
} from "@hermes/dashboard-kit";
import { buildDomainTabs, domainSlug, useExecutiveSummaryData } from "./executive-data";

function toneForScore(score: number): DashboardTone {
  if (score >= 80) return "success";
  if (score >= 65) return "warning";
  return "critical";
}

export default function ExecutiveSummaryPage() {
  const [activeDomain, setActiveDomain] = useState("all");
  const { data, isFetching, refetch } = useExecutiveSummaryData();
  const summary = data;
  const projects = summary?.projects ?? [];
  const actions = summary?.actions ?? [];
  const tabs = useMemo(() => buildDomainTabs(projects), [projects]);

  const visibleProjects = useMemo(() => {
    if (activeDomain === "all") return projects;
    return projects.filter((project) => domainSlug(project.domain) === activeDomain);
  }, [activeDomain, projects]);

  const averageHealth = projects.length
    ? Math.round(projects.reduce((total, project) => total + (project.healthScore ?? 0), 0) / projects.length)
    : 0;
  const atRisk = projects.filter((project) => project.tone === "warning" || project.tone === "critical").length;

  return (
    <DashboardShell
      sidebar={(
        <DashboardSidebar
          title="TLC Executive"
          description="Cross-project operating command."
          items={[
            { id: "summary", label: "Summary", href: "#summary", active: true, icon: Building2 },
            { id: "projects", label: "Projects", href: "#projects" },
            { id: "actions", label: "Actions", href: "#actions" },
            { id: "economics", label: "Cost And Capacity", href: "#economics" },
          ]}
          footer={<div className="text-xs text-muted-foreground">V6 executive layer reference implementation.</div>}
        />
      )}
      header={(
        <DashboardHeader
          title="TLC Executive Summary"
          eyebrow="Hermes central command"
          description="A top-level operating layer that summarizes project health without replacing project-specific dashboards."
          actions={<Button onClick={() => void refetch()}><RefreshCw className="h-4 w-4" />Refresh</Button>}
          meta={(
            <>
              <StatusPill tone={toneForScore(averageHealth)}>health {averageHealth}</StatusPill>
              <StatusPill tone={atRisk ? "warning" : "success"}>{atRisk} watch items</StatusPill>
              <StatusPill tone={summary?.source === "live" ? "success" : "info"}>
                {isFetching ? "refreshing" : `${summary?.source ?? "fallback"} data`}
              </StatusPill>
            </>
          )}
        />
      )}
    >
      <ExecutiveDomainTabs tabs={tabs} activeId={activeDomain} onSelect={setActiveDomain} />

      <ExecutiveHealthRollup
        className="scroll-mt-4"
        metrics={[
          { label: "Portfolio Health", value: `${averageHealth}/100`, detail: "weighted dashboard readiness", tone: toneForScore(averageHealth) },
          { label: "Active Projects", value: projects.length, detail: "registered operating dashboards", tone: "info" },
          { label: "Needs Attention", value: atRisk, detail: "warning or critical scorecards", tone: atRisk ? "warning" : "success" },
          { label: "Action Queue", value: actions.length, detail: "cross-project follow-ups", tone: "warning" },
        ]}
      />

      <DashboardSection
        id="projects"
        title="Project Scorecards"
        description="Business-level health without drilling into every operational dashboard."
      >
        <div className="grid gap-3 lg:grid-cols-2">
          {visibleProjects.map((project) => <ExecutiveProjectScorecard key={project.id} project={project} />)}
        </div>
      </DashboardSection>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_24rem]">
        <DashboardSection
          id="economics"
          title="Cost, Capacity, And Throughput"
          description="Executive economics sourced through the V6 server-state adapter."
        >
          {summary ? (
            <ExecutiveCostCapacityRollup
              cost={summary.cost}
              capacity={summary.capacity}
              throughput={summary.throughput}
            />
          ) : null}
          <div className="mt-4">
            <InsightPanel
              title="Executive Read"
              tone="info"
            >
              <p className="text-sm text-muted-foreground">
                The control plane now uses the same query-backed adapter pattern that future project scorecards should use. The next quality jump is adding per-project cost and capacity endpoints.
              </p>
            </InsightPanel>
          </div>
        </DashboardSection>

        <ExecutiveActionQueue id="actions" items={actions} />
      </div>
    </DashboardShell>
  );
}
