import { useState, useEffect, useLayoutEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/lib/api";
import { usePageHeader } from "@/contexts/usePageHeader";
import {
  VariantPicker,
  readVariant,
  writeVariant,
} from "@/components/projects/VariantPicker";
import { ProjectsWorkbench } from "@/components/projects/ProjectsWorkbench";
import { ProjectsOverview } from "@/components/projects/ProjectsOverview";

export default function ProjectsPage() {
  const [searchParams] = useSearchParams();
  const initialProjectId = searchParams.get("id") ?? undefined;
  const { setAfterTitle, setEnd } = usePageHeader();

  const [variant, setVariant] = useState<"workbench" | "overview">(readVariant);
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [unassignedCount, setUnassignedCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .getProjects()
      .then((res) => {
        if (cancelled) return;
        setProjects(res.projects);
        setUnassignedCount(res.unassigned.sessionCount);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setError(e instanceof Error ? e.message : "Failed to load projects");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function handleVariantChange(v: "workbench" | "overview") {
    setVariant(v);
    writeVariant(v);
  }

  // Counts + variant picker live in the shared page banner (like Sessions /
  // Cron) so the route body gets the full viewport height for the workbench.
  useLayoutEffect(() => {
    if (loading || error) {
      setAfterTitle(null);
      return;
    }
    setAfterTitle(
      <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40 whitespace-nowrap">
        {projects.length} projects
        {unassignedCount > 0 && ` · ${unassignedCount} unassigned`}
      </span>,
    );
    return () => {
      setAfterTitle(null);
    };
  }, [loading, error, projects.length, unassignedCount, setAfterTitle]);

  useEffect(() => {
    setEnd(<VariantPicker value={variant} onChange={handleVariantChange} />);
    return () => {
      setEnd(null);
    };
  }, [variant, setEnd]);

  if (variant === "workbench") {
    return (
      <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-hidden border border-midground/10 rounded-sm">
        <ProjectsWorkbench
          projects={projects}
          unassignedCount={unassignedCount}
          loading={loading}
          error={error}
          initialProjectId={initialProjectId}
        />
      </div>
    );
  }

  return (
    <div className="flex min-h-0 w-full min-w-0 flex-1 flex-col overflow-y-auto">
      <ProjectsOverview
        projects={projects}
        unassignedCount={unassignedCount}
        loading={loading}
        error={error}
      />
    </div>
  );
}
