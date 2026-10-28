import { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { api } from "@/lib/api";
import type { ProjectSummary } from "@/lib/api";
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

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Topbar */}
      <div className="flex items-center gap-3.5 px-5 h-[52px] shrink-0 border-b border-midground/10">
        <h1 className="font-mono font-medium tracking-[0.14em] uppercase text-sm">
          Projects
        </h1>
        {!loading && !error && (
          <span className="font-mono text-[10.5px] uppercase tracking-wider text-midground/40">
            {projects.length} projects
            {unassignedCount > 0 && ` · ${unassignedCount} unassigned`}
          </span>
        )}
        <div className="ml-auto">
          <VariantPicker value={variant} onChange={handleVariantChange} />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 min-h-0 overflow-auto">
        {variant === "workbench" ? (
          <div className="h-full">
            <ProjectsWorkbench
              projects={projects}
              unassignedCount={unassignedCount}
              loading={loading}
              error={error}
              initialProjectId={initialProjectId}
            />
          </div>
        ) : (
          <ProjectsOverview
            projects={projects}
            unassignedCount={unassignedCount}
            loading={loading}
            error={error}
          />
        )}
      </div>
    </div>
  );
}
