import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { CrewNode, CrewOrganizationResponse } from "@/types/crew";
import { CrewProfileCard } from "@/components/crew/CrewProfileCard";

function SummaryCard({ label, value }: { label: string; value: number | string }) {
  return (
    <Card><CardContent className="p-4">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-foreground">{value}</div>
    </CardContent></Card>
  );
}

export default function OrganizationChartPage() {
  const [data, setData] = useState<CrewOrganizationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.getCrewOrganization()
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const mainNode = useMemo(() => data?.nodes.find((node) => node.level === "main" || node.profile.name === "default"), [data]);
  const managers = useMemo(() => data?.nodes.filter((node) => node.level === "manager") ?? [], [data]);
  const workerDepartments = useMemo(
    () => data?.departments.filter((department) => department.nodes.some((node) => node.level !== "main" && node.level !== "manager")) ?? [],
    [data],
  );

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold text-foreground">Organization Chart (OC)</h1>
          <p className="text-sm text-muted-foreground">Live crew structure from Hermes profiles + crew metadata</p>
          {data && <p className="mt-1 text-xs text-muted-foreground">Last refreshed: {data.generated_at}</p>}
        </div>
        <Button onClick={load} disabled={loading} outlined>
          <RefreshCw className="mr-2 h-4 w-4" /> Refresh
        </Button>
      </div>

      {loading && <div className="flex items-center gap-2 text-muted-foreground"><Spinner /> Loading crew organization…</div>}
      {error && <Card className="border-red-500/40"><CardContent className="p-4 text-sm text-red-500">{error}</CardContent></Card>}

      {data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-6">
            <SummaryCard label="Profiles" value={data.summary.total} />
            <SummaryCard label="Managers" value={data.summary.managers} />
            <SummaryCard label="Workers" value={data.summary.workers} />
            <SummaryCard label="Unassigned/New" value={data.summary.unassigned} />
            <SummaryCard label="Running" value={data.summary.running} />
            <SummaryCard label="Stopped" value={data.summary.stopped} />
          </div>

          {data.source.warnings.length > 0 && (
            <Card className="border-amber-500/40"><CardContent className="space-y-1 p-4 text-sm text-amber-600">
              {data.source.warnings.map((warning) => <div key={warning}>{warning}</div>)}
            </CardContent></Card>
          )}

          <section className="space-y-3">
            <div className="flex items-center gap-2"><h2 className="text-xl font-semibold text-foreground">Main / COO</h2><Badge tone="outline">top layer</Badge></div>
            {mainNode ? <CrewProfileCard node={mainNode} /> : <p className="text-sm text-muted-foreground">No main profile found.</p>}
          </section>

          <section className="space-y-3">
            <div className="flex items-center gap-2"><h2 className="text-xl font-semibold text-foreground">Top Management</h2><Badge tone="outline">managers</Badge></div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {managers.map((node) => <CrewProfileCard key={node.profile.name} node={node} />)}
            </div>
          </section>

          <section className="space-y-4">
            <h2 className="text-xl font-semibold text-foreground">Team Departments</h2>
            {workerDepartments.map((department) => (
              <Card key={department.name}>
                <CardContent className="space-y-3 p-4">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div><h3 className="text-lg font-semibold text-foreground">{department.name}</h3><p className="text-xs text-muted-foreground">{department.count} profile(s)</p></div>
                    {department.managers.length > 0 && <Badge tone="secondary">Manager: {department.managers.join(", ")}</Badge>}
                  </div>
                  <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
                    {department.nodes.filter((node: CrewNode) => node.level !== "main" && node.level !== "manager").map((node: CrewNode) => <CrewProfileCard key={node.profile.name} node={node} compact />)}
                  </div>
                </CardContent>
              </Card>
            ))}
          </section>

          <section className="space-y-3">
            <div><h2 className="text-xl font-semibold text-foreground">Unassigned / New Profiles</h2><p className="text-sm text-muted-foreground">Classify this profile in crew metadata.</p></div>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
              {data.unassigned.map((node) => <CrewProfileCard key={node.profile.name} node={node} compact />)}
              {data.unassigned.length === 0 && <p className="text-sm text-muted-foreground">No unassigned profiles.</p>}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
