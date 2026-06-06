import { useCallback, useEffect, useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { CrewControlResponse, CrewHealth, CrewNode } from "@/types/crew";
import { CrewProfileCard } from "@/components/crew/CrewProfileCard";
import { CrewProfileDrawer } from "@/components/crew/CrewProfileDrawer";

function SummaryCard({ label, value }: { label: string; value: number | string }) {
  return (
    <Card><CardContent className="p-4">
      <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-foreground">{value}</div>
    </CardContent></Card>
  );
}

export default function CrewControlPage() {
  const [data, setData] = useState<CrewControlResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<CrewNode | null>(null);
  const [department, setDepartment] = useState("all");
  const [manager, setManager] = useState("all");
  const [status, setStatus] = useState("all");
  const [health, setHealth] = useState("all");
  const [missingEnv, setMissingEnv] = useState(false);
  const [missingSoul, setMissingSoul] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api.getCrewControl()
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { load(); }, [load]);

  const departments = useMemo(() => Array.from(new Set((data?.profiles ?? []).map((node) => node.department))).sort(), [data]);
  const managers = useMemo(() => Array.from(new Set((data?.profiles ?? []).map((node) => node.manager).filter(Boolean) as string[])).sort(), [data]);
  const filtered = useMemo(() => (data?.profiles ?? []).filter((node) => {
    if (department !== "all" && node.department !== department) return false;
    if (manager !== "all" && node.manager !== manager) return false;
    if (status !== "all" && node.profile.gateway_status !== status) return false;
    if (health !== "all" && node.health !== health) return false;
    if (missingEnv && node.profile.has_env) return false;
    if (missingSoul && node.profile.has_soul) return false;
    return true;
  }), [data, department, manager, status, health, missingEnv, missingSoul]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold text-foreground">Crew Control</h1>
          <p className="text-sm text-muted-foreground">Read-only operations view for Hermes profiles, config health, and runtime status.</p>
          {data && <p className="mt-1 text-xs text-muted-foreground">Last refreshed: {data.generated_at}</p>}
        </div>
        <Button onClick={load} disabled={loading} outlined>
          <RefreshCw className="mr-2 h-4 w-4" /> Refresh
        </Button>
      </div>

      {loading && <div className="flex items-center gap-2 text-muted-foreground"><Spinner /> Loading crew control…</div>}
      {error && <Card className="border-red-500/40"><CardContent className="p-4 text-sm text-red-500">{error}</CardContent></Card>}

      {data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-7">
            <SummaryCard label="Profiles" value={data.summary.total} />
            <SummaryCard label="Running" value={data.summary.running} />
            <SummaryCard label="Stopped" value={data.summary.stopped} />
            <SummaryCard label="Green" value={data.summary.green} />
            <SummaryCard label="Yellow" value={data.summary.yellow} />
            <SummaryCard label="Red/Gray" value={`${data.summary.red}/${data.summary.gray}`} />
            <SummaryCard label="Unassigned" value={data.summary.unassigned} />
          </div>

          <Card><CardContent className="grid gap-3 p-4 md:grid-cols-4 xl:grid-cols-6">
            <label className="space-y-1 text-sm"><span className="text-muted-foreground">Department</span><select className="w-full rounded-md border bg-background p-2" value={department} onChange={(event) => setDepartment(event.target.value)}><option value="all">All</option>{departments.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            <label className="space-y-1 text-sm"><span className="text-muted-foreground">Manager</span><select className="w-full rounded-md border bg-background p-2" value={manager} onChange={(event) => setManager(event.target.value)}><option value="all">All</option>{managers.map((item) => <option key={item} value={item}>{item}</option>)}</select></label>
            <label className="space-y-1 text-sm"><span className="text-muted-foreground">Gateway</span><select className="w-full rounded-md border bg-background p-2" value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All</option><option value="running">Running</option><option value="stopped">Stopped</option><option value="unknown">Unknown</option><option value="failed">Failed</option></select></label>
            <label className="space-y-1 text-sm"><span className="text-muted-foreground">Health</span><select className="w-full rounded-md border bg-background p-2" value={health} onChange={(event) => setHealth(event.target.value as CrewHealth | "all")}><option value="all">All</option><option value="green">Green</option><option value="yellow">Yellow</option><option value="red">Red</option><option value="gray">Gray</option></select></label>
            <label className="flex items-center gap-2 pt-6 text-sm"><input type="checkbox" checked={missingEnv} onChange={(event) => setMissingEnv(event.target.checked)} /> Missing .env</label>
            <label className="flex items-center gap-2 pt-6 text-sm"><input type="checkbox" checked={missingSoul} onChange={(event) => setMissingSoul(event.target.checked)} /> Missing SOUL.md</label>
          </CardContent></Card>

          <div className="flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-xl font-semibold text-foreground">Agents</h2>
            <Badge tone="outline">{filtered.length} shown</Badge>
          </div>
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
            {filtered.map((node) => <CrewProfileCard key={node.profile.name} node={node} onClick={setSelected} />)}
          </div>
          {filtered.length === 0 && <p className="text-sm text-muted-foreground">No profiles match the selected filters.</p>}
        </>
      )}

      <CrewProfileDrawer node={selected} onClose={() => setSelected(null)} />
    </div>
  );
}
