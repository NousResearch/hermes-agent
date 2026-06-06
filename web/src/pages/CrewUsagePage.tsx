import { useCallback, useEffect, useState } from "react";
import { RefreshCw } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { api } from "@/lib/api";
import type { CrewUsageResponse } from "@/types/crew";

const PERIODS: { label: string; days: number }[] = [
  { label: "Today", days: 0 },
  { label: "7 days", days: 7 },
  { label: "30 days", days: 30 },
  { label: "All time", days: -1 },
];

function fmtInt(n: number) {
  return (n ?? 0).toLocaleString();
}

function fmtCost(n: number) {
  return `~$${(n ?? 0).toFixed(2)}`;
}

function fmtTime(ts: number | null) {
  return ts ? new Date(ts * 1000).toLocaleString() : "\u2014";
}

function SummaryCard({
  label,
  value,
}: {
  label: string;
  value: number | string;
}) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">
          {label}
        </div>
        <div className="mt-1 text-2xl font-semibold text-foreground">
          {value}
        </div>
      </CardContent>
    </Card>
  );
}

export default function CrewUsagePage() {
  const [data, setData] = useState<CrewUsageResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [days, setDays] = useState(30);

  const load = useCallback((d: number) => {
    setLoading(true);
    setError(null);
    api
      .getCrewUsage(d)
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load(days);
  }, [load, days]);

  return (
    <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 p-6">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-3xl font-semibold text-foreground">
            Crew Usage / Token Monitor
          </h1>
          <p className="text-sm text-muted-foreground">
            Read-only per-profile token usage. Token counts are authoritative;
            cost is a best-effort estimate.
          </p>
          {data && (
            <p className="mt-1 text-xs text-muted-foreground">
              Last refreshed: {data.generated_at}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <select
            className="rounded-md border bg-background p-2 text-sm"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
          >
            {PERIODS.map((p) => (
              <option key={p.days} value={p.days}>
                {p.label}
              </option>
            ))}
          </select>
          <Button onClick={() => load(days)} disabled={loading} outlined>
            <RefreshCw className="mr-2 h-4 w-4" /> Refresh
          </Button>
        </div>
      </div>

      {loading && (
        <div className="flex items-center gap-2 text-muted-foreground">
          <Spinner /> Loading usage...
        </div>
      )}
      {error && (
        <Card className="border-red-500/40">
          <CardContent className="p-4 text-sm text-red-500">
            {error}
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
            <SummaryCard
              label="Sessions"
              value={fmtInt(data.totals.sessions)}
            />
            <SummaryCard
              label="Input tokens"
              value={fmtInt(data.totals.input_tokens)}
            />
            <SummaryCard
              label="Output tokens"
              value={fmtInt(data.totals.output_tokens)}
            />
            <SummaryCard
              label="Total tokens"
              value={fmtInt(data.totals.total_tokens)}
            />
            <SummaryCard
              label="Est. cost"
              value={fmtCost(data.totals.estimated_cost_usd)}
            />
          </div>

          {data.departments.map((dept) => (
            <Card key={dept.department}>
              <CardContent className="space-y-3 p-4">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <h2 className="text-xl font-semibold text-foreground">
                    {dept.department}
                  </h2>
                  <div className="text-xs text-muted-foreground">
                    {fmtInt(dept.total_tokens)} tokens ·{" "}
                    {fmtCost(dept.estimated_cost_usd)}
                  </div>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="text-left text-xs uppercase text-muted-foreground">
                      <tr>
                        <th className="p-2">Profile</th>
                        <th className="p-2">Model / Provider</th>
                        <th className="p-2">Run / Blocked</th>
                        <th className="p-2">Input</th>
                        <th className="p-2">Output</th>
                        <th className="p-2">Cache R/W</th>
                        <th className="p-2">Reasoning</th>
                        <th className="p-2">Total</th>
                        <th className="p-2">Est. cost</th>
                        <th className="p-2">Last active</th>
                      </tr>
                    </thead>
                    <tbody>
                      {dept.profiles.map((p) => (
                        <tr
                          key={p.profile_name}
                          className="border-t border-border/40"
                        >
                          <td className="p-2 font-medium">
                            {p.display_name}
                            <div className="text-xs text-muted-foreground">
                              {p.profile_name}
                            </div>
                          </td>
                          <td className="p-2">
                            {p.total.model ?? "\u2014"}
                            <div className="text-xs text-muted-foreground">
                              {p.total.provider ?? "\u2014"}
                            </div>
                          </td>
                          <td className="p-2">
                            {p.tasks.running} / {p.tasks.blocked}
                          </td>
                          <td className="p-2">
                            {fmtInt(p.total.input_tokens)}
                          </td>
                          <td className="p-2">
                            {fmtInt(p.total.output_tokens)}
                          </td>
                          <td className="p-2">
                            {fmtInt(p.total.cache_read_tokens)}/
                            {fmtInt(p.total.cache_write_tokens)}
                          </td>
                          <td className="p-2">
                            {fmtInt(p.total.reasoning_tokens)}
                          </td>
                          <td className="p-2 font-semibold">
                            {fmtInt(p.total.total_tokens)}
                          </td>
                          <td className="p-2">
                            {fmtCost(p.total.estimated_cost_usd)}
                          </td>
                          <td className="p-2 text-xs">
                            {fmtTime(p.total.last_active)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          ))}
        </>
      )}
    </div>
  );
}
