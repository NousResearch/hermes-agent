import { useCallback, useEffect, useLayoutEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  FileJson,
  Gauge,
  RefreshCw,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import { api } from "@/lib/api";
import type { SecureScoreDomainScore, SecureScoreResponse } from "@/lib/api";
import { isoTimeAgo, timeAgo } from "@/lib/utils";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { usePageHeader } from "@/contexts/usePageHeader";

function riskTone(risk: string | null | undefined): "success" | "warning" | "destructive" | "secondary" {
  const value = (risk ?? "").toLowerCase();
  if (["excellent", "good", "low"].some((x) => value.includes(x))) return "success";
  if (["fair", "medium", "moderate"].some((x) => value.includes(x))) return "warning";
  if (["poor", "high", "critical"].some((x) => value.includes(x))) return "destructive";
  return "secondary";
}

function scoreColor(score: number | null | undefined): string {
  if (typeof score !== "number") return "text-muted-foreground";
  if (score >= 75) return "text-success";
  if (score >= 50) return "text-warning";
  return "text-destructive";
}

function pctColor(value: number): string {
  if (value >= 75) return "bg-success";
  if (value >= 50) return "bg-warning";
  return "bg-destructive";
}

function DomainRow({ domain }: { domain: SecureScoreDomainScore }) {
  const pct = Math.max(0, Math.min(100, domain.percentage ?? 0));
  return (
    <div className="flex flex-col gap-1.5 border border-border bg-background/40 px-3 py-2.5">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium capitalize">{domain.domain}</span>
        <span className="ml-auto font-mono text-xs text-muted-foreground">
          {domain.score}/{domain.max_score} · {pct.toFixed(0)}%
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div className={`h-full ${pctColor(pct)}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export default function SecureScorePage() {
  const [data, setData] = useState<SecureScoreResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setAfterTitle, setEnd, setTitle } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getSecureScoreAnalytics()
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useLayoutEffect(() => {
    setTitle("SecureScore");
    setAfterTitle(
      <Button
        type="button"
        ghost
        size="icon"
        className="text-muted-foreground hover:text-foreground"
        onClick={load}
        disabled={loading}
        aria-label="Refresh"
      >
        {loading ? <Spinner /> : <RefreshCw />}
      </Button>,
    );
    setEnd(null);
    return () => {
      setTitle(null);
      setAfterTitle(null);
      setEnd(null);
    };
  }, [loading, load, setAfterTitle, setEnd, setTitle]);

  const findingTotal = useMemo(
    () => Object.values(data?.finding_counts ?? {}).reduce((sum, count) => sum + count, 0),
    [data?.finding_counts],
  );

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center py-24">
        <Spinner className="text-2xl text-primary" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 sm:p-6">
        <Card>
          <CardContent className="py-8 text-center text-sm text-destructive">
            Failed to load SecureScore: {error}
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8 p-4 sm:p-6">
      {data && (data.stale || data.missing) && (
        <Card>
          <CardContent className="flex items-start gap-3 py-4">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
            <div className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-warning">
                {data.missing ? "No SecureScore snapshot found" : "SecureScore snapshot is stale"}
              </span>
              <span className="text-muted-foreground">
                {data.message ??
                  (data.timestamp
                    ? `Last published ${timeAgo(data.timestamp)}. Re-run the report script to refresh.`
                    : "Run ~/.hermes/scripts/securescore-report.sh to populate the snapshot.")}
              </span>
              <span className="font-mono text-xs text-muted-foreground">
                ~/.hermes/scripts/securescore-report.sh
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      <section className="grid gap-3 md:grid-cols-4">
        <Card className="md:col-span-2">
          <CardContent className="flex items-center gap-4 py-6">
            <div className="flex h-14 w-14 items-center justify-center rounded-full border border-border bg-background/60">
              <Gauge className="h-7 w-7 text-primary" />
            </div>
            <div className="flex min-w-0 flex-col">
              <span className="text-xs uppercase tracking-wider text-muted-foreground">
                Overall score
              </span>
              <div className="flex items-baseline gap-3">
                <span className={`text-4xl font-bold ${scoreColor(data?.overall_score)}`}>
                  {typeof data?.overall_score === "number" ? data.overall_score.toFixed(1) : "—"}
                </span>
                <span className="text-sm text-muted-foreground">/ 100</span>
              </div>
              <span className="truncate text-sm text-muted-foreground">
                {data?.deployment_name ?? "SecureScore report"}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col gap-2 py-6">
            <span className="text-xs uppercase tracking-wider text-muted-foreground">Risk</span>
            <Badge tone={riskTone(data?.risk_band)} className="w-fit text-sm">
              {data?.risk_band ?? "Unknown"}
            </Badge>
            <span className="text-xs text-muted-foreground">
              {data?.maturity_level ?? "No maturity level"}
            </span>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="flex flex-col gap-2 py-6">
            <span className="text-xs uppercase tracking-wider text-muted-foreground">Findings</span>
            <span className="text-3xl font-bold text-foreground">{findingTotal || "—"}</span>
            <span className="text-xs text-muted-foreground">
              pass {data?.finding_counts.pass ?? 0} · partial {data?.finding_counts.partial ?? 0} · fail {data?.finding_counts.fail ?? 0}
            </span>
          </CardContent>
        </Card>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1fr_1fr]">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <ShieldCheck className="h-4 w-4 text-primary" /> Domains
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {(data?.domain_scores ?? []).length === 0 ? (
              <div className="py-6 text-center text-sm text-muted-foreground">
                No domain scores in the latest snapshot.
              </div>
            ) : (
              data!.domain_scores.map((domain) => (
                <DomainRow key={domain.domain} domain={domain} />
              ))
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-base">
              <Sparkles className="h-4 w-4 text-primary" /> Top recommendations
            </CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-2">
            {(data?.recommendations ?? []).length === 0 ? (
              <div className="py-6 text-center text-sm text-muted-foreground">
                No recommendations in the latest snapshot.
              </div>
            ) : (
              data!.recommendations.slice(0, 6).map((rec) => (
                <div key={`${rec.control_id ?? rec.title}-${rec.title}`} className="border border-border bg-background/40 px-3 py-2.5">
                  <div className="flex items-start gap-2">
                    <span className="min-w-0 flex-1 text-sm font-medium">{rec.title}</span>
                    {typeof rec.score_gain === "number" && (
                      <Badge tone="secondary">+{rec.score_gain.toFixed(1)}</Badge>
                    )}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                    {rec.control_id && <span className="font-mono">{rec.control_id}</span>}
                    {rec.domain && <span className="capitalize">{rec.domain}</span>}
                    {rec.severity && <span className="capitalize">{rec.severity}</span>}
                    {rec.effort && <span className="capitalize">{rec.effort}</span>}
                  </div>
                  {rec.remediation && (
                    <p className="mt-1 text-xs text-muted-foreground">{rec.remediation}</p>
                  )}
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </section>

      <Card>
        <CardContent className="flex flex-col gap-2 py-4 text-xs text-muted-foreground sm:flex-row sm:items-center">
          <FileJson className="h-4 w-4" />
          <span>
            Generated {data?.generated_at ? isoTimeAgo(data.generated_at) : "unknown"}
            {typeof data?.timestamp === "number" ? ` · snapshot ${timeAgo(data.timestamp)}` : ""}
          </span>
          {data?.report_path && (
            <span className="truncate font-mono sm:ml-auto">{data.report_path}</span>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
