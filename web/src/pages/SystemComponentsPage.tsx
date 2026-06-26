import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  CircleSlash,
  ExternalLink,
  RefreshCw,
  Server,
  Wallet,
  XCircle,
} from "lucide-react";
import { api } from "@/lib/api";
import type {
  SystemComponent,
  SystemComponentStatus,
  SystemComponentsResponse,
} from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { H2 } from "@nous-research/ui/ui/components/typography/h2";
import { Card, CardContent } from "@nous-research/ui/ui/components/card";
import { usePageHeader } from "@/contexts/usePageHeader";

// Status → badge tone + icon. "absent" (service not running) is deliberately
// neutral, not an error: it's the expected state for stack components a given
// install simply doesn't run.
const STATUS_TONE: Record<
  SystemComponentStatus,
  "success" | "warning" | "destructive" | "secondary"
> = {
  up: "success",
  degraded: "warning",
  down: "destructive",
  absent: "secondary",
};

function StatusIcon({ status }: { status: SystemComponentStatus }) {
  switch (status) {
    case "up":
      return <CheckCircle2 className="h-4 w-4 text-success" />;
    case "degraded":
      return <AlertTriangle className="h-4 w-4 text-warning" />;
    case "down":
      return <XCircle className="h-4 w-4 text-destructive" />;
    default:
      return <CircleSlash className="h-4 w-4 text-muted-foreground" />;
  }
}

// Cost guardrail: flag auxiliary routes that point at known-expensive
// aggregator routes (OpenRouter "Fusion", auto-routing) so a cheap-by-default
// auxiliary task doesn't silently get billed at premium rates. Display-only
// heuristic — case-insensitive substring match on provider + model.
const EXPENSIVE_ROUTE_HINTS = ["fusion", "openrouter/auto", ":online"];

function expensiveReason(
  provider: string | null | undefined,
  model: string | null | undefined,
): string | null {
  const hay = `${provider ?? ""} ${model ?? ""}`.toLowerCase();
  const hit = EXPENSIVE_ROUTE_HINTS.find((h) => hay.includes(h));
  return hit ? `Matches "${hit}" — verify this is intentional for an auxiliary route` : null;
}

function ComponentRow({ c }: { c: SystemComponent }) {
  return (
    <div className="flex items-center gap-3 border border-border bg-background/40 px-3 py-2.5">
      <StatusIcon status={c.status} />
      <div className="flex min-w-0 flex-col">
        <span className="text-sm font-medium">{c.name}</span>
        <span className="truncate font-mono text-xs text-muted-foreground">
          {c.endpoint}
        </span>
      </div>
      <div className="ml-auto flex items-center gap-2">
        {typeof c.http_status === "number" && (
          <span className="font-mono text-xs text-muted-foreground">
            HTTP {c.http_status}
          </span>
        )}
        {typeof c.latency_ms === "number" && c.status !== "absent" && (
          <span className="font-mono text-xs text-muted-foreground">
            {c.latency_ms.toFixed(0)}ms
          </span>
        )}
        {c.error && c.status !== "absent" && (
          <span
            className="max-w-[16rem] truncate text-xs text-destructive"
            title={c.error}
          >
            {c.error}
          </span>
        )}
        <Badge tone={STATUS_TONE[c.status]}>{c.status}</Badge>
        {c.admin_url && (
          <a
            href={c.admin_url}
            target="_blank"
            rel="noreferrer"
            className="text-primary hover:text-foreground"
            aria-label={`Open ${c.name} admin UI`}
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </div>
    </div>
  );
}

export default function SystemComponentsPage() {
  const [data, setData] = useState<SystemComponentsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { setAfterTitle, setEnd, setTitle } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getSystemComponentsAnalytics()
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useLayoutEffect(() => {
    // The route path ("/system-components") would otherwise derive an ugly
    // "System-components" title — override it explicitly.
    setTitle("System Components");
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
            Failed to load system components: {error}
          </CardContent>
        </Card>
      </div>
    );
  }

  const summary = data?.summary;
  const components = data?.components ?? [];
  const aux = data?.auxiliary_routes ?? null;

  const auxRows: { label: string; provider: string | null; model: string | null }[] =
    aux
      ? [
          {
            label: "Compression",
            provider: aux.compression_provider,
            model: aux.compression_model,
          },
          {
            label: "Vision",
            provider: aux.vision?.provider ?? null,
            model: aux.vision?.model ?? null,
          },
          {
            label: "Web extract",
            provider: aux.web_extract?.provider ?? null,
            model: aux.web_extract?.model ?? null,
          },
        ]
      : [];

  return (
    <div className="flex flex-col gap-8 p-4 sm:p-6">
      {/* Stale / missing snapshot warning */}
      {data && (data.stale || data.missing) && (
        <Card>
          <CardContent className="flex items-start gap-3 py-4">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning" />
            <div className="flex flex-col gap-1 text-sm">
              <span className="font-medium text-warning">
                {data.missing ? "No snapshot found" : "Snapshot is stale"}
              </span>
              <span className="text-muted-foreground">
                {data.message ??
                  (data.timestamp
                    ? `Last published ${timeAgo(data.timestamp)}. Re-run the publisher to refresh.`
                    : "Run scripts/publish_system_components_status.py to populate the snapshot.")}
              </span>
              <span className="font-mono text-xs text-muted-foreground">
                python scripts/publish_system_components_status.py
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Summary counts */}
      {summary && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          {(
            [
              ["Total", summary.total, "text-foreground"],
              ["Up", summary.up, "text-success"],
              ["Degraded", summary.degraded, "text-warning"],
              ["Down", summary.down, "text-destructive"],
              ["Absent", summary.absent, "text-muted-foreground"],
            ] as const
          ).map(([label, value, tone]) => (
            <Card key={label}>
              <CardContent className="py-3 text-center">
                <div className={`text-2xl font-bold ${tone}`}>{value}</div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">
                  {label}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Components */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Server className="h-4 w-4" /> Local AI stack
        </H2>
        {components.length === 0 ? (
          <Card>
            <CardContent className="py-6 text-center text-sm text-muted-foreground">
              No component data. Run the publisher to populate the snapshot.
            </CardContent>
          </Card>
        ) : (
          <Card>
            <CardContent className="flex flex-col gap-2 py-4">
              {components.map((c) => (
                <ComponentRow key={c.name} c={c} />
              ))}
            </CardContent>
          </Card>
        )}
        {data?.timestamp && (
          <span className="text-xs text-muted-foreground">
            Snapshot published {timeAgo(data.timestamp)}
            {data.source ? ` · source: ${data.source}` : ""}
          </span>
        )}
      </section>

      {/* Auxiliary routes / cost guardrails */}
      <section className="flex flex-col gap-3">
        <H2 variant="sm" className="flex items-center gap-2 text-muted-foreground">
          <Wallet className="h-4 w-4" /> Auxiliary routes · cost guardrails
        </H2>
        <Card>
          <CardContent className="flex flex-col gap-2 py-4">
            {!aux ? (
              <p className="text-sm text-muted-foreground">
                No auxiliary-route info in the snapshot. The publisher reads
                the <span className="font-mono">auxiliary.*</span> provider /
                model fields from <span className="font-mono">config.yaml</span>.
              </p>
            ) : (
              auxRows.map((row) => {
                const reason = expensiveReason(row.provider, row.model);
                const configured = row.provider || row.model;
                return (
                  <div
                    key={row.label}
                    className="flex items-center gap-3 border border-border bg-background/40 px-3 py-2.5"
                  >
                    <span className="w-28 shrink-0 text-sm font-medium">
                      {row.label}
                    </span>
                    <span className="truncate font-mono text-xs text-muted-foreground">
                      {configured
                        ? `${row.provider ?? "auto"} · ${row.model ?? "(auto)"}`
                        : "auto (default)"}
                    </span>
                    <div className="ml-auto">
                      {reason ? (
                        <Badge tone="warning" title={reason}>
                          expensive route
                        </Badge>
                      ) : configured ? (
                        <Badge tone="secondary">override</Badge>
                      ) : (
                        <Badge tone="outline">default</Badge>
                      )}
                    </div>
                  </div>
                );
              })
            )}
            <p className="mt-1 text-xs text-muted-foreground">
              Auxiliary tasks (context compression, vision, web extract) should
              use cheap models. Routes flagged{" "}
              <span className="text-warning">expensive route</span> match a
              premium aggregator pattern (e.g. OpenRouter Fusion) — confirm
              they're intentional.
            </p>
          </CardContent>
        </Card>
      </section>
    </div>
  );
}
