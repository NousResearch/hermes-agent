import { useCallback, useEffect, useState } from "react";
import { Activity, Database, Gauge, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type {
  AnalyticsWindow,
  CostEstimateResponse,
  ProviderQuotasResponse,
  TokenTrendsResponse,
  UsageRatesResponse,
} from "@/lib/api";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@nous-research/ui/ui/components/card";

const WINDOWS: AnalyticsWindow[] = ["1h", "24h", "7d", "30d"];
const CHART_HEIGHT_PX = 140;

// Mirrors the backend hermes_token_codec.format_token_count.
function fmt(n: number | null | undefined): string {
  const v = Math.max(0, Math.trunc(Number.isFinite(n as number) ? (n as number) : 0));
  if (v < 1000) return String(v);
  if (v >= 1_000_000) {
    const s = v / 1_000_000;
    return `${(s < 10 ? s.toFixed(2) : s < 100 ? s.toFixed(1) : s.toFixed(0)).replace(/\.?0+$/, "")}M`;
  }
  const s = v / 1000;
  return `${(s < 10 ? s.toFixed(2) : s < 100 ? s.toFixed(1) : s.toFixed(0)).replace(/\.?0+$/, "")}K`;
}

function usd(v: number | null | undefined): string {
  const n = Number(v ?? 0);
  if (n < 0.01) return `$${n.toFixed(4)}`;
  if (n < 10) return `$${n.toFixed(2)}`;
  return `$${n.toFixed(1)}`;
}

function pctColor(pct: number | null | undefined): string {
  const p = pct ?? 0;
  if (p >= 90) return "var(--destructive, #ef4444)";
  if (p >= 70) return "#f59e0b";
  return "color-mix(in srgb, var(--series-input-token, #4ade80) 70%, transparent)";
}

// ── % of limit bar ──────────────────────────────────────────────────────────
function LimitBar({ label, value, pct }: { label: string; value: string; pct: number | null }) {
  const width = pct == null ? 0 : Math.min(100, pct);
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-baseline justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-foreground">
          {value}
          {pct != null && (
            <span className="ml-1.5 text-muted-foreground">({pct.toFixed(1)}%)</span>
          )}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded bg-muted/40">
        <div
          className="h-full rounded transition-all"
          style={{ width: `${width}%`, backgroundColor: pctColor(pct) }}
        />
      </div>
    </div>
  );
}

function Metric({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <span className="font-mono text-lg text-foreground">{value}</span>
      {sub && <span className="text-[11px] text-text-tertiary">{sub}</span>}
    </div>
  );
}

// ── Usage rates ───────────────────────────────────────────────────────────
function UsageRatesCard({ data }: { data: UsageRatesResponse }) {
  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2">
        <Gauge className="h-4 w-4 text-muted-foreground" />
        <CardTitle>Request &amp; token rates</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Metric label="RPM (peak)" value={String(data.rpm.peak)} sub={`now ${data.rpm.current}`} />
          <Metric label="TPM (peak)" value={fmt(data.tpm.peak)} sub={`now ${fmt(data.tpm.current)}`} />
          <Metric label="Requests / day" value={String(data.rpd)} />
          <Metric label="Tokens / day" value={fmt(data.tpd)} />
        </div>

        {data.providers.length > 0 ? (
          <div className="flex flex-col gap-4">
            {data.providers.map((p) => (
              <div key={p.provider} className="flex flex-col gap-2 rounded border border-border p-3">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-foreground">{p.display ?? p.provider}</span>
                  {p.tier && <span className="text-xs text-muted-foreground">{p.tier}</span>}
                </div>
                <LimitBar label="RPM vs limit" value={`${data.rpm.peak} / ${p.limits.rpm ?? "—"}`} pct={p.pct_of_limit.rpm} />
                <LimitBar label="TPM in vs limit" value={`${fmt(data.tpm.peak_input)} / ${p.limits.tpm_input != null ? fmt(p.limits.tpm_input) : "—"}`} pct={p.pct_of_limit.tpm_input} />
                {p.limits.rpd != null && (
                  <LimitBar label="RPD vs limit" value={`${data.rpd} / ${p.limits.rpd}`} pct={p.pct_of_limit.rpd} />
                )}
              </div>
            ))}
          </div>
        ) : (
          <p className="text-xs text-muted-foreground">No active provider in this window to compare against.</p>
        )}
      </CardContent>
    </Card>
  );
}

// ── Token trends ──────────────────────────────────────────────────────────
function TokenTrendsCard({ data }: { data: TokenTrendsResponse }) {
  const series = data.series;
  const maxT = Math.max(1, ...series.map((b) => b.input + b.output));
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-muted-foreground" />
          <CardTitle>Token trend</CardTitle>
        </div>
        <div className="flex items-center gap-3 text-xs">
          <span className="text-muted-foreground">
            cache hit{" "}
            <span className="font-mono text-foreground">
              {data.cache_hit_rate != null ? `${data.cache_hit_rate}%` : "—"}
            </span>
          </span>
          <span className="inline-flex items-center gap-1 text-muted-foreground">
            <span className="inline-block h-2 w-2" style={{ backgroundColor: "color-mix(in srgb, var(--series-input-token) 70%, transparent)" }} /> in
          </span>
          <span className="inline-flex items-center gap-1 text-muted-foreground">
            <span className="inline-block h-2 w-2" style={{ backgroundColor: "color-mix(in srgb, var(--series-output-token) 70%, transparent)" }} /> out
          </span>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {series.length === 0 ? (
          <p className="text-xs text-muted-foreground">No token activity in this window.</p>
        ) : (
          <div className="flex items-end gap-[2px]" style={{ height: CHART_HEIGHT_PX }}>
            {series.map((b) => {
              const inH = Math.round((b.input / maxT) * CHART_HEIGHT_PX);
              const outH = Math.round((b.output / maxT) * CHART_HEIGHT_PX);
              return (
                <div
                  key={b.bucket_start}
                  className="group relative flex min-w-0 flex-1 flex-col justify-end"
                  style={{ height: CHART_HEIGHT_PX }}
                >
                  <div className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-2 hidden -translate-x-1/2 group-hover:block">
                    <div className="whitespace-nowrap border border-border bg-card px-2.5 py-1.5 text-xs text-foreground shadow-lg">
                      <div>in: {fmt(b.input)}</div>
                      <div>out: {fmt(b.output)}</div>
                      <div>req: {b.requests}</div>
                      <div>cache: {b.cache_hit_rate != null ? `${b.cache_hit_rate}%` : "—"}</div>
                    </div>
                  </div>
                  <div className="w-full" style={{ backgroundColor: "color-mix(in srgb, var(--series-input-token) 70%, transparent)", height: Math.max(inH, b.input > 0 ? 1 : 0) }} />
                  <div className="w-full" style={{ backgroundColor: "color-mix(in srgb, var(--series-output-token) 70%, transparent)", height: Math.max(outH, b.output > 0 ? 1 : 0) }} />
                </div>
              );
            })}
          </div>
        )}
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Metric label="avg in / req" value={fmt(data.averages_per_request.input)} />
          <Metric label="avg out / req" value={fmt(data.averages_per_request.output)} />
          <Metric label="total in" value={fmt(data.totals.input)} />
          <Metric label="total out" value={fmt(data.totals.output)} />
        </div>
      </CardContent>
    </Card>
  );
}

// ── Cost estimate ─────────────────────────────────────────────────────────
function CostEstimateCard({ data }: { data: CostEstimateResponse }) {
  return (
    <Card>
      <CardHeader className="flex-row items-center gap-2">
        <Database className="h-4 w-4 text-muted-foreground" />
        <CardTitle>Cost estimate</CardTitle>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Metric label="Window total" value={usd(data.total_cost_usd)} />
          <Metric label="Daily (proj.)" value={usd(data.projection.daily_usd)} />
          <Metric label="Monthly (proj.)" value={usd(data.projection.monthly_usd)} />
          <Metric
            label="By tier"
            value={usd(data.cost_by_tier.input + data.cost_by_tier.output + data.cost_by_tier.cache)}
            sub={`in ${usd(data.cost_by_tier.input)} · out ${usd(data.cost_by_tier.output)} · cache ${usd(data.cost_by_tier.cache)}`}
          />
        </div>
        {data.models.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs text-muted-foreground">
                  <th className="py-1 pr-3 font-normal">Model</th>
                  <th className="py-1 pr-3 font-normal">Provider</th>
                  <th className="py-1 pr-3 text-right font-normal">In</th>
                  <th className="py-1 pr-3 text-right font-normal">Out</th>
                  <th className="py-1 text-right font-normal">Cost</th>
                </tr>
              </thead>
              <tbody>
                {data.models.map((m, i) => (
                  <tr key={`${m.model}-${i}`} className="border-t border-border/60">
                    <td className="py-1.5 pr-3 font-mono text-foreground">{m.model ?? "—"}</td>
                    <td className="py-1.5 pr-3 text-muted-foreground">{m.provider ?? "—"}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">{fmt(m.tokens.input)}</td>
                    <td className="py-1.5 pr-3 text-right font-mono">{fmt(m.tokens.output)}</td>
                    <td className="py-1.5 text-right font-mono text-foreground">
                      {usd(m.cost_usd)}
                      {m.cost_source !== "pricing" && <span className="ml-1 text-[10px] text-text-tertiary">est</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {data.has_unpriced_models && (
          <p className="text-[11px] text-text-tertiary">
            “est” = no published pricing for that model; stored session estimate used.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// ── Provider quotas ───────────────────────────────────────────────────────
function ProviderQuotasCard({ data }: { data: ProviderQuotasResponse }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Provider quota reference</CardTitle>
      </CardHeader>
      <CardContent className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-muted-foreground">
              <th className="py-1 pr-3 font-normal">Provider</th>
              <th className="py-1 pr-3 text-right font-normal">RPM</th>
              <th className="py-1 pr-3 text-right font-normal">RPD</th>
              <th className="py-1 pr-3 text-right font-normal">TPM in</th>
              <th className="py-1 pr-3 text-right font-normal">TPM out</th>
              <th className="py-1 text-left font-normal">Tier</th>
            </tr>
          </thead>
          <tbody>
            {data.data.map((q) => (
              <tr key={q.provider} className="border-t border-border/60">
                <td className="py-1.5 pr-3 text-foreground">
                  {q.source_url ? (
                    <a href={q.source_url} target="_blank" rel="noreferrer" className="underline decoration-dotted">
                      {q.display ?? q.provider}
                    </a>
                  ) : (
                    q.display ?? q.provider
                  )}
                </td>
                <td className="py-1.5 pr-3 text-right font-mono">{q.rpm ?? "—"}</td>
                <td className="py-1.5 pr-3 text-right font-mono">{q.rpd ?? "—"}</td>
                <td className="py-1.5 pr-3 text-right font-mono">{q.tpm_input != null ? fmt(q.tpm_input) : "—"}</td>
                <td className="py-1.5 pr-3 text-right font-mono">{q.tpm_output != null ? fmt(q.tpm_output) : "—"}</td>
                <td className="py-1.5 text-muted-foreground">{q.tier ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="mt-2 text-[11px] text-text-tertiary">
          Representative published limits (entry tier); real limits are account/tier specific.
        </p>
      </CardContent>
    </Card>
  );
}

// ── Container ─────────────────────────────────────────────────────────────
export default function RealtimeAnalytics() {
  const [window, setWindow] = useState<AnalyticsWindow>("24h");
  const [rates, setRates] = useState<UsageRatesResponse | null>(null);
  const [trends, setTrends] = useState<TokenTrendsResponse | null>(null);
  const [cost, setCost] = useState<CostEstimateResponse | null>(null);
  const [quotas, setQuotas] = useState<ProviderQuotasResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.getUsageRates(window),
      api.getTokenTrends(window),
      api.getCostEstimate(window),
      api.getProviderQuotas(),
    ])
      .then(([r, t, c, q]) => {
        setRates(r);
        setTrends(t);
        setCost(c);
        setQuotas(q);
      })
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [window]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="font-mondwest text-display text-base tracking-wider text-foreground">
          Real-time usage (message-level)
        </h2>
        <div className="flex flex-wrap items-center gap-1.5">
          {WINDOWS.map((w) => (
            <Button key={w} type="button" size="sm" outlined={window !== w} onClick={() => setWindow(w)}>
              {w}
            </Button>
          ))}
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
          </Button>
        </div>
      </div>

      {error && (
        <Card>
          <CardContent className="py-6">
            <p className="text-center text-sm text-destructive">{error}</p>
          </CardContent>
        </Card>
      )}

      {loading && !rates ? (
        <div className="flex items-center justify-center py-16">
          <Spinner className="text-2xl text-primary" />
        </div>
      ) : (
        <>
          {rates && <UsageRatesCard data={rates} />}
          {trends && <TokenTrendsCard data={trends} />}
          {cost && <CostEstimateCard data={cost} />}
          {quotas && <ProviderQuotasCard data={quotas} />}
        </>
      )}
    </div>
  );
}
