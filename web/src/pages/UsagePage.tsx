import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import {
  Activity,
  AlertCircle,
  BarChart3,
  Check,
  Clock,
  Cpu,
  RefreshCw,
  Shield,
  X,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import type { UsageQuotasResponse } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { Button } from "@nous-research/ui/ui/components/button";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Stats } from "@nous-research/ui/ui/components/stats";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import { Badge } from "@nous-research/ui/ui/components/badge";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function AuthBadge({ auth }: { auth?: { logged_in: boolean; configured: boolean } }) {
  if (!auth) return <Badge tone="secondary">Unknown</Badge>;
  if (auth.logged_in) {
    return (
      <Badge tone="success" className="gap-1">
        <Check className="h-3 w-3" />
        Authed
      </Badge>
    );
  }
  if (auth.configured) {
    return (
      <Badge tone="warning" className="gap-1">
        <Shield className="h-3 w-3" />
        Configured
      </Badge>
    );
  }
  return (
    <Badge tone="destructive" className="gap-1">
      <X className="h-3 w-3" />
      Not configured
    </Badge>
  );
}

function ProviderCard({
  title,
  provider,
  model,
  auth,
}: {
  title: string;
  provider: string;
  model: string;
  auth?: { logged_in: boolean; configured: boolean };
}) {
  return (
    <div className="flex items-center justify-between rounded-lg border border-border bg-card px-4 py-3">
      <div className="flex flex-col gap-0.5">
        <span className="text-xs text-muted-foreground">{title}</span>
        <span className="font-mono-ui text-sm">
          {provider || "—"}
          {model ? ` / ${model}` : ""}
        </span>
      </div>
      <AuthBadge auth={auth} />
    </div>
  );
}

function QuotaCard({
  provider,
  usage,
}: {
  provider: string;
  usage:
    | { available: true; title: string; plan: string | null; windows: any[]; details: string[] }
    | { available: false; reason: string };
}) {
  if (!usage.available) {
    return (
      <Card>
        <CardContent className="py-6">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <AlertCircle className="h-4 w-4" />
            <span className="font-medium">{provider}</span>
            <span>— {usage.reason}</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{usage.title}</CardTitle>
          {usage.plan && <Badge tone="outline">{usage.plan}</Badge>}
        </div>
        <div className="text-xs text-muted-foreground">{provider}</div>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        {usage.windows.map((w, i) => (
          <div key={i} className="flex flex-col gap-1.5">
            <div className="flex items-center justify-between text-sm">
              <span className="font-medium">{w.label}</span>
              <span className="text-muted-foreground">
                {w.used_percent != null ? `${Math.round(100 - w.used_percent)}% remaining` : w.detail || "—"}
              </span>
            </div>
            {w.used_percent != null && (
              <div className="h-2 w-full rounded-full bg-secondary">
                <div
                  className="h-2 rounded-full bg-primary"
                  style={{ width: `${Math.min(100, Math.max(0, w.used_percent))}%` }}
                />
              </div>
            )}
            {w.reset_at && (
              <span className="text-xs text-muted-foreground">
                Resets {new Date(w.reset_at).toLocaleString()}
              </span>
            )}
            {w.detail && w.used_percent == null && (
              <span className="text-xs text-muted-foreground">{w.detail}</span>
            )}
          </div>
        ))}
        {usage.details.length > 0 && (
          <ul className="list-disc pl-4 text-xs text-muted-foreground space-y-0.5">
            {usage.details.map((d, i) => (
              <li key={i}>{d}</li>
            ))}
          </ul>
        )}
      </CardContent>
    </Card>
  );
}

export default function UsagePage() {
  const [data, setData] = useState<UsageQuotasResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getUsageQuotas()
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, []);

  useLayoutEffect(() => {
    setAfterTitle(
      <Button
        type="button"
        ghost
        size="icon"
        className="text-muted-foreground hover:text-foreground"
        onClick={load}
        disabled={loading}
        aria-label={t.common.refresh}
      >
        {loading ? <Spinner /> : <RefreshCw />}
      </Button>,
    );
    setEnd(null);
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [loading, load, setAfterTitle, setEnd, t.common.refresh]);

  useEffect(() => {
    load();
  }, [load]);

  const usageEntries = data ? Object.entries(data.account_usage) : [];

  return (
    <div className="flex flex-col gap-6">
      {loading && !data && (
        <div className="flex items-center justify-center py-24">
          <Spinner className="text-2xl text-primary" />
        </div>
      )}

      {error && (
        <Card>
          <CardContent className="py-6">
            <p className="text-sm text-destructive text-center">{error}</p>
          </CardContent>
        </Card>
      )}

      {data && (
        <>
          {/* Provider chain */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">Provider Chain</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <ProviderCard
                  title="Primary"
                  provider={data.primary.provider}
                  model={data.primary.model}
                  auth={data.provider_auth[data.primary.provider]}
                />
                {data.fallback_chain.length > 0 && (
                  <div className="flex flex-col gap-2">
                    <span className="text-xs text-muted-foreground">Fallback chain</span>
                    {data.fallback_chain.map((fb, i) => (
                      <ProviderCard
                        key={`${fb.provider}-${fb.model}-${i}`}
                        title={`#${i + 1}`}
                        provider={fb.provider}
                        model={fb.model}
                        auth={data.provider_auth[fb.provider]}
                      />
                    ))}
                  </div>
                )}
                {data.fallback_chain.length === 0 && (
                  <p className="text-xs text-muted-foreground">No fallback providers configured.</p>
                )}
              </CardContent>
            </Card>

            {/* Current session */}
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Activity className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">Current Session</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                {data.current_session ? (
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Model</span>
                      <span className="font-mono-ui">{data.current_session.model || "—"}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Provider</span>
                      <span className="font-mono-ui">{data.current_session.billing_provider || "—"}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-muted-foreground">Started</span>
                      <span>{timeAgo(data.current_session.started_at)}</span>
                    </div>
                    <Stats
                      items={[
                        { label: "Messages", value: String(data.current_session.message_count) },
                        { label: "Tool calls", value: String(data.current_session.tool_call_count) },
                        { label: "API calls", value: String(data.current_session.api_calls) },
                        {
                          label: "Tokens",
                          value: formatTokens(
                            data.current_session.input_tokens + data.current_session.output_tokens,
                          ),
                        },
                        {
                          label: "Cache",
                          value: formatTokens(data.current_session.cache_read_tokens),
                        },
                        {
                          label: "Reasoning",
                          value: formatTokens(data.current_session.reasoning_tokens),
                        },
                      ]}
                    />
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-6">No session data available.</p>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Quotas */}
          {usageEntries.length > 0 && (
            <div className="grid gap-6 lg:grid-cols-2">
              {usageEntries.map(([provider, usage]) => (
                <QuotaCard key={provider} provider={provider} usage={usage} />
              ))}
            </div>
          )}

          {/* Usage windows */}
          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Clock className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">24h Usage</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <Stats
                  items={[
                    { label: "Sessions", value: String(data.usage_24h.sessions) },
                    { label: "Messages", value: String(data.usage_24h.messages) },
                    { label: "Tool calls", value: String(data.usage_24h.tool_calls) },
                    { label: "API calls", value: String(data.usage_24h.api_calls) },
                    {
                      label: "Tokens",
                      value: formatTokens(
                        data.usage_24h.input_tokens + data.usage_24h.output_tokens,
                      ),
                    },
                  ]}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">7d Usage</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <Stats
                  items={[
                    { label: "Sessions", value: String(data.usage_7d.sessions) },
                    { label: "Messages", value: String(data.usage_7d.messages) },
                    { label: "Tool calls", value: String(data.usage_7d.tool_calls) },
                    { label: "API calls", value: String(data.usage_7d.api_calls) },
                    {
                      label: "Tokens",
                      value: formatTokens(
                        data.usage_7d.input_tokens + data.usage_7d.output_tokens,
                      ),
                    },
                  ]}
                />
              </CardContent>
            </Card>
          </div>

          {/* Per-provider 24h */}
          {data.providers_24h.length > 0 && (
            <Card>
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Cpu className="h-5 w-5 text-muted-foreground" />
                  <CardTitle className="text-base">24h Provider Breakdown</CardTitle>
                </div>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  <table className="w-full font-mondwest normal-case text-sm">
                    <thead>
                      <tr className="border-b border-border text-muted-foreground text-xs">
                        <th className="text-left py-2 pr-4 font-medium">Provider</th>
                        <th className="text-right py-2 px-4 font-medium">Sessions</th>
                        <th className="text-right py-2 pl-4 font-medium">Tokens</th>
                      </tr>
                    </thead>
                    <tbody>
                      {data.providers_24h.map((p) => (
                        <tr
                          key={p.provider}
                          className="border-b border-border/50 hover:bg-secondary/20 transition-colors"
                        >
                          <td className="py-2 pr-4 font-mono-ui text-xs">{p.provider}</td>
                          <td className="text-right py-2 px-4 text-muted-foreground">
                            {p.sessions}
                          </td>
                          <td className="text-right py-2 pl-4">{formatTokens(p.tokens)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
