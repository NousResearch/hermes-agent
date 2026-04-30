import { useCallback, useEffect, useLayoutEffect, useState } from "react";
import {
  Brain,
  Cpu,
  DollarSign,
  Eye,
  RefreshCw,
  Wrench,
  Zap,
} from "lucide-react";
import { api } from "@/lib/api";
import type { ModelsAnalyticsModelEntry, ModelsAnalyticsResponse } from "@/lib/api";
import { timeAgo } from "@/lib/utils";
import { formatTokenCount } from "@/lib/format";
import { Button, Spinner, Stats } from "@nous-research/ui";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@nous-research/ui";
import { usePageHeader } from "@/contexts/usePageHeader";
import { useI18n } from "@/i18n";
import { PluginSlot } from "@/plugins";

const PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(n: number): string {
  if (n >= 1) return `$${n.toFixed(2)}`;
  if (n >= 0.01) return `$${n.toFixed(3)}`;
  if (n > 0) return `$${n.toFixed(4)}`;
  return "$0";
}

/** Short model name: strip vendor prefix like "openrouter/" or "anthropic/". */
function shortModelName(model: string): string {
  const slashIdx = model.indexOf("/");
  if (slashIdx > 0) return model.slice(slashIdx + 1);
  return model;
}

/** Extract vendor prefix from a model string like "anthropic/claude-opus-4.7" → "anthropic". */
function modelVendor(model: string, fallback?: string): string {
  const slashIdx = model.indexOf("/");
  if (slashIdx > 0) return model.slice(0, slashIdx);
  return fallback || "";
}

function TokenBar({
  input,
  output,
  cacheRead,
  reasoning,
}: {
  input: number;
  output: number;
  cacheRead: number;
  reasoning: number;
}) {
  const total = input + output + cacheRead + reasoning;
  if (total === 0) return null;

  const segments = [
    { value: cacheRead, color: "bg-blue-400/60", label: "Cache Read" },
    { value: reasoning, color: "bg-purple-400/60", label: "Reasoning" },
    { value: input, color: "bg-[#ffe6cb]/70", label: "Input" },
    { value: output, color: "bg-emerald-500/70", label: "Output" },
  ].filter((s) => s.value > 0);

  return (
    <div className="space-y-1">
      <div className="flex h-2 w-full overflow-hidden rounded-sm bg-muted/30">
        {segments.map((s, i) => (
          <div
            key={i}
            className={`${s.color} transition-all duration-300`}
            style={{ width: `${(s.value / total) * 100}%` }}
          />
        ))}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-[10px] text-muted-foreground">
        {segments.map((s, i) => (
          <span key={i} className="flex items-center gap-1">
            <span className={`inline-block h-1.5 w-1.5 rounded-full ${s.color}`} />
            {s.label} {formatTokens(s.value)}
          </span>
        ))}
      </div>
    </div>
  );
}

function CapabilityBadges({
  capabilities,
}: {
  capabilities: ModelsAnalyticsModelEntry["capabilities"];
}) {
  const hasAny =
    capabilities.supports_tools ||
    capabilities.supports_vision ||
    capabilities.supports_reasoning ||
    capabilities.model_family;
  if (!hasAny) return null;

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {capabilities.supports_tools && (
        <span className="inline-flex items-center gap-1 bg-emerald-500/10 px-1.5 py-0.5 text-[10px] font-medium text-emerald-600 dark:text-emerald-400">
          <Wrench className="h-2.5 w-2.5" /> Tools
        </span>
      )}
      {capabilities.supports_vision && (
        <span className="inline-flex items-center gap-1 bg-blue-500/10 px-1.5 py-0.5 text-[10px] font-medium text-blue-600 dark:text-blue-400">
          <Eye className="h-2.5 w-2.5" /> Vision
        </span>
      )}
      {capabilities.supports_reasoning && (
        <span className="inline-flex items-center gap-1 bg-purple-500/10 px-1.5 py-0.5 text-[10px] font-medium text-purple-600 dark:text-purple-400">
          <Brain className="h-2.5 w-2.5" /> Reasoning
        </span>
      )}
      {capabilities.model_family && (
        <span className="inline-flex items-center bg-muted px-1.5 py-0.5 text-[10px] font-medium text-muted-foreground">
          {capabilities.model_family}
        </span>
      )}
    </div>
  );
}

function ModelCard({
  entry,
  rank,
}: {
  entry: ModelsAnalyticsModelEntry;
  rank: number;
}) {
  const { t } = useI18n();
  const provider = entry.provider || modelVendor(entry.model);
  const totalTokens = entry.input_tokens + entry.output_tokens;
  const caps = entry.capabilities;

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0 flex-1">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground/50 text-xs font-mono">
                #{rank}
              </span>
              <CardTitle className="text-sm font-mono-ui truncate">
                {shortModelName(entry.model)}
              </CardTitle>
            </div>
            <div className="flex items-center gap-2 mt-1">
              {provider && (
                <Badge tone="secondary" className="text-[9px]">
                  {provider}
                </Badge>
              )}
              {caps.context_window && caps.context_window > 0 && (
                <span className="text-[10px] text-muted-foreground">
                  {formatTokenCount(caps.context_window)} ctx
                </span>
              )}
              {caps.max_output_tokens && caps.max_output_tokens > 0 && (
                <span className="text-[10px] text-muted-foreground">
                  {formatTokenCount(caps.max_output_tokens)} out
                </span>
              )}
            </div>
          </div>
          <div className="text-right shrink-0">
            <div className="text-xs font-mono font-semibold">
              {formatTokens(totalTokens)}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {t.models.tokens}
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-3 pt-0">
        <TokenBar
          input={entry.input_tokens}
          output={entry.output_tokens}
          cacheRead={entry.cache_read_tokens}
          reasoning={entry.reasoning_tokens}
        />

        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className="text-center">
            <div className="font-mono font-semibold">{entry.sessions}</div>
            <div className="text-[10px] text-muted-foreground">
              {t.models.sessions}
            </div>
          </div>
          <div className="text-center">
            <div className="font-mono font-semibold">
              {formatTokens(entry.avg_tokens_per_session)}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {t.models.avgPerSession}
            </div>
          </div>
          <div className="text-center">
            <div className="font-mono font-semibold">
              {entry.api_calls > 0 ? formatTokens(entry.api_calls) : "—"}
            </div>
            <div className="text-[10px] text-muted-foreground">
              {t.models.apiCalls}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-between text-[10px] text-muted-foreground border-t border-border/30 pt-2">
          <div className="flex items-center gap-3">
            {entry.estimated_cost > 0 && (
              <span className="flex items-center gap-0.5">
                <DollarSign className="h-2.5 w-2.5" />
                {formatCost(entry.estimated_cost)}
              </span>
            )}
            {entry.tool_calls > 0 && (
              <span className="flex items-center gap-0.5">
                <Zap className="h-2.5 w-2.5" />
                {entry.tool_calls} {t.models.toolCalls}
              </span>
            )}
          </div>
          {entry.last_used_at > 0 && (
            <span>{timeAgo(entry.last_used_at)}</span>
          )}
        </div>

        <CapabilityBadges capabilities={entry.capabilities} />
      </CardContent>
    </Card>
  );
}

export default function ModelsPage() {
  const [days, setDays] = useState(30);
  const [data, setData] = useState<ModelsAnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { t } = useI18n();
  const { setAfterTitle, setEnd } = usePageHeader();

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .getModelsAnalytics(days)
      .then(setData)
      .catch((err) => setError(String(err)))
      .finally(() => setLoading(false));
  }, [days]);

  useLayoutEffect(() => {
    const periodLabel =
      PERIODS.find((p) => p.days === days)?.label ?? `${days}d`;
    setAfterTitle(
      <span className="flex items-center gap-2">
        {loading && <Spinner className="shrink-0 text-base text-primary" />}
        <Badge tone="secondary" className="text-[10px]">
          {periodLabel}
        </Badge>
      </span>,
    );
    setEnd(
      <div className="flex w-full min-w-0 flex-wrap items-center justify-end gap-2 sm:gap-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {PERIODS.map((p) => (
            <Button
              key={p.label}
              type="button"
              size="sm"
              outlined={days !== p.days}
              onClick={() => setDays(p.days)}
            >
              {p.label}
            </Button>
          ))}
        </div>
        <Button
          type="button"
          size="sm"
          outlined
          onClick={load}
          disabled={loading}
          prefix={loading ? <Spinner /> : <RefreshCw />}
        >
          {t.common.refresh}
        </Button>
      </div>,
    );
    return () => {
      setAfterTitle(null);
      setEnd(null);
    };
  }, [days, loading, load, setAfterTitle, setEnd, t.common.refresh]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <div className="flex flex-col gap-6">
      <PluginSlot name="models:top" />
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
          <Card>
            <CardContent className="py-6">
              <Stats
                items={[
                  {
                    label: t.models.modelsUsed,
                    value: String(data.totals.distinct_models),
                  },
                  {
                    label: t.analytics.totalTokens,
                    value: formatTokens(
                      data.totals.total_input + data.totals.total_output,
                    ),
                  },
                  {
                    label: t.analytics.input,
                    value: formatTokens(data.totals.total_input),
                  },
                  {
                    label: t.analytics.output,
                    value: formatTokens(data.totals.total_output),
                  },
                  {
                    label: t.models.estimatedCost,
                    value: formatCost(data.totals.total_estimated_cost),
                  },
                  {
                    label: t.analytics.totalSessions,
                    value: String(data.totals.total_sessions),
                  },
                ]}
              />
            </CardContent>
          </Card>

          {data.models.length > 0 ? (
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
              {data.models.map((m, i) => (
                <ModelCard key={`${m.model}:${m.provider}`} entry={m} rank={i + 1} />
              ))}
            </div>
          ) : (
            <Card>
              <CardContent className="py-12">
                <div className="flex flex-col items-center text-muted-foreground">
                  <Cpu className="h-8 w-8 mb-3 opacity-40" />
                  <p className="text-sm font-medium">{t.models.noModelsData}</p>
                  <p className="text-xs mt-1 text-muted-foreground/60">
                    {t.models.startSession}
                  </p>
                </div>
              </CardContent>
            </Card>
          )}
        </>
      )}

      <PluginSlot name="models:bottom" />
    </div>
  );
}
