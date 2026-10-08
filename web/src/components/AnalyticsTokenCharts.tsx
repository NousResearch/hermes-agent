import type { ReactNode } from "react";
import { BarChart3, Cpu } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@nous-research/ui/ui/components/card";
import type { AnalyticsDailyEntry, AnalyticsDailyModelEntry } from "@/lib/api-analytics";
import {
  MODEL_SERIES_COLORS,
  OTHER_MODELS,
  formatDate,
  formatTokens,
  ioRatio,
  periodRatio,
  stackModelsByDay,
} from "@/lib/analytics-series";
import { useI18n } from "@/i18n";

const PANEL_HEIGHT_PX = 96;
const STACK_HEIGHT_PX = 160;
const OTHER_COLOR = "var(--color-muted-foreground)";

const formatRatio = (r: number | null) => (r === null ? "—" : `${r.toFixed(1)}×`);

function Tooltip({ children }: { children: ReactNode }) {
  return (
    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10 pointer-events-none">
      <div className="font-mondwest normal-case bg-card border border-border px-2.5 py-1.5 text-xs text-foreground shadow-lg whitespace-nowrap">
        {children}
      </div>
    </div>
  );
}

function DateAxis({ days }: { days: string[] }) {
  return (
    <div className="flex justify-between mt-2 font-mondwest normal-case text-xs text-text-tertiary">
      <span>{days.length > 0 ? formatDate(days[0]) : ""}</span>
      {days.length > 2 && <span>{formatDate(days[Math.floor(days.length / 2)])}</span>}
      <span>{days.length > 1 ? formatDate(days[days.length - 1]) : ""}</span>
    </div>
  );
}

interface PanelProps {
  title: string;
  /** Headline for the period, direct-labelled beside the title. */
  summary: string;
  color: string;
  daily: AnalyticsDailyEntry[];
  value: (d: AnalyticsDailyEntry) => number | null;
  format: (v: number | null) => string;
  /** Dashed reference line (e.g. the period-average ratio). */
  reference?: number | null;
}

/** One measure per panel on its own y-scale: input dwarfs output by an order of magnitude, so a
 *  shared axis flattens output to a sliver (#20412). */
function DailyPanel({ title, summary, color, daily, value, format, reference }: PanelProps) {
  const { t } = useI18n();
  const values = daily.map(value);
  const max = Math.max(...values.map((v) => v ?? 0), reference ?? 0, Number.MIN_VALUE);

  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 mb-2 font-mondwest normal-case text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          <span className="h-2.5 w-2.5" style={{ backgroundColor: color }} />
          {title}
        </span>
        <span className="text-foreground">{summary}</span>
      </div>
      <div className="relative flex items-end gap-[2px]" style={{ height: PANEL_HEIGHT_PX }}>
        {reference != null && (
          <div
            className="absolute inset-x-0 border-t border-dashed border-muted-foreground/60 pointer-events-none"
            style={{ bottom: (reference / max) * PANEL_HEIGHT_PX }}
          />
        )}
        {daily.map((d, i) => {
          const v = values[i];
          const h = v ? Math.max(Math.round((v / max) * PANEL_HEIGHT_PX), 1) : 0;
          return (
            <div key={d.day} className="flex-1 min-w-0 h-full group relative flex flex-col justify-end">
              <Tooltip>
                <div className="font-medium">{formatDate(d.day)}</div>
                <div>{t.analytics.input}: {formatTokens(d.input_tokens)}</div>
                <div>{t.analytics.output}: {formatTokens(d.output_tokens)}</div>
                <div>{t.analytics.ioRatio}: {formatRatio(ioRatio(d.input_tokens, d.output_tokens))}</div>
              </Tooltip>
              <div
                className="w-full rounded-t-[2px]"
                style={{ height: h, backgroundColor: `color-mix(in srgb, ${color} 75%, transparent)` }}
                aria-label={`${formatDate(d.day)}: ${format(v)}`}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function TokenSplitCharts({ daily }: { daily: AnalyticsDailyEntry[] }) {
  const { t } = useI18n();
  if (daily.length === 0) return null;

  const totalIn = daily.reduce((s, d) => s + d.input_tokens, 0);
  const totalOut = daily.reduce((s, d) => s + d.output_tokens, 0);
  const avg = periodRatio(daily);
  const tokens = (v: number | null) => formatTokens(v ?? 0);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-base">{t.analytics.dailyTokenUsage}</CardTitle>
        </div>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <DailyPanel
          title={t.analytics.inputPerDay}
          summary={formatTokens(totalIn)}
          color="var(--series-input-token)"
          daily={daily}
          value={(d) => d.input_tokens}
          format={tokens}
        />
        <DailyPanel
          title={t.analytics.outputPerDay}
          summary={formatTokens(totalOut)}
          color="var(--series-output-token)"
          daily={daily}
          value={(d) => d.output_tokens}
          format={tokens}
        />
        <DailyPanel
          title={t.analytics.ioRatio}
          summary={t.analytics.ratioAverage.replace("{ratio}", formatRatio(avg))}
          color="var(--color-muted-foreground)"
          daily={daily}
          value={(d) => ioRatio(d.input_tokens, d.output_tokens)}
          format={formatRatio}
          reference={avg}
        />
        <DateAxis days={daily.map((d) => d.day)} />
      </CardContent>
    </Card>
  );
}

export function ModelStackChart({
  daily,
  dailyByModel,
}: {
  daily: AnalyticsDailyEntry[];
  dailyByModel: AnalyticsDailyModelEntry[];
}) {
  const { t } = useI18n();
  if (dailyByModel.length === 0) return null;

  const stack = stackModelsByDay(dailyByModel, daily.map((d) => d.day));
  const colorOf = (model: string) =>
    model === OTHER_MODELS ? OTHER_COLOR : MODEL_SERIES_COLORS[stack.models.indexOf(model)];
  const nameOf = (model: string) => (model === OTHER_MODELS ? t.analytics.otherModels : model);
  const max = Math.max(stack.max, 1);

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center gap-2">
          <Cpu className="h-5 w-5 text-muted-foreground" />
          <CardTitle className="text-base">{t.analytics.tokensByModel}</CardTitle>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 font-mondwest normal-case text-xs text-muted-foreground">
          {stack.models.map((m) => (
            <div key={m} className="flex items-center gap-1.5 min-w-0">
              <span className="h-2.5 w-2.5 shrink-0" style={{ backgroundColor: colorOf(m) }} />
              <span className="truncate font-mono-ui">{nameOf(m)}</span>
            </div>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex items-end gap-[2px]" style={{ height: STACK_HEIGHT_PX }}>
          {stack.days.map((d) => (
            <div key={d.day} className="flex-1 min-w-0 h-full group relative flex flex-col justify-end">
              {d.total > 0 && (
                <Tooltip>
                  <div className="font-medium">
                    {formatDate(d.day)} · {formatTokens(d.total)}
                  </div>
                  {[...d.segments].reverse().map((s) => (
                    <div key={s.model} className="flex items-center gap-1.5">
                      <span className="h-2 w-2 shrink-0" style={{ backgroundColor: colorOf(s.model) }} />
                      {nameOf(s.model)}:{" "}
                      {t.analytics.inOut
                        .replace("{input}", formatTokens(s.input_tokens))
                        .replace("{output}", formatTokens(s.output_tokens))}
                    </div>
                  ))}
                </Tooltip>
              )}
              {/* column-reverse: the largest series sits on the baseline. */}
              <div className="flex flex-col-reverse gap-[2px]">
                {d.segments.map((s, i) => (
                  <div
                    key={s.model}
                    className={i === d.segments.length - 1 ? "w-full rounded-t-[2px]" : "w-full"}
                    style={{
                      height: Math.max(
                        Math.round(((s.input_tokens + s.output_tokens) / max) * STACK_HEIGHT_PX),
                        1,
                      ),
                      backgroundColor: colorOf(s.model),
                    }}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
        <DateAxis days={stack.days.map((d) => d.day)} />
      </CardContent>
    </Card>
  );
}
