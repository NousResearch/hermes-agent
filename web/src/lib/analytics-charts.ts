import type { AnalyticsDailyModelEntry } from "@/lib/api";

/** Folded series key for models past the colored slots. Not a valid model id. */
export const OTHER_MODELS = "\u0000other";

export interface ModelStackSegment {
  key: string;
  tokens: number;
}

export interface ModelStackDay {
  day: string;
  total: number;
  segments: ModelStackSegment[];
}

export interface ModelStack {
  /** Series in fixed slot order: busiest model over the whole period first, then OTHER_MODELS if folded. */
  series: string[];
  days: ModelStackDay[];
}

const tokensOf = (r: AnalyticsDailyModelEntry) => (r.input_tokens || 0) + (r.output_tokens || 0);

/**
 * Stack each day's tokens by model. The top ``maxSeries`` models by period total keep their own
 * series (ranked once for the period, so a model keeps its color on every day); the rest fold into
 * OTHER_MODELS rather than cycling colors.
 */
export function stackDailyByModel(rows: AnalyticsDailyModelEntry[], maxSeries: number): ModelStack {
  const periodTotals = new Map<string, number>();
  for (const r of rows) periodTotals.set(r.model, (periodTotals.get(r.model) ?? 0) + tokensOf(r));

  const ranked = [...periodTotals]
    .filter(([, total]) => total > 0)
    .sort(([a, x], [b, y]) => y - x || a.localeCompare(b))
    .map(([model]) => model);
  const kept = new Set(ranked.slice(0, maxSeries));
  const series = ranked.length > maxSeries ? [...kept, OTHER_MODELS] : [...kept];

  const byDay = new Map<string, Map<string, number>>();
  for (const r of rows) {
    const key = kept.has(r.model) ? r.model : OTHER_MODELS;
    const day = byDay.get(r.day) ?? new Map<string, number>();
    day.set(key, (day.get(key) ?? 0) + tokensOf(r));
    byDay.set(r.day, day);
  }

  const days = [...byDay.keys()].sort().map((day) => {
    const totals = byDay.get(day)!;
    const segments = series
      .map((key) => ({ key, tokens: totals.get(key) ?? 0 }))
      .filter((s) => s.tokens > 0);
    return { day, total: segments.reduce((sum, s) => sum + s.tokens, 0), segments };
  });

  return { series, days };
}

/** Input tokens per output token; null when there was no output to divide by. */
export function inputOutputRatio(input: number, output: number): number | null {
  return output > 0 ? input / output : null;
}
