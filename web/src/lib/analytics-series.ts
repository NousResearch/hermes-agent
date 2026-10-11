// Pure series shaping for the Analytics token charts (#20412).

export interface DayTokens {
  day: string;
  input_tokens: number;
  output_tokens: number;
}

export interface DayModelTokens extends DayTokens {
  model: string;
}

export interface ModelSegment {
  model: string;
  input_tokens: number;
  output_tokens: number;
}

export interface ModelStackDay {
  day: string;
  total: number;
  segments: ModelSegment[];
}

export interface ModelStack {
  /** Series in legend/stack order; OTHER_MODELS last when present. */
  models: string[];
  days: ModelStackDay[];
  max: number;
}

/** Categorical slots for the per-model stack (validated on the dashboard's dark canvas). A 7th+
 *  model folds into OTHER_MODELS rather than getting a generated hue. */
export const MODEL_SERIES_COLORS = [
  "#3987e5",
  "#d95926",
  "#199e70",
  "#c98500",
  "#d55181",
  "#9085e9",
] as const;

export const OTHER_MODELS = "__other__";

export function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

export function formatDate(day: string): string {
  const d = new Date(day + "T00:00:00");
  return Number.isNaN(d.getTime())
    ? day
    : d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

/** Input tokens per output token; null when the day produced no output (undefined ratio). */
export function ioRatio(input: number, output: number): number | null {
  return output > 0 ? input / output : null;
}

/** Token-weighted ratio for the whole period: total input over total output. */
export function periodRatio(daily: DayTokens[]): number | null {
  const input = daily.reduce((sum, d) => sum + d.input_tokens, 0);
  const output = daily.reduce((sum, d) => sum + d.output_tokens, 0);
  return ioRatio(input, output);
}

/** Stack each day's tokens by model. Models rank by period volume; the top
 *  ``MODEL_SERIES_COLORS.length`` keep their own series and the rest fold into OTHER_MODELS so a
 *  hue is never cycled. ``days`` fixes the x-axis so the stack lines up with the daily charts. */
export function stackModelsByDay(rows: DayModelTokens[], days: string[]): ModelStack {
  const volume = new Map<string, number>();
  for (const r of rows) {
    volume.set(r.model, (volume.get(r.model) ?? 0) + r.input_tokens + r.output_tokens);
  }
  const ranked = [...volume.keys()].sort(
    (a, b) => volume.get(b)! - volume.get(a)! || a.localeCompare(b),
  );
  const slots = MODEL_SERIES_COLORS.length;
  // Folding a single leftover model into "Other" would hide its name for nothing.
  const kept = ranked.length > slots ? ranked.slice(0, slots - 1) : ranked;
  const keptSet = new Set(kept);
  const models = ranked.length > kept.length ? [...kept, OTHER_MODELS] : kept;

  const byDay = new Map<string, Map<string, ModelSegment>>();
  for (const r of rows) {
    const series = keptSet.has(r.model) ? r.model : OTHER_MODELS;
    const day = byDay.get(r.day) ?? new Map<string, ModelSegment>();
    byDay.set(r.day, day);
    const seg = day.get(series) ?? { model: series, input_tokens: 0, output_tokens: 0 };
    seg.input_tokens += r.input_tokens;
    seg.output_tokens += r.output_tokens;
    day.set(series, seg);
  }

  let max = 0;
  const stackDays = days.map((day) => {
    const segs = byDay.get(day);
    const segments = models
      .map((m) => segs?.get(m))
      .filter((s): s is ModelSegment => !!s && s.input_tokens + s.output_tokens > 0);
    const total = segments.reduce((sum, s) => sum + s.input_tokens + s.output_tokens, 0);
    max = Math.max(max, total);
    return { day, total, segments };
  });
  return { models, days: stackDays, max };
}
