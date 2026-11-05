/**
 * Time-window presets for the Analytics page.
 *
 * A range is either a positive day count (7 / 30 / 90) or the literal `"all"`
 * for the un-windowed "All time" preset, which the backend maps to `?days=all`
 * and serves without a date cutoff.
 *
 * This module is deliberately dependency-free so the range logic can be unit
 * tested without mounting the page or mocking the API layer.
 */
export type AnalyticsRange = number | "all";

/** The wire form of the un-windowed range. */
export const ANALYTICS_ALL = "all" as const;

/** Adjacent presets rendered before the "All" button. */
export const ANALYTICS_PERIODS = [
  { label: "7d", days: 7 },
  { label: "30d", days: 30 },
  { label: "90d", days: 90 },
] as const;

/** True when `range` is the un-windowed "All time" preset. */
export function isAllRange(range: AnalyticsRange): range is "all" {
  return range === "all";
}

/**
 * Divisor for the "~N sessions/day" headline.
 *
 * A numeric range divides by its own day count. All-time has no fixed window,
 * so it divides by the number of days that actually carry activity — bounded
 * below by 1 so an empty dataset can never produce a divide-by-zero, which
 * would render as `Infinity` or `NaN`.
 */
export function analyticsAvgDivisor(range: AnalyticsRange, activeDays: number): number {
  if (!isAllRange(range)) return range;
  const days = Number.isFinite(activeDays) ? Math.floor(activeDays) : 0;
  return Math.max(1, days);
}
