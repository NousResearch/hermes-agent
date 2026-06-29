/**
 * Format a token count as a human-readable string (e.g. 1M, 128K, 4096).
 * Strips trailing ".0" for clean round numbers.
 */
export function formatTokenCount(n: number): string {
  // Round the mantissa to one decimal *before* settling on a unit, then strip a
  // trailing ".0" for whole numbers. Rounding after the unit is chosen lets a
  // value in the [999.95, 1000) band (e.g. 999_999) print as "1000.0K" instead
  // of crossing into the next unit, so promote when the rounded mantissa hits
  // 1000.
  const fmt = (value: number, unit: string): string =>
    `${value.toFixed(Number.isInteger(value) ? 0 : 1)}${unit}`;
  if (n >= 1_000_000) {
    return fmt(Math.round((n / 1_000_000) * 10) / 10, "M");
  }
  if (n >= 1_000) {
    const k = Math.round((n / 1_000) * 10) / 10;
    if (k >= 1000) return fmt(Math.round((n / 1_000_000) * 10) / 10, "M");
    return fmt(k, "K");
  }
  return String(n);
}
