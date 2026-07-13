/**
 * Compact a token count as K/M, promoting at the toFixed(1) rounding
 * boundary so 999,950-999,999 render as "1.0M" rather than "1000.0K".
 *
 * The unit tier is chosen from the raw value while the mantissa is shown at
 * one decimal, so a naive `n >= 1_000_000` check leaves values in
 * [999_950, 999_999] on the K branch where toFixed(1) rounds them up to the
 * impossible "1000.0K". Promoting at 999_950 keeps display and tier in sync.
 */
export function formatTokens(n: number): string {
  if (n >= 999_950) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}
