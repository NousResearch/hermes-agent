/**
 * Format a token count as a human-readable string (e.g. 1M, 128K, 4096).
 * Strips trailing ".0" for clean round numbers by default; pass
 * `{ decimals: 1 }` to force a single decimal place (see {@link formatTokens}).
 *
 * The unit tier is promoted at the `toFixed(1)` rounding boundary, so values
 * in [999_950, 999_999] render as "1.0M" rather than the impossible
 * "1000.0K" — the mantissa would otherwise round up past the K tier it was
 * chosen for.
 */
export function formatTokenCount(n: number, opts?: { decimals?: 0 | 1 }): string {
  const forced = opts?.decimals;
  if (n >= 999_950)
    return `${(n / 1_000_000).toFixed(forced ?? (n % 1_000_000 === 0 ? 0 : 1))}M`;
  if (n >= 1_000)
    return `${(n / 1_000).toFixed(forced ?? (n % 1_000 === 0 ? 0 : 1))}K`;
  return String(n);
}

/**
 * Compact a token count with a fixed single decimal (e.g. "1.0K", "1.5M"),
 * used by usage/analytics stat tiles that want a stable decimal width.
 *
 * Delegates to {@link formatTokenCount} so the K/M rounding-boundary
 * promotion lives in one place and the two formatters can never diverge on
 * thresholds — only on the trailing-zero policy.
 */
export function formatTokens(n: number): string {
  return formatTokenCount(n, { decimals: 1 });
}
