/**
 * Format an elapsed duration as a compact human-readable string
 * (e.g. 234ms, 2.3s, 45s, 3m 12s). Shared by the sidebar's ToolCall and
 * TaskRow elapsed/age labels.
 */
export function fmtElapsed(ms: number): string {
  const sec = Math.max(0, ms) / 1000;
  if (sec < 1) return `${Math.round(ms)}ms`;
  if (sec < 10) return `${sec.toFixed(1)}s`;
  if (sec < 60) return `${Math.round(sec)}s`;

  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  return s ? `${m}m ${s}s` : `${m}m`;
}

/**
 * Format a token count as a human-readable string (e.g. 1M, 128K, 4096).
 * Strips trailing ".0" for clean round numbers.
 */
export function formatTokenCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n % 1_000_000 === 0 ? 0 : 1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(n % 1_000 === 0 ? 0 : 1)}K`;
  return String(n);
}
