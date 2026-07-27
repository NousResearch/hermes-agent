/**
 * Log-line level classification for the dashboard Logs page.
 *
 * Prefers the structured level token emitted by hermes_logging
 * ("2026-07-26 13:07:45,228 INFO …"); falls back to word-boundary matching
 * for untimestamped lines (tracebacks, wrapped continuations). Plain
 * substring matching is deliberately avoided — INFO lines carrying payloads
 * like "parse_errors=0" or paths like "errors.log" must not render red.
 */

export type LogLevel = "error" | "warning" | "info" | "debug";

// Level token as emitted by hermes_logging, anchored to the line head so
// payload text can't spoof the level.
const LEVEL_TOKEN_RE =
  /^\d{4}-\d{2}-\d{2}[ T][\d:,.]+\s+(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL|FATAL)\b/;

export function classifyLine(line: string): LogLevel {
  const token = LEVEL_TOKEN_RE.exec(line)?.[1];
  if (token) {
    if (token === "ERROR" || token === "CRITICAL" || token === "FATAL")
      return "error";
    if (token === "WARNING" || token === "WARN") return "warning";
    if (token === "DEBUG") return "debug";
    return "info";
  }
  const upper = line.toUpperCase();
  if (
    /\b(ERROR|CRITICAL|FATAL)\b/.test(upper) ||
    upper.startsWith("TRACEBACK (")
  )
    return "error";
  if (/\b(WARNING|WARN)\b/.test(upper)) return "warning";
  if (/\bDEBUG\b/.test(upper)) return "debug";
  return "info";
}
