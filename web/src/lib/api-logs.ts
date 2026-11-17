// /api/logs request + response shapes for the dashboard Logs page. Kept out of
// api.ts so that facade stays under its code-health line cap.

export interface LogsResponse {
  file: string;
  lines: string[];
}

export interface LogsQuery {
  file?: string;
  lines?: number;
  level?: string;
  component?: string;
  /** Session ID substring: only lines tagged with that session (and their tracebacks). */
  session?: string;
}

export function logsQueryString(params: LogsQuery): string {
  const qs = new URLSearchParams();
  if (params.file) qs.set("file", params.file);
  if (params.lines) qs.set("lines", String(params.lines));
  if (params.level && params.level !== "ALL") qs.set("level", params.level);
  if (params.component && params.component !== "all") qs.set("component", params.component);
  if (params.session) qs.set("session", params.session);
  return qs.toString();
}
