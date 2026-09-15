/** Dashboard-only subagent roster. Fed by /api/events — never calls sidecar RPCs. */

export const DASHBOARD_SUBAGENT_EVENT_TYPES = new Set([
  "subagent.start",
  "subagent.spawn_requested",
  "subagent.progress",
  "subagent.tool",
  "subagent.thinking",
  "subagent.complete",
]);

const TERMINAL_STATUSES = new Set([
  "completed",
  "failed",
  "error",
  "timeout",
  "cancelled",
  "canceled",
  "interrupted",
]);

const MAX_TRANSCRIPT_LINES = 32;
const MAX_TRANSCRIPT_CHARS = 16 * 1024;

export interface DashboardSubagentRow {
  id: string;
  goal: string;
  status: string;
  /** Epoch milliseconds, or null when `started_at` was missing. */
  startedAt: number | null;
  transcript: string[];
}

export type DashboardSubagentRoster = Record<string, DashboardSubagentRow>;

export interface DashboardSubagentEvent {
  type?: unknown;
  payload?: unknown;
}

export function isDashboardSubagentEventType(type: unknown): type is string {
  return typeof type === "string" && DASHBOARD_SUBAGENT_EVENT_TYPES.has(type);
}

export function isDashboardSubagentTerminal(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

export function listDashboardSubagents(
  roster: DashboardSubagentRoster,
): DashboardSubagentRow[] {
  return Object.values(roster);
}

/** Gateway sends unix seconds; some clocks use ms. Values below 1e12 are seconds. */
export function normalizeStartedAtMs(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    return null;
  }
  return value < 1e12 ? value * 1000 : value;
}

export function formatDashboardSubagentElapsed(
  startedAt: number | null,
  now = Date.now(),
): string {
  if (startedAt == null) {
    return "—";
  }
  const seconds = Math.max(0, Math.floor((now - startedAt) / 1000));
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${seconds % 60}s`;
}

function asNonEmptyString(value: unknown): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function completeStatus(raw: unknown): string {
  return typeof raw === "string" && TERMINAL_STATUSES.has(raw) ? raw : "failed";
}

function liveStatus(raw: unknown, previous?: string): string {
  if (previous && TERMINAL_STATUSES.has(previous)) {
    return previous;
  }
  if (typeof raw === "string" && raw.length > 0) {
    return raw;
  }
  return previous ?? "running";
}

function transcriptFragments(payload: Record<string, unknown>): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const key of ["text", "summary", "tool_preview"] as const) {
    const line = asNonEmptyString(payload[key]);
    if (line && !seen.has(line)) {
      seen.add(line);
      out.push(line);
    }
  }
  return out;
}

function boundTranscript(lines: string[]): string[] {
  let next = lines.length > MAX_TRANSCRIPT_LINES ? lines.slice(-MAX_TRANSCRIPT_LINES) : lines;
  let chars = next.reduce((n, line) => n + line.length, 0);
  while (next.length > 1 && chars > MAX_TRANSCRIPT_CHARS) {
    chars -= next[0].length;
    next = next.slice(1);
  }
  return next;
}

export function applyDashboardSubagentEvent(
  roster: DashboardSubagentRoster,
  event: DashboardSubagentEvent,
): DashboardSubagentRoster {
  if (!isDashboardSubagentEventType(event.type)) {
    return roster;
  }
  if (!event.payload || typeof event.payload !== "object") {
    return roster;
  }

  const payload = event.payload as Record<string, unknown>;
  const id = asNonEmptyString(payload.subagent_id);
  if (!id) {
    return roster;
  }

  const prev = roster[id];
  const isComplete = event.type === "subagent.complete";
  const status = isComplete
    ? completeStatus(payload.status)
    : liveStatus(payload.status, prev?.status);

  const goal = asNonEmptyString(payload.goal) ?? prev?.goal ?? id;
  const startedAt =
    normalizeStartedAtMs(payload.started_at) ?? prev?.startedAt ?? null;

  const additions = transcriptFragments(payload);
  const transcript = boundTranscript([...(prev?.transcript ?? []), ...additions]);

  return {
    ...roster,
    [id]: { id, goal, status, startedAt, transcript },
  };
}
