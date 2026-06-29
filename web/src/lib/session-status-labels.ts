export type SessionStatusKind =
  | "blocked_by_monitor"
  | "compression_tip"
  | "queued_steer"
  | "running_tool"
  | "stale_db_runtime_active";

export interface SessionStatusLabel {
  kind: SessionStatusKind;
  label: string;
  title: string;
  tone: "destructive" | "outline" | "success" | "warning";
}

interface SessionStatusEvidence {
  _lineage_root_id?: string | null;
  compression_tip_session_id?: string | null;
  id?: string | null;
  is_active?: boolean;
  last_tool_runtime_event?: unknown;
  lineage_root?: string | null;
  queued_steer_count?: number | string | null;
  status_evidence_source?: readonly string[] | null;
}

const MONITOR_BLOCKED_SOURCES = new Set([
  "blocked",
  "blocked_by_monitor",
  "monitor_blocked",
]);

function hasSource(
  sources: readonly string[] | null | undefined,
  source: string,
): boolean {
  return Array.isArray(sources) && sources.includes(source);
}

function hasMonitorBlockedSource(
  sources: readonly string[] | null | undefined,
): boolean {
  return (
    Array.isArray(sources) &&
    sources.some((source) => MONITOR_BLOCKED_SOURCES.has(source))
  );
}

function safePositiveCount(value: number | string | null | undefined): number {
  const count = typeof value === "string" ? Number(value) : value;
  return Number.isFinite(count) && Number(count) > 0 ? Number(count) : 0;
}

export function getSessionStatusLabels(
  session: SessionStatusEvidence,
): SessionStatusLabel[] {
  const labels: SessionStatusLabel[] = [];
  const sources = session.status_evidence_source;
  const queuedSteerCount = safePositiveCount(session.queued_steer_count);

  if (hasMonitorBlockedSource(sources)) {
    labels.push({
      kind: "blocked_by_monitor",
      label: "Blocked by monitor evidence",
      title: "A monitor-safe blocker signal is attached to this session.",
      tone: "destructive",
    });
  }

  if (session.last_tool_runtime_event != null) {
    labels.push({
      kind: "running_tool",
      label: "Running tool",
      title: "A value-free runtime tool event is attached to this session.",
      tone: "success",
    });
  }

  if (queuedSteerCount > 0) {
    labels.push({
      kind: "queued_steer",
      label: "Queued steer",
      title: `${queuedSteerCount} queued steer${queuedSteerCount === 1 ? "" : "s"} waiting without exposing prompt text.`,
      tone: "warning",
    });
  }

  if (
    hasSource(sources, "compression_projection") ||
    (session.compression_tip_session_id &&
      session.id &&
      session.compression_tip_session_id !== session.id) ||
    (session._lineage_root_id && session._lineage_root_id !== session.id) ||
    (session.lineage_root && session.lineage_root !== session.id)
  ) {
    labels.push({
      kind: "compression_tip",
      label: "Compression tip",
      title: "This row is projected to the live compression tip.",
      tone: "outline",
    });
  }

  if (hasSource(sources, "active_session_registry")) {
    labels.push({
      kind: "stale_db_runtime_active",
      label: "Stale DB counter, runtime active",
      title: "Runtime registry evidence says this session is active even if DB counters lag.",
      tone: "warning",
    });
  }

  return labels;
}
