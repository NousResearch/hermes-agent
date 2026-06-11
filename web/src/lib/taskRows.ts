/**
 * Task-row state for the ChatSidebar "tasks" card — types plus the pure
 * reducer helpers that turn AgentTaskRegistry snapshots into row entries.
 *
 * Snapshots flow in from two places (see ChatSidebar):
 *
 *   * a one-shot `task.list` seed over the JSON-RPC sidecar (RUNNING
 *     records only — the registry never lists terminal ones), and
 *   * live `task.started` / `task.completed` frames on the /api/events
 *     socket, whose payload is the registry snapshot itself.
 *
 * Kept dependency-free and pure on purpose: the web package has no test
 * runner today, so isolating the reducer logic here makes it trivially
 * testable the day one lands (and keeps TaskRow.tsx component-only for
 * the react-refresh lint rule).
 */

/** Wire shape of an AgentTaskRecord snapshot (action_runtime/task_registry.py). */
export interface TaskSnapshot {
  task_id?: string;
  intent?: string;
  status?: string;
  goal?: string;
  /** Display label; a single child's label IS its goal. Absent when unset. */
  label?: string;
  /** Epoch seconds (server clock). */
  started_at?: number;
  finished_at?: number | null;
  tool_count?: number;
  last_tool?: string | null;
  tools?: string[];
  error?: string;
  session_id?: string;
  trace_id?: string;
}

/** Normalized row state derived from one or more snapshots of a task. */
export interface TaskEntry {
  task_id: string;
  intent: string;
  /** "running" | "succeeded" | "failed" | "partial" | "needs_input" | "blocked" */
  status: string;
  /** label || goal || task_id — what the row shows. */
  title: string;
  /** Epoch ms; 0 when the snapshot carried no timestamp. */
  startedAtMs: number;
  finishedAtMs?: number;
  toolCount: number;
  lastTool?: string;
  error?: string;
  /** Client receipt time of the terminal frame — the linger clock. */
  doneAtMs?: number;
}

/** How long a terminal row stays visible before pruneTasks drops it. */
export const TASK_LINGER_MS = 60_000;

export function taskEntryFromSnapshot(
  snap: TaskSnapshot,
  nowMs: number,
  prev?: TaskEntry,
): TaskEntry {
  const status = snap.status || prev?.status || "running";
  return {
    task_id: snap.task_id ?? prev?.task_id ?? "",
    intent: snap.intent || prev?.intent || "task",
    status,
    title: snap.label || snap.goal || prev?.title || snap.task_id || "",
    startedAtMs: snap.started_at
      ? snap.started_at * 1000
      : (prev?.startedAtMs ?? 0),
    finishedAtMs: snap.finished_at
      ? snap.finished_at * 1000
      : prev?.finishedAtMs,
    toolCount: snap.tool_count ?? prev?.toolCount ?? 0,
    lastTool: snap.last_tool ?? prev?.lastTool,
    error: snap.error ?? prev?.error,
    // Stamp the linger clock with the *client* receipt time the first time
    // we see the task terminal — server epochs are not compared against
    // Date.now() so a skewed clock can't make rows vanish early or never.
    doneAtMs: status !== "running" ? (prev?.doneAtMs ?? nowMs) : undefined,
  };
}

/** Upsert one snapshot (task.started / task.completed payload) by task_id. */
export function upsertTask(
  prev: TaskEntry[],
  snap: TaskSnapshot,
  nowMs: number,
): TaskEntry[] {
  if (!snap.task_id) {
    return prev;
  }
  const i = prev.findIndex((t) => t.task_id === snap.task_id);
  if (i === -1) {
    return [...prev, taskEntryFromSnapshot(snap, nowMs)];
  }
  const next = prev.slice();
  next[i] = taskEntryFromSnapshot(snap, nowMs, prev[i]);
  return next;
}

/**
 * Merge the task.list seed into whatever live frames already arrived.
 * Existing entries win: the events socket opens in parallel with the
 * sidecar handshake, so a task.completed frame can legitimately beat the
 * seed response — the stale RUNNING snapshot must not resurrect the row.
 */
export function mergeSeed(
  prev: TaskEntry[],
  snaps: TaskSnapshot[],
  nowMs: number,
): TaskEntry[] {
  let next = prev;
  for (const snap of snaps) {
    if (!snap.task_id || next.some((t) => t.task_id === snap.task_id)) {
      continue;
    }
    next = [...next, taskEntryFromSnapshot(snap, nowMs)];
  }
  return next;
}

/**
 * Drop terminal rows whose linger window elapsed. Returns `prev` unchanged
 * (same reference) when nothing expired so setState callers no-op.
 */
export function pruneTasks(
  prev: TaskEntry[],
  nowMs: number,
  lingerMs: number = TASK_LINGER_MS,
): TaskEntry[] {
  const keep = prev.filter(
    (t) => t.doneAtMs === undefined || nowMs - t.doneAtMs < lingerMs,
  );
  return keep.length === prev.length ? prev : keep;
}
