/**
 * Real-time subagent state management via nanostore.
 * 
 * Receives subagent.* events from the gateway WebSocket and maintains
 * an in-memory store of all subagents for the current session.
 */

import { atom } from "nanostores";

export type SubagentStatus =
  | "completed"
  | "error"
  | "failed"
  | "interrupted"
  | "queued"
  | "running"
  | "timeout";

export interface SubagentProgress {
  apiCalls?: number;
  costUsd?: number;
  depth: number;
  durationSeconds?: number;
  filesRead?: string[];
  filesWritten?: string[];
  goal: string;
  id: string;
  index: number;
  inputTokens?: number;
  iteration?: number;
  model?: string;
  notes: string[]; // last 6 progress notes
  outputTail?: Array<{
    isError: boolean;
    preview: string;
    tool: string;
  }>;
  outputTokens?: number;
  parentId: null | string;
  reasoningTokens?: number;
  startedAt?: number;
  status: SubagentStatus;
  summary?: string;
  taskCount: number;
  thinking: string[]; // last 6 thinking lines
  toolCount: number;
  tools: string[]; // last 8 tool calls
  toolsets?: string[];
}

export interface SubagentEventPayload {
  api_calls?: number;
  cost_usd?: number;
  depth?: number;
  duration_seconds?: number;
  files_read?: string[];
  files_written?: string[];
  goal?: string;
  input_tokens?: number;
  iteration?: number;
  model?: string;
  output_tail?: Array<{
    is_error?: boolean;
    preview?: string;
    tool?: string;
  }>;
  output_tokens?: number;
  parent_id?: null | string;
  reasoning_tokens?: number;
  status?: SubagentStatus;
  subagent_id?: string;
  summary?: string;
  task_count?: number;
  task_index?: number;
  text?: string;
  tool_count?: number;
  tool_name?: string;
  tool_preview?: string;
  toolsets?: string[];
}

/**
 * In-memory store of all subagents for the current session.
 * Keyed by subagent_id for O(1) upsert.
 */
export const subagentStore = atom<Record<string, SubagentProgress>>({});

const TERMINAL_STATUSES = new Set<SubagentStatus>([
  "completed",
  "error",
  "failed",
  "interrupted",
  "timeout",
]);

/**
 * Push unique values to the end of an array, keeping only the last `maxLen` items.
 */
function pushUnique(arr: string[], value: string, maxLen = 6): string[] {
  if (!value) return arr;
  const newArr = arr.filter((v) => v !== value);
  newArr.push(value);
  return newArr.slice(-maxLen);
}

/**
 * Upsert a subagent from a gateway event.
 * 
 * - Spawns a new entry if subagent_id is not in the store
 * - Updates existing entry if already present
 * - Respects terminal status (won't overwrite completed/failed/etc with running)
 * - Maintains capped arrays for streaming content (tools, thinking, notes)
 */
export function upsertSubagent(
  eventType: string,
  payload: SubagentEventPayload
): void {
  const id = payload.subagent_id;
  if (!id) return;

  const store = subagentStore.get();
  const existing = store[id];

  // Terminal status guard: don't overwrite completed states
  if (existing && TERMINAL_STATUSES.has(existing.status)) {
    return;
  }

  const isNew = !existing;
  
  // Determine status from event type
  let status: SubagentStatus = existing?.status ?? "running";
  if (eventType === "subagent.spawn_requested") {
    status = "queued";
  } else if (eventType === "subagent.start") {
    status = "running";
  } else if (eventType === "subagent.complete") {
    status = payload.status ?? "completed";
  } else if (eventType === "subagent.interrupt") {
    status = "interrupted";
  }

  // Extract streaming content
  const toolText = payload.tool_name
    ? `${payload.tool_name}(${payload.tool_preview ?? ""})`
    : undefined;
  const thinkingText = eventType === "subagent.thinking" ? payload.text : undefined;
  const noteText = eventType === "subagent.progress" ? payload.text : undefined;

  // Build updated subagent
  const updated: SubagentProgress = {
    id,
    goal: payload.goal ?? existing?.goal ?? "",
    status,
    depth: payload.depth ?? existing?.depth ?? 0,
    index: payload.task_index ?? existing?.index ?? 0,
    taskCount: payload.task_count ?? existing?.taskCount ?? 0,
    parentId: payload.parent_id ?? existing?.parentId ?? null,
    model: payload.model ?? existing?.model,
    startedAt: existing?.startedAt ?? (isNew ? Date.now() : undefined),
    durationSeconds: payload.duration_seconds ?? existing?.durationSeconds,
    costUsd: payload.cost_usd ?? existing?.costUsd,
    inputTokens: payload.input_tokens ?? existing?.inputTokens,
    outputTokens: payload.output_tokens ?? existing?.outputTokens,
    reasoningTokens: payload.reasoning_tokens ?? existing?.reasoningTokens,
    toolCount: payload.tool_count ?? existing?.toolCount ?? 0,
    filesRead: payload.files_read ?? existing?.filesRead,
    filesWritten: payload.files_written ?? existing?.filesWritten,
    toolsets: payload.toolsets ?? existing?.toolsets,
    iteration: payload.iteration ?? existing?.iteration,
    apiCalls: payload.api_calls ?? existing?.apiCalls,
    outputTail: payload.output_tail?.map((t) => ({
      isError: t.is_error ?? false,
      preview: t.preview ?? "",
      tool: t.tool ?? "",
    })) ?? existing?.outputTail,
    summary: payload.summary ?? existing?.summary,
    
    // Streaming arrays: push unique, cap at max length
    tools: toolText
      ? pushUnique(existing?.tools ?? [], toolText, 8)
      : existing?.tools ?? [],
    thinking: thinkingText
      ? pushUnique(existing?.thinking ?? [], thinkingText, 6)
      : existing?.thinking ?? [],
    notes: noteText
      ? pushUnique(existing?.notes ?? [], noteText, 6)
      : existing?.notes ?? [],
  };

  subagentStore.set({
    ...store,
    [id]: updated,
  });
}

/**
 * Clear all subagents (e.g., on session change).
 */
export function clearSubagents(): void {
  subagentStore.set({});
}

/**
 * Get all subagents sorted by depth then index.
 */
export function getSortedSubagents(): SubagentProgress[] {
  const store = subagentStore.get();
  return Object.values(store).sort((a, b) => 
    (a.depth - b.depth) || (a.index - b.index)
  );
}
