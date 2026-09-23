/**
 * Minimal UI projections for the chat renderer.
 *
 * These are *deliberately small* — they describe what the React components
 * in `components/chat/` actually render, sourced from the gateway contract
 * (`@hermes/shared`) and the REST session payload (`@/lib/api`).
 *
 * They are NOT a re-statement of the wire schema — that's the contract's job.
 * Adding a field here means a component consumes it; if a renderer doesn't
 * need a field, it doesn't go here.
 */

import type {
  MessageCompletePayload,
  StreamDeltaPayload,
  ToolCompletePayload,
  ToolStartPayload,
  TranscriptMessage,
} from "@hermes/shared";

import type { SessionMessage } from "@/lib/api";

/* -------------------------------------------------------------------------- */
/* Roles                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * Roles the chat bubbles recognise. Both REST and gateway rows surface these;
 * anything else falls through to "system" in the existing styling.
 */
export type ChatRole = "user" | "assistant" | "system" | "tool";

/* -------------------------------------------------------------------------- */
/* Attachments                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Rendered projection for an attachment queued onto the next turn.
 *
 * The backend attaches images via `image.attach` / `image.attach_bytes`
 * (separate JSON-RPC methods) — they ride the next `prompt.submit` as
 * queued attachments, NOT as a field on `PromptSubmitParams`. This shape is
 * for the local previews we render; the wire does not carry `dataUri` /
 * `width` / `height` together (it returns `AttachedImageResult` separately).
 *
 * `kind` discriminates image vs PDF so the chip renderer can pick the
 * correct icon / preview. PDFs NEVER carry `dataUri` (no thumbnail —
 * keeping the chip text-only is intentional so the renderer doesn't
 * waste memory on large PDFs). Image chips may include a tiny data URI
 * thumbnail for inline preview.
 */
export interface ChatAttachment {
  /** Stable id for React keys and queued-attachment tracking. */
  id: string;
  /** Attachment kind — drives chip rendering. */
  kind: "image" | "pdf";
  /** Filename hint (or paste label). */
  name?: string;
  /** MIME / extension hint (e.g. `"image/png"`, `"application/pdf"`). */
  mime?: string;
  /** Host path on the gateway box, when known. */
  path?: string;
  /** Inline data URI for thumbnail rendering. Optional — large pastes / PDFs skip it. */
  dataUri?: string;
  /** Reported dimensions, when the backend measured them. Images only. */
  width?: number;
  height?: number;
  /** Token / size estimate surfaced in the chip. */
  tokenEstimate?: number;
  bytes?: number;
  /** PDF only: page count the gateway attached. */
  pageCount?: number;
}

/* -------------------------------------------------------------------------- */
/* Tool calls                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Projected tool-call record used by `ToolCallCard`. Both shapes source it:
 *
 * - REST `SessionMessage.tool_calls`: `{ id, function: { name, arguments } }`
 *   (matches the existing ChatPage / SessionsPage styling contract).
 * - Gateway `TranscriptMessage.args`: already-decoded object, rendered as
 *   pretty JSON.
 *
 * `argumentsRaw` is the raw string form (REST path); `argumentsDecoded` is
 * the object form (gateway path). The card prefers the decoded form when
 * present and falls back to pretty-printing the raw string.
 */
export interface ChatToolCall {
  id: string;
  name: string;
  /** Raw JSON string (REST path). */
  argumentsRaw?: string;
  /** Pre-decoded object (gateway path). */
  argumentsDecoded?: Record<string, unknown>;
}

/* -------------------------------------------------------------------------- */
/* Transcript                                                                  */
/* -------------------------------------------------------------------------- */

/**
 * Renderable transcript row — the union the chat renderer is happy to walk.
 *
 * The existing `SessionsPage` source is REST (`SessionMessage`); Phase 2 will
 * feed `TranscriptMessage` from the gateway. Both pass through unchanged into
 * the same `MessageBubble`, so this union keeps a single component path.
 */
export type TranscriptRow =
  | (TranscriptMessage & { source: "gateway" })
  | (SessionMessage & { source: "rest" });

/* -------------------------------------------------------------------------- */
/* Streaming assistant state                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Live projection for an in-flight assistant turn — the bubble the renderer
 * appends to while `message.delta` events accumulate, then resolves on
 * `message.final` / `assistant.message` / a turn-end event.
 *
 * This is the minimal state `MessageBubble` needs to render a streaming
 * reply without reaching into the gateway transport. The actual delta
 * reducer that builds it lives one step closer to `gatewayClient`.
 */
export interface StreamingAssistantState {
  sessionId: string;
  /** Accumulated text so far; deltas append here. */
  text: string;
  /** True between `message.delta` start and the matching terminal event. */
  streaming: boolean;
  /** First delta timestamp (ms) — used for the bubble's "started" badge. */
  startedAt?: number;
  /** Final timestamp once the turn resolves. */
  finalizedAt?: number;
  /** Optional companion reasoning stream (contract `reasoning` sidecar). */
  reasoning?: string;
}

/* -------------------------------------------------------------------------- */
/* Tool-call state                                                             */
/* -------------------------------------------------------------------------- */

/**
 * Live projection for an in-flight tool call — the card the renderer shows
 * while arguments stream in and the result arrives.
 *
 * Mirrors the live `ChatToolCall` plus an execution status so the card can
 * show a spinner / error / collapsed result without a second source of truth.
 */
export interface ChatToolCallState extends ChatToolCall {
  /** Execution lifecycle of the tool call. */
  status: "pending" | "running" | "success" | "error";
  /** Tool result text (when the backend forwards it on the same event). */
  result?: string;
  /** Error message if the tool call failed. */
  error?: string;
}

/* ========================================================================== */
/* Chat UI store — used only by the new structured /chat-ui renderer.         */
/*                                                                            */
/* The existing CLI ChatPage does not consume this — it stays on PTY bytes.   */
/* Every shape here is derived from the gateway contract, not invented.       */
/* ========================================================================== */

/**
 * Tool-call progress tail — incremental text the gateway streams while a
 * tool is executing (e.g. partial shell output for `bash`/`process`).
 *
 * The brief calls this a `tool.progress` field; the actual wire event is
 * `tool.complete.result_text` arriving with the terminal frame. The renderer
 * still exposes a `progress` projection so the card can show a "running"
 * tail while arguments stream in via `tool.start`/`tool.generating`.
 */
export interface ChatToolProgress {
  /** Last line(s) of streamed output, trimmed to a tail. */
  tail?: string;
  /** Wall-clock time the renderer last updated the tail. */
  updatedAt?: number;
}

/**
 * Renderable tool card as projected from gateway events for the new
 * Chat UI. Distinct from `ChatToolCallState` (CLI/Sessions path) so the
 * structured renderer can show running / pending / error without leaking
 * REST-shape fields.
 */
export interface ChatUIToolCall extends ChatToolCall {
  /** Stable id from `tool.start.tool_id` — used for progress updates. */
  toolId: string;
  /** Lifecycle status from gateway events. */
  status: "pending" | "running" | "success" | "error";
  /** Live progress tail (kept short; trimmed to the last few lines). */
  progress?: ChatToolProgress;
  /** Result text (from `tool.complete.result_text`/`summary`). */
  result?: string;
  /** Error message when the tool call failed. */
  error?: string;
  /** Wall-clock duration once the tool completes. */
  durationS?: number;
  /** First-emitted timestamp (ms). */
  startedAt?: number;
  /** Completion timestamp (ms), when known. */
  completedAt?: number;
}

/**
 * Renderable assistant turn as projected from gateway events.
 *
 * One per assistant reply in the visible window. `text` accumulates
 * `message.delta`s; `reasoning` accumulates `reasoning.delta`s; `tools`
 * maps gateway `tool.*` events onto cards (keyed by `tool_id`).
 */
export interface ChatUIAssistantTurn {
  /** Stable id (uses `row_id` when the gateway supplies one, else synthetic). */
  id: string;
  role: "assistant";
  /** Accumulated streaming text. */
  text: string;
  /** Accumulated reasoning companion. */
  reasoning: string;
  /** Whether the turn is still accepting deltas. */
  streaming: boolean;
  /** Tools the assistant has executed as part of this turn. */
  tools: Record<string, ChatUIToolCall>;
  /** First emission (ms). */
  startedAt: number;
  /** Finalization (ms) once `message.complete` (or error terminal) arrives. */
  finalizedAt?: number;
  /** Usage block from `message.complete.usage`, if any. */
  usage?: MessageCompletePayload["usage"];
  /** Final status string from the gateway. */
  status?: MessageCompletePayload["status"];
  /** Failure reason if the turn failed. */
  failureReason?: string;
}

/**
 * Renderable user turn — the `prompt.submit` round-trip echoed back as a
 * row the user can scroll-past. The reducer derives this from
 * `prompt.submit` and from the transcript when seeding.
 */
export interface ChatUIUserTurn {
  id: string;
  role: "user";
  text: string;
  /** Local attachments queued for this turn, when available. */
  attachments?: ChatAttachment[];
  timestamp: number;
}

/**
 * Renderable system / status row (compaction summaries, notices, todo
 * snapshots, etc.). The renderer keeps these in chronological position with
 * the rest of the turns.
 */
export interface ChatUISystemTurn {
  id: string;
  role: "system" | "tool";
  text: string;
  toolName?: string;
  toolCallId?: string;
  timestamp?: number;
}

/**
 * Discriminated union of every renderable row in the Chat UI store.
 */
export type ChatUITurn =
  | ChatUIUserTurn
  | ChatUIAssistantTurn
  | ChatUISystemTurn;

/**
 * Whole-store shape for the Chat UI reducer.
 */
export interface ChatUIStore {
  /** Active Hermes session id (`session_id` from create/resume). */
  sessionId: string | null;
  /** Stable session key (`stored_session_id`) for navigation. */
  storedSessionId: string | null;
  /** Current connection state. */
  connection: "idle" | "connecting" | "open" | "closed" | "error";
  /** Active model + reasoning level (from `session.info`). */
  model: string | null;
  provider: string | null;
  reasoningEffort: string | null;
  /** Title (from `session.title` events). */
  title: string | null;
  /** Renderable rows in chronological order. */
  turns: ChatUITurn[];
  /** Index for `session.events.since` replay / rewind. */
  lastSeenSeq: number;
  /** Transient prompt-submit busy flag — true while waiting on `message.complete`. */
  submitting: boolean;
  /** Last error message surfaced in the UI. */
  error: string | null;
  /** Last failure reason from `message.complete.failure_reason` / `error`. */
  failureReason: string | null;
  /** True once `session.resume` has produced the seeded transcript. */
  hydrated: boolean;
}

/**
 * Reducer action — a gateway event (or local action) projected to the
 * state shape the renderer cares about.
 *
 * - `event/<name>` entries come from the live GatewayClient subscription.
 * - `session.<created|resumed|hydrated>` seed/rehydrate the store.
 * - `submit/started|submit/done` track the local prompt round-trip.
 */
export type ChatUIAction =
  | { type: "session/created"; sessionId: string; storedSessionId?: string | null }
  | {
      type: "session/resumed";
      sessionId: string;
      storedSessionId?: string | null;
      title?: string | null;
      model?: string | null;
      provider?: string | null;
      reasoningEffort?: string | null;
      history: TranscriptMessage[];
      historySeq?: number;
    }
  | { type: "connection/state"; state: ChatUIStore["connection"] }
  | { type: "session/info"; title?: string | null; model?: string | null; provider?: string | null; reasoningEffort?: string | null }
  | { type: "session/title"; title: string }
  | { type: "submit/started" }
  | { type: "submit/done"; ok: boolean; failureReason?: string }
  | { type: "message/start" }
  | { type: "message/delta"; payload: StreamDeltaPayload }
  | { type: "message/interim"; payload: { text: string } }
  | { type: "message/complete"; payload: MessageCompletePayload }
  | { type: "reasoning/delta"; payload: StreamDeltaPayload }
  | { type: "reasoning/available"; payload: StreamDeltaPayload }
  | { type: "tool/start"; payload: ToolStartPayload }
  | { type: "tool/progress"; payload: { tool_id: string; tail?: string } }
  | { type: "tool/complete"; payload: ToolCompletePayload }
  | { type: "tool/generating"; payload: { name: string } }
  | { type: "user/submitted"; text: string; attachments?: ChatAttachment[] }
  | { type: "error"; message: string }
  | { type: "reset" };

/**
 * Synthetic stable id used when the gateway does not supply a `row_id`
 * (e.g. for a streaming draft assistant turn that has not finalized).
 * Backed by a monotonic counter so React keys stay stable across reducer
 * passes; never compared across reloads.
 */
export type ChatUIIdFactory = () => string;

/* -------------------------------------------------------------------------- */
/* Projection helpers                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Project a gateway `TranscriptMessage` onto the renderable row the
 * Chat UI uses. The brief allows reusing the existing MessageBubble +
 * ToolCallCard, so we map each role to the closest ChatUITurn shape.
 */
export function projectTranscriptRow(
  row: TranscriptMessage,
  fallbackId: string,
): ChatUITurn | null {
  // Strip null sentinels before string ops — the contract sometimes
  // surfaces `null` for unused optional fields.
  const text = (row.text ?? "") as string;
  const role = row.role;

  if (role === "user") {
    return {
      id: String(row.row_id ?? fallbackId),
      role: "user",
      text,
      timestamp: Number(row.timestamp ?? Date.now()),
    };
  }

  if (role === "assistant") {
    const tools: Record<string, ChatUIToolCall> = {};
    return {
      id: String(row.row_id ?? fallbackId),
      role: "assistant",
      text,
      reasoning: (row.reasoning as string | null) ?? "",
      streaming: false,
      tools,
      startedAt: Number(row.timestamp ?? Date.now()),
      finalizedAt: Number(row.timestamp ?? Date.now()),
    };
  }

  if (role === "system") {
    return {
      id: String(row.row_id ?? fallbackId),
      role: "system",
      text,
      timestamp: Number(row.timestamp ?? Date.now()),
    };
  }

  if (role === "tool") {
    return {
      id: String(row.row_id ?? fallbackId),
      role: "tool",
      text,
      toolName: (row.name as string | null) ?? undefined,
      timestamp: Number(row.timestamp ?? Date.now()),
    };
  }

  return null;
}
