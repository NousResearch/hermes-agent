/**
 * Reducer for the native chat surface: one transcript made of the gateway's
 * `session.history` rows and its live notifications. Kept pure so the mapping
 * (which row an event belongs to, when a stream seals) is testable without a
 * socket. The TUI's `createGatewayEventHandler` is the reference semantics.
 */
import type { GatewayEvent, TranscriptMessage } from "@hermes/shared";

/** Events the native transcript consumes; narrowing the union keeps payloads typed. */
export type ChatEvent =
  | GatewayEvent<"message.start">
  | GatewayEvent<"message.delta">
  | GatewayEvent<"message.complete">
  | GatewayEvent<"message.adviser">
  | GatewayEvent<"reasoning.delta">
  | GatewayEvent<"tool.start">
  | GatewayEvent<"tool.complete">
  | GatewayEvent<"error">;

export type ChatRow =
  | { id: string; kind: "user"; text: string }
  | { id: string; kind: "assistant"; text: string; streaming: boolean }
  | { id: string; kind: "adviser"; text: string; severity: string; label: string }
  | {
      id: string;
      kind: "tool";
      name: string;
      args?: unknown;
      output?: string;
      status: "running" | "done";
    }
  | { id: string; kind: "reasoning"; text: string }
  | { id: string; kind: "system"; text: string; tone: "muted" | "error" };

export interface ChatState {
  rows: ChatRow[];
  /** Row that `message.delta` appends to; null between turns. */
  streamingId: string | null;
  /** Row that `reasoning.delta` appends to. */
  reasoningId: string | null;
}

export const EMPTY_CHAT: ChatState = {
  rows: [],
  streamingId: null,
  reasoningId: null,
};

const SEVERITIES = new Set(["blocker", "concern", "nit"]);

/** `adviser · concern · reviewer` / `- [concern · reviewer]` → `concern`. */
export function adviserSeverity(text: string): string {
  const match = text.match(/\[(?:adviser\s*·\s*)?([a-z]+)\s*·/i);
  if (match && SEVERITIES.has(match[1].toLowerCase())) return match[1].toLowerCase();
  const header = text.split("\n", 1)[0] ?? "";
  const token = header.match(/\b(blocker|concern|nit)\b/i);
  return token ? token[1].toLowerCase() : "advisory";
}

export function textOf(message: TranscriptMessage): string {
  if (typeof message.text === "string" && message.text) return message.text;
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") return part;
        if (part && typeof part === "object" && "text" in part) {
          return String((part as { text?: unknown }).text ?? "");
        }
        return "";
      })
      .join("");
  }
  return "";
}

function historyRow(message: TranscriptMessage, index: number): ChatRow | null {
  const text = textOf(message).trim();
  const role = String(message.role || "");
  const kind = String(message.display_kind || "");
  if (role === "user") {
    if (!text) return null;
    if (kind === "adviser") {
      return {
        id: `history-${message.row_id ?? index}`,
        kind: "adviser",
        text,
        severity: adviserSeverity(text),
        label: "adviser",
      };
    }
    return { id: `history-${message.row_id ?? index}`, kind: "user", text };
  }
  if (role === "assistant") {
    if (!text) return null;
    return {
      id: `history-${message.row_id ?? index}`,
      kind: "assistant",
      text,
      streaming: false,
    };
  }
  if (role === "tool") {
    return {
      id: `history-${message.row_id ?? index}`,
      kind: "tool",
      name: String(message.name || message.context || "tool"),
      args: message.args,
      output: text,
      status: "done",
    };
  }
  if (text && kind !== "compaction") {
    return { id: `history-${message.row_id ?? index}`, kind: "system", text, tone: "muted" };
  }
  return null;
}

export function rowsFromHistory(messages: TranscriptMessage[]): ChatRow[] {
  return messages
    .map((message, index) => historyRow(message, index))
    .filter((row): row is ChatRow => row !== null);
}

function replaceRow(state: ChatState, id: string, update: (row: ChatRow) => ChatRow): ChatState {
  let found = false;
  const rows = state.rows.map((row) => {
    if (row.id !== id) return row;
    found = true;
    return update(row);
  });
  return found ? { ...state, rows } : state;
}

export function applyEvent(state: ChatState, event: ChatEvent): ChatState {
  switch (event.type) {
    case "message.start": {
      if (state.streamingId) return state;
      const id = `live-assistant-${event.seq ?? Date.now()}`;
      return {
        ...state,
        streamingId: id,
        reasoningId: null,
        rows: [...state.rows, { id, kind: "assistant", text: "", streaming: true }],
      };
    }
    case "message.delta": {
      const text = String(event.payload?.text ?? "");
      if (!text) return state;
      if (state.streamingId) {
        return replaceRow(state, state.streamingId, (row) =>
          row.kind === "assistant" ? { ...row, text: row.text + text } : row,
        );
      }
      const id = `live-assistant-${event.seq ?? Date.now()}`;
      return {
        ...state,
        streamingId: id,
        rows: [...state.rows, { id, kind: "assistant", text, streaming: true }],
      };
    }
    case "message.complete": {
      const text = String(event.payload?.text ?? "");
      if (!state.streamingId) {
        if (!text.trim()) return state;
        return {
          ...state,
          rows: [
            ...state.rows,
            { id: `live-assistant-${event.seq ?? Date.now()}`, kind: "assistant", text, streaming: false },
          ],
        };
      }
      const id = state.streamingId;
      return {
        ...state,
        streamingId: null,
        rows: state.rows.map((row) =>
          row.id === id && row.kind === "assistant"
            ? { ...row, text: text || row.text, streaming: false }
            : row,
        ),
      };
    }
    case "message.adviser": {
      const payload = event.payload;
      if (!payload?.text || state.rows.some((row) => row.id === payload.id)) return state;
      return {
        ...state,
        rows: [
          ...state.rows,
          {
            id: payload.id,
            kind: "adviser",
            text: payload.text,
            severity: String(payload.severity || adviserSeverity(payload.text)).toLowerCase(),
            label: payload.label || "adviser",
          },
        ],
      };
    }
    case "reasoning.delta": {
      const text = String(event.payload?.text ?? "");
      if (!text) return state;
      if (state.reasoningId) {
        return replaceRow(state, state.reasoningId, (row) =>
          row.kind === "reasoning" ? { ...row, text: row.text + text } : row,
        );
      }
      const id = `reasoning-${event.seq ?? Date.now()}`;
      return {
        ...state,
        reasoningId: id,
        rows: [...state.rows, { id, kind: "reasoning", text }],
      };
    }
    case "tool.start": {
      const payload = event.payload;
      if (!payload?.tool_id) return state;
      if (state.rows.some((row) => row.id === payload.tool_id)) return state;
      return {
        ...state,
        rows: [
          ...state.rows,
          {
            id: payload.tool_id,
            kind: "tool",
            name: payload.name || "tool",
            args: payload.args ?? null,
            status: "running",
          },
        ],
      };
    }
    case "tool.complete": {
      const payload = event.payload;
      if (!payload?.tool_id) return state;
      const output = String(payload.result_text ?? payload.summary ?? "").trim();
      return replaceRow(state, payload.tool_id, (row) =>
        row.kind === "tool" ? { ...row, status: "done", output: output || row.output } : row,
      );
    }
    case "error": {
      const message = String(event.payload?.message ?? "Request failed");
      return { ...state, rows: [...state.rows, { id: `error-${Date.now()}`, kind: "system", text: message, tone: "error" }] };
    }
    default:
      return state;
  }
}
