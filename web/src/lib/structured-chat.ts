export type StructuredHistoryMessage = {
  role?: unknown;
  text?: unknown;
  timestamp?: unknown;
  row_id?: unknown;
  name?: unknown;
  context?: unknown;
};

export type StructuredGatewayEvent = {
  payload?: unknown;
  seq?: unknown;
  session_id?: string;
  type: string;
};

export type StructuredTimelineItem = {
  id: string;
  kind: "message" | "tool" | "interaction" | "subagent" | "diagnostic";
  role?: string;
  text: string;
  status?: string;
  title?: string;
};

export type StructuredChatState = {
  sessionId: string;
  aliasSessionIds: string[];
  items: StructuredTimelineItem[];
  lastSeq: number;
  streamingMessageId?: string;
};

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" ? (value as Record<string, unknown>) : {};
}

function text(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function displayValue(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === undefined || value === null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

const SILENT_EVENT_TYPES = new Set([
  "gateway.ready",
  "session.info",
  "session.usage",
  "session.revoked",
  "skin.changed",
  "thinking.delta",
  "reasoning.delta",
  "reasoning.available",
  "status.update",
  "tool.generating",
  "turn.end",
  "todo.updated",
  "notification.show",
  "notification.clear",
]);

export function createStructuredChatState(
  sessionId: string,
  aliasSessionIds: string[] = [],
): StructuredChatState {
  return {
    sessionId,
    aliasSessionIds: [...new Set(aliasSessionIds.filter((id) => id && id !== sessionId))],
    items: [],
    lastSeq: 0,
  };
}

export function parseStructuredGatewayEvent(raw: string): StructuredGatewayEvent | null {
  try {
    const parsed: unknown = JSON.parse(raw);
    const record = parsed && typeof parsed === "object" ? parsed as Record<string, unknown> : {};
    const params = record.method === "event" && record.params && typeof record.params === "object"
      ? record.params as Record<string, unknown>
      : record;
    const type = params.type;
    if (typeof type !== "string" || !type) return null;
    return {
      type,
      session_id: typeof params.session_id === "string" ? params.session_id : undefined,
      seq: params.seq,
      payload: params.payload,
    };
  } catch {
    return null;
  }
}

function eventBelongsToSession(state: StructuredChatState, event: StructuredGatewayEvent): boolean {
  const id = event.session_id;
  if (!id) return false;
  return id === state.sessionId || state.aliasSessionIds.includes(id);
}

export const STRUCTURED_HISTORY_POLL_MS = 2000;

export function isTranscriptPinnedToBottom(
  el: { scrollTop: number; scrollHeight: number; clientHeight: number },
  slop = 64,
): boolean {
  return el.scrollHeight - el.scrollTop - el.clientHeight <= slop;
}

export function pinTranscriptToBottom(el: { scrollTop: number; scrollHeight: number }): void {
  el.scrollTop = el.scrollHeight;
}

export function replaceStructuredChatHistory(
  state: StructuredChatState,
  messages: StructuredHistoryMessage[],
): StructuredChatState {
  const items = messages.flatMap((message, index): StructuredTimelineItem[] => {
    const role = text(message.role);
    if (role === "tool") {
      return [{
        id: `history-tool-${String(message.row_id ?? index)}`,
        kind: "tool",
        role,
        status: "complete",
        text: text(message.context),
        title: text(message.name) || "Tool",
      }];
    }
    const body = text(message.text);
    if (!body) return [];
    return [{
      id: `history-message-${String(message.row_id ?? index)}`,
      kind: "message",
      role: role || "assistant",
      text: body,
    }];
  });
  const pending = state.items.filter((item) => item.status === "pending" && item.role === "user");
  const kept = pending.filter((item) => !items.some((row) => row.role === "user" && row.text === item.text));
  return { ...state, items: [...items, ...kept], streamingMessageId: undefined };
}

export function appendPendingUserMessage(state: StructuredChatState, body: string): StructuredChatState {
  const textValue = body.trim();
  if (!textValue) return state;
  return {
    ...state,
    lastSeq: state.lastSeq + 1,
    items: [...state.items, {
      id: `pending-user-${state.lastSeq + 1}`,
      kind: "message",
      role: "user",
      text: textValue,
      status: "pending",
    }],
  };
}

export function reduceStructuredChatEvent(
  state: StructuredChatState,
  event: StructuredGatewayEvent,
): StructuredChatState {
  if (!eventBelongsToSession(state, event)) return state;
  const seq = typeof event.seq === "number" ? event.seq : undefined;
  if (seq !== undefined && seq <= state.lastSeq) return state;
  const next = { ...state, lastSeq: seq ?? state.lastSeq };
  const payload = record(event.payload);

  if (event.type === "message.start") {
    const id = `live-message-${seq ?? state.items.length}`;
    return {
      ...next,
      streamingMessageId: id,
      items: [...state.items, { id, kind: "message", role: "assistant", text: "", status: "running" }],
    };
  }
  if (event.type === "message.delta") {
    const delta = text(payload.text);
    const id = state.streamingMessageId ?? `live-message-${seq ?? state.items.length}`;
    const found = state.items.some((item) => item.id === id);
    return {
      ...next,
      streamingMessageId: id,
      items: found
        ? state.items.map((item) => item.id === id ? { ...item, text: item.text + delta } : item)
        : [...state.items, { id, kind: "message", role: "assistant", text: delta, status: "running" }],
    };
  }
  if (event.type === "message.complete" && state.streamingMessageId) {
    return {
      ...next,
      streamingMessageId: undefined,
      items: state.items.map((item) => item.id === state.streamingMessageId
        ? { ...item, status: text(payload.status) || "complete", text: item.text || text(payload.text) }
        : item),
    };
  }

  if (event.type === "tool.start") {
    const id = text(payload.tool_id) || `tool-${seq ?? state.items.length}`;
    return {
      ...next,
      items: [...state.items, {
        id,
        kind: "tool",
        status: "running",
        text: text(payload.context),
        title: text(payload.name) || "Tool",
      }],
    };
  }
  if (event.type === "tool.progress" || event.type === "tool.complete") {
    const id = text(payload.tool_id);
    const status = event.type === "tool.complete"
      ? (payload.error ? "error" : "complete")
      : "running";
    const detail = text(payload.text) || text(payload.error) || displayValue(payload.result);
    return {
      ...next,
      items: state.items.map((item) => item.id === id
        ? { ...item, status, text: detail || item.text }
        : item),
    };
  }

  if (["clarify.request", "approval.request", "sudo.request", "secret.request"].includes(event.type)) {
    const id = text(payload.request_id) || `request-${seq ?? state.items.length}`;
    return {
      ...next,
      items: [...state.items, {
        id,
        kind: "interaction",
        status: "pending",
        text: text(payload.question) || text(payload.message) || "Antwort erforderlich",
        title: event.type.replace(".request", ""),
      }],
    };
  }

  if (event.type === "error" || event.type === "turn.error") {
    return {
      ...next,
      items: [...state.items, {
        id: `event-${seq ?? state.items.length}`,
        kind: "diagnostic",
        status: "error",
        text: text(payload.message) || text(payload.text) || "Fehler",
        title: event.type,
      }],
    };
  }

  if (SILENT_EVENT_TYPES.has(event.type)) {
    return next;
  }

  if (event.type.startsWith("subagent.")) {
    const id = text(payload.subagent_id) || text(payload.agent_id) || `subagent-${seq ?? state.items.length}`;
    const terminal = event.type.endsWith("complete") || event.type.endsWith("completed");
    const card = {
      id,
      kind: "subagent" as const,
      role: text(payload.role),
      status: terminal ? "complete" : "running",
      text: text(payload.result),
      title: text(payload.name) || text(payload.role) || "Subagent",
    };
    const found = state.items.some((item) => item.id === id);
    return {
      ...next,
      items: found
        ? state.items.map((item) => item.id === id ? {
            ...item,
            status: card.status,
            text: card.text || item.text,
            role: card.role || item.role,
            title: text(payload.name) || item.title,
          } : item)
        : [...state.items, card],
    };
  }

  return {
    ...next,
    items: [...state.items, {
      id: `event-${seq ?? state.items.length}`,
      kind: "diagnostic",
      status: "unknown",
      text: "Unbekanntes strukturiertes Ereignis",
      title: event.type,
    }],
  };
}
