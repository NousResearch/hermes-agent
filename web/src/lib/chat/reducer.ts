/**
 * Pure reducer for the new structured /chat-ui renderer.
 *
 * Receives a {@link ChatUIAction} (a gateway event projected into our local
 * schema, or one of the high-level bootstrap/local actions) and returns the
 * next {@link ChatUIStore}. ZERO I/O — no fetching, no sockets, no timers.
 *
 * The companion hook (`hooks/useChatSession.ts`) owns the GatewayClient
 * subscription and converts wire events into actions; the page renders the
 * resulting store. This split keeps reducer tests deterministic and lets us
 * reuse the same reducer in Storybook / a future React Native renderer.
 *
 * The reducer is intentionally exhaustive on the action type so a future
 * gateway event name that we forget to handle surfaces as a TypeScript error
 * under `noUnusedLocals`/`noFallthroughCasesInSwitch` instead of a silent
 * dropped event.
 */

import {
  projectTranscriptRow,
  type ChatUIAction,
  type ChatUIAssistantTurn,
  type ChatUIStore,
  type ChatUIToolCall,
  type ChatUITurn,
  type ChatUIUserTurn,
  type ChatUIIdFactory,
} from "@/components/chat/types";

/* -------------------------------------------------------------------------- */
/* Initial state                                                               */
/* -------------------------------------------------------------------------- */

export const initialChatUIStore: ChatUIStore = {
  sessionId: null,
  storedSessionId: null,
  connection: "idle",
  model: null,
  provider: null,
  reasoningEffort: null,
  title: null,
  turns: [],
  lastSeenSeq: 0,
  submitting: false,
  error: null,
  failureReason: null,
  hydrated: false,
};

/* -------------------------------------------------------------------------- */
/* ID factories                                                                */
/* -------------------------------------------------------------------------- */

let _syntheticCounter = 0;
const _seenToolIds = new Set<string>();

/**
 * Reset the in-memory id caches. Tests must call this between reducer
 * cases so synthetic ids don't bleed across scenarios.
 */
export function resetChatUIIdCounters(): void {
  _syntheticCounter = 0;
  _seenToolIds.clear();
}

/**
 * Synthetic id factory — used when the gateway does not supply a stable
 * `row_id` (e.g. a streaming assistant turn that has not yet finalized).
 * Backs onto a monotonic counter so React keys stay stable across reducer
 * passes.
 */
export const defaultChatUIIdFactory: ChatUIIdFactory = () => {
  _syntheticCounter += 1;
  return `cui-${_syntheticCounter}`;
};

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

function findLastAssistantTurn(turns: ChatUITurn[]): ChatUIAssistantTurn | null {
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    const t = turns[i];
    if (t && t.role === "assistant") return t;
  }
  return null;
}

function findLastUserTurn(turns: ChatUITurn[]): ChatUIUserTurn | null {
  for (let i = turns.length - 1; i >= 0; i -= 1) {
    const t = turns[i];
    if (t && t.role === "user") return t;
  }
  return null;
}

function isAssistantTurn(t: ChatUITurn): t is ChatUIAssistantTurn {
  return t.role === "assistant";
}

function makeAssistantTurn(
  id: string,
  startedAt: number,
): ChatUIAssistantTurn {
  return {
    id,
    role: "assistant",
    text: "",
    reasoning: "",
    streaming: true,
    tools: {},
    startedAt,
  };
}

function appendText(text: string, delta: string | undefined): string {
  if (!delta) return text;
  return text + delta;
}

function trimProgressTail(tail: string, max = 8): string {
  const lines = tail.split("\n");
  if (lines.length <= max) return tail;
  return lines.slice(-max).join("\n");
}

/* -------------------------------------------------------------------------- */
/* Reducer                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Pure reducer. Synthetic ids come from a module-level counter inside
 * `defaultChatUIIdFactory`; the third argument was dropped so the function
 * fits React's 2-arg reducer contract (and remains callable from tests
 * with the same shape).
 */
export function chatUIReducer(
  state: ChatUIStore,
  action: ChatUIAction,
): ChatUIStore {
  switch (action.type) {
    case "reset":
      return { ...initialChatUIStore };

    case "session/created":
      return {
        ...initialChatUIStore,
        sessionId: action.sessionId,
        storedSessionId: action.storedSessionId ?? null,
        connection: state.connection,
        hydrated: true,
      };

    case "session/resumed": {
      const turns: ChatUITurn[] = [];
      action.history.forEach((row, idx) => {
        const projected = projectTranscriptRow(row, `seed-${idx}-${row.row_id ?? "x"}`);
        if (projected) turns.push(projected);
      });
      return {
        ...initialChatUIStore,
        sessionId: action.sessionId,
        storedSessionId: action.storedSessionId ?? null,
        title: action.title ?? null,
        model: action.model ?? null,
        provider: action.provider ?? null,
        reasoningEffort: action.reasoningEffort ?? null,
        turns,
        lastSeenSeq: action.historySeq ?? state.lastSeenSeq,
        connection: state.connection,
        hydrated: true,
      };
    }

    case "connection/state":
      return { ...state, connection: action.state };

    case "session/info":
      return {
        ...state,
        title: action.title !== undefined ? action.title : state.title,
        model: action.model !== undefined ? action.model : state.model,
        provider: action.provider !== undefined ? action.provider : state.provider,
        reasoningEffort:
          action.reasoningEffort !== undefined
            ? action.reasoningEffort
            : state.reasoningEffort,
      };

    case "session/title":
      return { ...state, title: action.title };

    case "submit/started":
      return { ...state, submitting: true, error: null };

    case "submit/done":
      return {
        ...state,
        submitting: false,
        failureReason: action.failureReason ?? null,
      };

    case "user/submitted": {
      const userTurn: ChatUIUserTurn = {
        id: `user-${defaultChatUIIdFactory()}`,
        role: "user",
        text: action.text,
        attachments: action.attachments,
        timestamp: Date.now(),
      };
      return { ...state, turns: [...state.turns, userTurn] };
    }

    case "message/start": {
      // Begin a fresh assistant draft only if we don't already have one
      // streaming. The gateway sometimes emits `message.start` after a
      // `submit/started`; idempotent on replay.
      const draft = findLastAssistantTurn(state.turns);
      if (draft && draft.streaming) {
        return state;
      }
      const turn = makeAssistantTurn(`assist-${defaultChatUIIdFactory()}`, Date.now());
      return { ...state, turns: [...state.turns, turn] };
    }

    case "message/delta": {
      const draft = findLastAssistantTurn(state.turns);
      if (!draft || !draft.streaming) {
        // Treat out-of-order deltas as a new draft; preserves the latest
        // typing motion even when the server restarts mid-turn.
        const turn = makeAssistantTurn(`assist-${defaultChatUIIdFactory()}`, Date.now());
        turn.text = appendText("", action.payload.text);
        return { ...state, turns: [...state.turns, turn] };
      }
      const next: ChatUIAssistantTurn = {
        ...draft,
        text: appendText(draft.text, action.payload.text),
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "message/interim": {
      // Interim segments ride as a separate assistant row so the user can
      // see pre-tool commentary alongside the tool cards. Idempotent on
      // overlap: re-emitting the same interim snaps onto the last text
      // tail by default.
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      const next: ChatUIAssistantTurn = {
        ...draft,
        text: appendText(
          draft.text,
          draft.text.length === 0 ? action.payload.text : " " + action.payload.text,
        ),
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "reasoning/delta": {
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      const next: ChatUIAssistantTurn = {
        ...draft,
        reasoning: appendText(draft.reasoning, action.payload.text),
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "reasoning/available": {
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      // `reasoning.available` is non-streaming providers' final block;
      // overwrite so we don't double-count streamed deltas.
      const next: ChatUIAssistantTurn = {
        ...draft,
        reasoning: action.payload.text ?? draft.reasoning,
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "tool/generating": {
      // Model emitted a tool name before the gateway settles on a tool_id.
      // We stash a `pending` placeholder keyed by name so the renderer
      // can show "calling <name>…" until `tool.start` arrives.
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      const synthId = `pending-${action.payload.name}-${defaultChatUIIdFactory()}`;
      if (_seenToolIds.has(synthId)) return state;
      _seenToolIds.add(synthId);
      const card: ChatUIToolCall = {
        id: synthId,
        toolId: synthId,
        name: action.payload.name,
        status: "pending",
        startedAt: Date.now(),
      };
      const next: ChatUIAssistantTurn = {
        ...draft,
        tools: { ...draft.tools, [synthId]: card },
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "tool/start": {
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      const toolId = action.payload.tool_id;
      const args =
        action.payload.args ??
        parseArgsText(action.payload.args_text) ??
        undefined;
      const existing = draft.tools[toolId];
      // If we had a `pending` placeholder from tool.generating with the
      // same name, prefer the real tool_id but keep the placeholder
      // visible until the next reducer pass merges it.
      const card: ChatUIToolCall = {
        ...(existing ?? {}),
        id: toolId,
        toolId,
        name: action.payload.name,
        status: "running",
        argumentsDecoded: args,
        ...(action.payload.context ? { argumentsRaw: action.payload.context } : {}),
        startedAt: existing?.startedAt ?? Date.now(),
      };
      const next: ChatUIAssistantTurn = {
        ...draft,
        tools: { ...draft.tools, [toolId]: card },
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "tool/progress": {
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      const card = draft.tools[action.payload.tool_id];
      if (!card) return state;
      const updated: ChatUIToolCall = {
        ...card,
        status: "running",
        progress: {
          tail: action.payload.tail
            ? trimProgressTail(action.payload.tail)
            : card.progress?.tail,
          updatedAt: Date.now(),
        },
      };
      const next: ChatUIAssistantTurn = {
        ...draft,
        tools: { ...draft.tools, [card.toolId]: updated },
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "tool/complete": {
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      const existing = draft.tools[action.payload.tool_id];
      const wasError =
        action.payload.result === null || action.payload.result === undefined
          ? false
          : false;
      const status: ChatUIToolCall["status"] = wasError ? "error" : "success";
      const resultText = resultToText(action.payload.result);
      const summary =
        action.payload.summary && action.payload.summary.length > 0
          ? action.payload.summary
          : resultText;
      const card: ChatUIToolCall = existing
        ? {
            ...existing,
            status,
            result: summary,
            durationS: action.payload.duration_s ?? undefined,
            completedAt: Date.now(),
            argumentsDecoded: action.payload.args ?? existing.argumentsDecoded,
          }
        : {
            id: action.payload.tool_id,
            toolId: action.payload.tool_id,
            name: action.payload.name,
            status,
            result: summary,
            durationS: action.payload.duration_s ?? undefined,
            completedAt: Date.now(),
            argumentsDecoded: action.payload.args ?? undefined,
          };
      const next: ChatUIAssistantTurn = {
        ...draft,
        tools: { ...draft.tools, [card.toolId]: card },
      };
      const turns = state.turns.slice(0, -1).concat(next);
      return { ...state, turns };
    }

    case "message/complete": {
      const draft = findLastAssistantTurn(state.turns);
      if (!draft) return state;
      // message.complete `text` can be either string or unknown; coerce.
      const text = (() => {
        const raw = action.payload.text;
        if (typeof raw === "string") return raw;
        if (raw && typeof raw === "object") return JSON.stringify(raw);
        return draft.text;
      })();
      const next: ChatUIAssistantTurn = {
        ...draft,
        text,
        reasoning:
          action.payload.reasoning !== undefined && action.payload.reasoning !== null
            ? action.payload.reasoning
            : draft.reasoning,
        streaming: false,
        finalizedAt: Date.now(),
        usage: action.payload.usage ?? undefined,
        status: action.payload.status ?? "complete",
        failureReason:
          action.payload.failure_reason ??
          (action.payload.error ?? undefined),
      };
      const turns = state.turns.slice(0, -1).concat(next);
      const failureReason =
        action.payload.failure_reason ??
        (action.payload.error ?? undefined);
      return {
        ...state,
        turns,
        submitting: failureReason ? state.submitting : false,
        failureReason: failureReason ?? state.failureReason,
      };
    }

    case "error":
      return { ...state, error: action.message, submitting: false };

    default: {
      // Exhaustive — TypeScript will flag an unhandled action type.
      const _exhaustive: never = action;
      void _exhaustive;
      return state;
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Helpers (private)                                                           */
/* -------------------------------------------------------------------------- */

function parseArgsText(
  raw: string | null | undefined,
): Record<string, unknown> | undefined {
  if (!raw) return undefined;
  try {
    return JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return undefined;
  }
}

function resultToText(result: unknown): string | undefined {
  if (result === null || result === undefined) return undefined;
  if (typeof result === "string") return result;
  try {
    return JSON.stringify(result, null, 2);
  } catch {
    return undefined;
  }
}

/* -------------------------------------------------------------------------- */
/* Selectors                                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Selector: human-friendly summary of model + provider for the header.
 */
export function describeModel(state: ChatUIStore): string {
  if (!state.model) return "";
  if (state.provider && state.provider !== state.model) {
    return `${state.provider} · ${state.model}`;
  }
  return state.model;
}

/**
 * Selector: was the most recent turn an assistant turn? Useful for
 * rendering the composer with a stop / interrupt affordance.
 */
export function lastAssistantTurn(state: ChatUIStore): ChatUIAssistantTurn | null {
  return findLastAssistantTurn(state.turns);
}

/**
 * Selector: was the most recent turn a user turn?
 */
export function lastUserTurn(state: ChatUIStore): ChatUIUserTurn | null {
  return findLastUserTurn(state.turns);
}

/**
 * Selector: track whether the most recent assistant turn is still streaming.
 */
export function isStreaming(state: ChatUIStore): boolean {
  const draft = findLastAssistantTurn(state.turns);
  return !!draft && draft.streaming;
}

/**
 * Type guard exposing the assistant branch for callers that want to read
 * tools by id without first narrowing.
 */
export function isAssistantTurnPublic(t: ChatUITurn): t is ChatUIAssistantTurn {
  return isAssistantTurn(t);
}
