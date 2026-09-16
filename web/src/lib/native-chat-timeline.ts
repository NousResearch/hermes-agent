import type { GatewayEvent } from "@/lib/gatewayClient";

export type TimelineEntryStatus = "streaming" | "complete" | "error";

export type TimelineEntry = {
  id: string;
  sessionId: string;
  turnId?: string;
  messageId?: string;
  text: string;
  status: TimelineEntryStatus;
  error?: string;
  eventIds: readonly string[];
};

export type NativeChatTimelineState = {
  entries: readonly TimelineEntry[];
  seenEventIds: ReadonlySet<string>;
  lastSeqBySession: Readonly<Record<string, number>>;
};

export type TimelineAction =
  | { type: "append"; event: TimelineEventInput; entryId?: string }
  | { type: "update"; event: TimelineEventInput; entryId?: string }
  | { type: "complete"; event: TimelineEventInput; entryId?: string }
  | { type: "error"; event: TimelineEventInput; entryId?: string }
  | { type: "reset-sequence"; session: string }
  | { type: "rebind-session"; fromSession: string; toSession: string; fromScope?: string; toScope?: string }
  | { type: "reset" };

/** The gateway's event envelope is intentionally loose; this adapter only
 * consumes the fields currently emitted by the native page/gateway. */
export type TimelineEventInput = Pick<GatewayEvent, "type" | "session_id" | "payload"> & {
  session_key?: string;
  event_id?: string;
  seq?: number;
};

type Payload = Record<string, unknown>;

export const initialNativeChatTimeline: NativeChatTimelineState = {
  entries: [],
  seenEventIds: new Set<string>(),
  lastSeqBySession: {},
};

function objectPayload(payload: unknown): Payload {
  return typeof payload === "object" && payload !== null ? payload as Payload : {};
}

function stringField(payload: Payload, ...keys: string[]): string | undefined {
  for (const key of keys) {
    if (typeof payload[key] === "string" && payload[key]) return payload[key] as string;
  }
  return undefined;
}

function eventId(event: TimelineEventInput, payload: Payload): string | undefined {
  return event.event_id ?? stringField(payload, "event_id", "eventId");
}

function sessionId(event: TimelineEventInput): string {
  return event.session_id ?? "unknown-session";
}

function turnId(payload: Payload): string | undefined {
  return stringField(payload, "turn_id", "turnId");
}

function messageId(payload: Payload): string | undefined {
  return stringField(payload, "message_id", "messageId", "assistant_id", "assistantId");
}

function entryId(event: TimelineEventInput, payload: Payload, explicit?: string): string {
  const message = messageId(payload);
  const turn = turnId(payload);
  return explicit
    ?? message
    ?? (turn ? `turn:${sessionId(event)}:${turn}` : `unbound:${sessionId(event)}`);
}

function eventText(payload: Payload): string {
  const value = payload.text ?? payload.message;
  return typeof value === "string" ? value : "";
}

function scopedEventKey(scope: string, id: string): string {
  return JSON.stringify([scope, id]);
}

function parseScopedEventKey(value: string): [string, string] | null {
  try {
    const parsed: unknown = JSON.parse(value);
    return Array.isArray(parsed)
      && parsed.length === 2
      && typeof parsed[0] === "string"
      && typeof parsed[1] === "string"
      ? [parsed[0], parsed[1]]
      : null;
  } catch {
    return null;
  }
}

function eventKey(event: TimelineEventInput, id: string): string {
  return scopedEventKey(event.session_key ?? sessionId(event), id);
}

function rekeyEventScopes(values: Set<string>, fromScope?: string, toScope?: string): void {
  if (!fromScope || !toScope || fromScope === toScope) return;
  const additions: Array<{ oldKey: string; newKey: string }> = [];
  for (const value of values) {
    const parsed = parseScopedEventKey(value);
    if (parsed?.[0] === fromScope) additions.push({ oldKey: value, newKey: scopedEventKey(toScope, parsed[1]) });
  }
  for (const { oldKey, newKey } of additions) {
    values.delete(oldKey);
    values.add(newKey);
  }
}

function findEntryIndex(
  entries: readonly TimelineEntry[],
  id: string,
  session: string,
  turn: string | undefined,
  allowTurnMismatch = false,
): number {
  return entries.findIndex((entry) =>
    entry.id === id
      && entry.sessionId === session
      && (allowTurnMismatch
        || turn === undefined
        || entry.turnId === undefined
        || entry.turnId === turn),
  );
}

function withEventId(entry: TimelineEntry, id: string | undefined): TimelineEntry {
  if (!id || entry.eventIds.includes(id)) return entry;
  return { ...entry, eventIds: [...entry.eventIds, id] };
}

function recordEventMetadata(
  state: NativeChatTimelineState,
  session: string,
  key: string | undefined,
  seq: number | undefined,
): NativeChatTimelineState {
  const seenEventIds = key && !state.seenEventIds.has(key)
    ? new Set([...state.seenEventIds, key])
    : state.seenEventIds;
  const previousSeq = state.lastSeqBySession[session];
  const lastSeqBySession = seq !== undefined
    && (previousSeq === undefined || seq > previousSeq)
    ? { ...state.lastSeqBySession, [session]: seq }
    : state.lastSeqBySession;
  if (seenEventIds === state.seenEventIds && lastSeqBySession === state.lastSeqBySession) return state;
  return { ...state, seenEventIds, lastSeqBySession };
}

export function reduceNativeChatTimeline(
  state: NativeChatTimelineState = initialNativeChatTimeline,
  action: TimelineAction,
): NativeChatTimelineState {
  if (action.type === "reset") {
    return { entries: [], seenEventIds: new Set<string>(), lastSeqBySession: {} };
  }
  if (action.type === "reset-sequence") {
    if (!Object.prototype.hasOwnProperty.call(state.lastSeqBySession, action.session)) return state;
    const lastSeqBySession = { ...state.lastSeqBySession };
    delete lastSeqBySession[action.session];
    return { ...state, lastSeqBySession };
  }
  if (action.type === "rebind-session") {
    if (action.fromSession === action.toSession && action.fromScope === action.toScope) return state;
    const entries = state.entries.map((entry) => entry.sessionId === action.fromSession && action.fromSession !== action.toSession
      ? { ...entry, sessionId: action.toSession }
      : entry);
    const lastSeqBySession = { ...state.lastSeqBySession };
    if (action.fromSession !== action.toSession) delete lastSeqBySession[action.fromSession];
    const seenEventIds = new Set(state.seenEventIds);
    rekeyEventScopes(seenEventIds, action.fromScope, action.toScope);
    return { ...state, entries, seenEventIds, lastSeqBySession };
  }
  const payload = objectPayload(action.event.payload);
  const session = sessionId(action.event);
  const turn = turnId(payload);
  const id = eventId(action.event, payload);
  const key = id ? eventKey(action.event, id) : undefined;
  const seq = action.event.seq ?? (typeof payload.seq === "number" ? payload.seq : undefined);

  if (key && state.seenEventIds.has(key)) return state;
  const previousSeq = state.lastSeqBySession[session];
  // Sequence numbers are per session. Missing sequence numbers remain valid;
  // a late sequenced event cannot rewrite newer state.
  if (seq !== undefined && previousSeq !== undefined && seq <= previousSeq) return recordEventMetadata(state, session, key, seq);

  const explicitIdentity = action.entryId !== undefined;
  const resolvedEntryId = action.entryId ?? entryId(action.event, payload);
  const explicitMessageId = messageId(payload);
  const index = findEntryIndex(state.entries, resolvedEntryId, session, turn, explicitIdentity);
  const text = eventText(payload);
  const entries = [...state.entries];

  if (action.type === "append") {
    if (index >= 0) {
      const current = entries[index];
      const effectiveTurnId = turn ?? current.turnId;
      const effectiveMessageId = explicitMessageId ?? current.messageId;
      entries[index] = withEventId({
        ...current,
        sessionId: session,
        ...(effectiveTurnId !== undefined ? { turnId: effectiveTurnId } : {}),
        ...(effectiveMessageId !== undefined ? { messageId: effectiveMessageId } : {}),
      }, id);
    } else {
      entries.push({
        id: resolvedEntryId,
        sessionId: session,
        ...(turn !== undefined ? { turnId: turn } : {}),
        ...(explicitMessageId !== undefined ? { messageId: explicitMessageId } : {}),
        text,
        status: "streaming",
        eventIds: id ? [id] : [],
      });
    }
  } else {
    if (index < 0) return recordEventMetadata(state, session, key, seq);
    const current = entries[index];
    const currentWithoutError = { ...current };
    if (action.type !== "error") delete currentWithoutError.error;
    const effectiveCurrent = currentWithoutError;
    const effectiveTurnId = turn ?? effectiveCurrent.turnId;
    const effectiveMessageId = explicitMessageId ?? effectiveCurrent.messageId;
    const identityMetadata = {
      ...(effectiveTurnId !== undefined ? { turnId: effectiveTurnId } : {}),
      ...(effectiveMessageId !== undefined ? { messageId: effectiveMessageId } : {}),
    };
    const next: TimelineEntry = action.type === "update"
      ? { ...effectiveCurrent, ...identityMetadata, sessionId: session, text: effectiveCurrent.text + text, status: "streaming" }
      : action.type === "complete"
        ? { ...effectiveCurrent, ...identityMetadata, sessionId: session, text: text || effectiveCurrent.text, status: "complete" }
        : { ...effectiveCurrent, ...identityMetadata, sessionId: session, status: "error", error: stringField(payload, "error", "message") ?? "Unknown error" };
    entries[index] = withEventId(next, id);
  }

  const nextState: NativeChatTimelineState = { entries, seenEventIds: state.seenEventIds, lastSeqBySession: state.lastSeqBySession };
  return recordEventMetadata(nextState, session, key, seq);
}

export function projectTimelineEntries(entries: readonly TimelineEntry[]): Array<{ id: string; role: "assistant"; text: string; streaming: boolean; turnId?: string; messageId?: string }> {
  return entries.map((entry) => ({
    id: entry.id,
    role: "assistant" as const,
    text: entry.text,
    streaming: entry.status === "streaming",
    ...(entry.turnId !== undefined ? { turnId: entry.turnId } : {}),
    ...(entry.messageId !== undefined ? { messageId: entry.messageId } : {}),
  }));
}

export function reduceNativeChatTimelineEvent(
  state: NativeChatTimelineState,
  event: TimelineEventInput,
): NativeChatTimelineState {
  const actionType: TimelineAction["type"] | undefined = event.type === "message.start" ? "append"
    : event.type === "message.delta" ? "update"
      : event.type === "message.complete" ? "complete" : event.type === "error" ? "error" : undefined;
  if (!actionType) return state;
  return reduceNativeChatTimeline(state, { type: actionType, event });
}
