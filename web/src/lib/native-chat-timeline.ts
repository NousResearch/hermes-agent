import type { GatewayEvent } from "@/lib/gatewayClient";

export type TimelineEntryStatus = "streaming" | "complete" | "error";

export type TimelineEntry = {
  id: string;
  sessionId: string;
  turnId: string;
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
  | { type: "error"; event: TimelineEventInput; entryId?: string };

/** The gateway's event envelope is intentionally loose; this adapter only
 * consumes the fields currently emitted by the native page/gateway. */
export type TimelineEventInput = Pick<GatewayEvent, "type" | "session_id" | "payload"> & {
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

function turnId(payload: Payload): string {
  return stringField(payload, "turn_id", "turnId") ?? "default-turn";
}

function entryId(event: TimelineEventInput, payload: Payload, explicit?: string): string {
  return explicit
    ?? stringField(payload, "message_id", "messageId", "assistant_id", "assistantId")
    ?? `turn:${sessionId(event)}:${turnId(payload)}`;
}

function eventText(payload: Payload): string {
  const value = payload.text ?? payload.message;
  return typeof value === "string" ? value : "";
}

function eventKey(event: TimelineEventInput, id: string): string {
  return `${sessionId(event)}:${id}`;
}

function findEntryIndex(entries: readonly TimelineEntry[], id: string, session: string, turn: string): number {
  return entries.findIndex((entry) => entry.id === id && entry.sessionId === session && entry.turnId === turn);
}

function withEventId(entry: TimelineEntry, id: string | undefined): TimelineEntry {
  if (!id || entry.eventIds.includes(id)) return entry;
  return { ...entry, eventIds: [...entry.eventIds, id] };
}

export function reduceNativeChatTimeline(
  state: NativeChatTimelineState = initialNativeChatTimeline,
  action: TimelineAction,
): NativeChatTimelineState {
  const payload = objectPayload(action.event.payload);
  const session = sessionId(action.event);
  const turn = turnId(payload);
  const id = eventId(action.event, payload);
  const key = id ? eventKey(action.event, id) : undefined;

  if (key && state.seenEventIds.has(key)) return state;
  const seq = action.event.seq ?? (typeof payload.seq === "number" ? payload.seq : undefined);
  const previousSeq = state.lastSeqBySession[session];
  // Sequence numbers are per session. Missing sequence numbers remain valid;
  // a late sequenced event cannot rewrite newer state.
  if (seq !== undefined && previousSeq !== undefined && seq <= previousSeq) return state;

  const index = findEntryIndex(state.entries, action.entryId ?? entryId(action.event, payload), session, turn);
  const text = eventText(payload);
  const entries = [...state.entries];

  if (action.type === "append") {
    if (index >= 0) return state;
    entries.push({ id: action.entryId ?? entryId(action.event, payload), sessionId: session, turnId: turn, text, status: "streaming", eventIds: id ? [id] : [] });
  } else {
    if (index < 0) return state;
    const current = entries[index];
    const next: TimelineEntry = action.type === "update"
      ? { ...current, text: current.text + text, status: "streaming" }
      : action.type === "complete"
        ? { ...current, text: text || current.text, status: "complete" }
        : { ...current, status: "error", error: stringField(payload, "error", "message") ?? "Unknown error" };
    entries[index] = withEventId(next, id);
  }

  const seenEventIds = new Set(state.seenEventIds);
  if (key) seenEventIds.add(key);
  const lastSeqBySession = seq === undefined || (previousSeq !== undefined && seq <= previousSeq)
    ? state.lastSeqBySession
    : { ...state.lastSeqBySession, [session]: seq };
  return { entries, seenEventIds, lastSeqBySession };
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
