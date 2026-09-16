import { describe, expect, it } from "vitest";

import {
  initialNativeChatTimeline,
  reduceNativeChatTimeline,
  reduceNativeChatTimelineEvent,
  projectTimelineEntries,
  type NativeChatTimelineState,
} from "./native-chat-timeline";

const event = (type: string, payload: Record<string, unknown> = {}, session_id = "s1") => ({
  type,
  session_id,
  payload,
});

function stream(...events: Array<{ type: string; session_id?: string; payload?: Record<string, unknown>; event_id?: string; seq?: number }>): NativeChatTimelineState {
  return events.reduce(reduceNativeChatTimelineEvent, initialNativeChatTimeline);
}

describe("native chat timeline reducer", () => {
  it("keeps event identity keys distinct across delimiter-bearing scopes", () => {
    const first = reduceNativeChatTimelineEvent(initialNativeChatTimeline, { ...event("message.start", { turn_id: "t1" }, "a"), event_id: "b:c" });
    const second = reduceNativeChatTimelineEvent(first, { ...event("message.start", { turn_id: "t2" }, "a:b"), event_id: "c" });
    expect(second.seenEventIds).toHaveLength(2);
  });

  it("does not delete rekeyed event IDs when the destination is a source prefix", () => {
    const state = reduceNativeChatTimelineEvent(initialNativeChatTimeline, { ...event("message.start", { turn_id: "t1" }), session_key: "a", event_id: "x" });
    const rebound = reduceNativeChatTimeline(state, { type: "rebind-session", fromSession: "s1", toSession: "s1", fromScope: "a", toScope: "a:" });
    expect(rebound.seenEventIds).toHaveLength(1);
  });

  it("does not advance a sequence watermark for a duplicate event ID", () => {
    const first = reduceNativeChatTimelineEvent(initialNativeChatTimeline, { ...event("message.start", { message_id: "m1" }), event_id: "same", seq: 5 });
    const duplicate = reduceNativeChatTimelineEvent(first, { ...event("message.delta", { message_id: "m1", text: "duplicate" }), event_id: "same", seq: 9 });
    const legitimate = reduceNativeChatTimelineEvent(duplicate, { ...event("message.delta", { message_id: "m1", text: "legitimate" }), event_id: "next", seq: 6 });
    expect(duplicate.lastSeqBySession.s1).toBe(5);
    expect(legitimate.entries[0]?.text).toBe("legitimate");
    expect(legitimate.lastSeqBySession.s1).toBe(6);
  });

  it("appends, updates, and completes one turn immutably", () => {
    const started = stream(event("message.start", { turn_id: "t1", message_id: "m1" }));
    const updated = reduceNativeChatTimelineEvent(started, event("message.delta", { turn_id: "t1", message_id: "m1", text: "hello" }));
    const completed = reduceNativeChatTimelineEvent(updated, event("message.complete", { turn_id: "t1", message_id: "m1", text: "hello world" }));

    expect(started.entries).toHaveLength(1);
    expect(started.entries[0].text).toBe("");
    expect(updated.entries[0].text).toBe("hello");
    expect(completed.entries[0]).toMatchObject({ text: "hello world", status: "complete" });
    expect(started).not.toBe(updated);
    expect(started.entries).not.toBe(updated.entries);
  });

  it("keeps an explicit entry when a tagged delta follows an untagged start", () => {
    const started = reduceNativeChatTimeline(initialNativeChatTimeline, {
      type: "append",
      event: event("message.start", { message_id: "m1" }),
      entryId: "assistant-1",
    });
    const updated = reduceNativeChatTimeline(started, {
      type: "update",
      event: event("message.delta", { turn_id: "t1", text: "tagged delta" }),
      entryId: "assistant-1",
    });

    expect(updated.entries[0]?.text).toBe("tagged delta");
    expect(updated.entries[0]?.turnId).toBe("t1");
  });

  it("continues an unbound start with an unbound delta without inventing a turn identity", () => {
    const started = reduceNativeChatTimelineEvent(initialNativeChatTimeline, event("message.start"));
    const updated = reduceNativeChatTimelineEvent(started, event("message.delta", { text: "unbound continuation" }));

    expect(updated.entries).toHaveLength(1);
    expect(updated.entries[0]).toMatchObject({ text: "unbound continuation", status: "streaming" });
    expect(updated.entries[0]).not.toHaveProperty("turnId");
  });

  it("keeps an explicit entry when an untagged delta follows a tagged start", () => {
    const started = reduceNativeChatTimeline(initialNativeChatTimeline, {
      type: "append",
      event: event("message.start", { turn_id: "t1", message_id: "m1" }),
      entryId: "assistant-1",
    });
    const updated = reduceNativeChatTimeline(started, {
      type: "update",
      event: event("message.delta", { text: "untagged delta" }),
      entryId: "assistant-1",
    });

    expect(updated.entries[0]?.text).toBe("untagged delta");
    expect(updated.entries[0]?.turnId).toBe("t1");
  });

  it("treats an explicit append identity as authoritative across turn tags", () => {
    const first = reduceNativeChatTimeline(initialNativeChatTimeline, {
      type: "append",
      event: { ...event("message.start", { turn_id: "t1" }), event_id: "start-1", seq: 1 },
      entryId: "assistant-1",
    });
    const replay = reduceNativeChatTimeline(first, {
      type: "append",
      event: { ...event("message.start", { turn_id: "t2" }), event_id: "start-2", seq: 3 },
      entryId: "assistant-1",
    });
    const late = reduceNativeChatTimeline(replay, {
      type: "update",
      event: { ...event("message.delta", { turn_id: "t2", text: "late" }), event_id: "delta-late", seq: 2 },
      entryId: "assistant-1",
    });

    expect(replay.entries).toHaveLength(1);
    expect(replay.entries[0]?.turnId).toBe("t2");
    expect(replay.seenEventIds.has(JSON.stringify(["s1", "start-2"]))).toBe(true);
    expect(replay.lastSeqBySession.s1).toBe(3);
    expect(late.entries).toEqual(replay.entries);
    expect(late.lastSeqBySession.s1).toBe(3);
    expect(late.seenEventIds.has(JSON.stringify(["s1", "delta-late"]))).toBe(true);
  });

  it("records sequence metadata for an ignored missing-entry update", () => {
    const started = reduceNativeChatTimeline(initialNativeChatTimeline, {
      type: "append",
      event: { ...event("message.start", { turn_id: "t1" }), seq: 1 },
      entryId: "assistant-1",
    });
    const ignored = reduceNativeChatTimeline(started, {
      type: "update",
      event: { ...event("message.delta", { turn_id: "t1", text: "missing" }), event_id: "missing-3", seq: 3 },
      entryId: "other-entry",
    });
    const late = reduceNativeChatTimeline(ignored, {
      type: "update",
      event: { ...event("message.delta", { turn_id: "t1", text: "late" }), seq: 2 },
      entryId: "assistant-1",
    });

    expect(ignored.lastSeqBySession.s1).toBe(3);
    expect(ignored.seenEventIds.has(JSON.stringify(["s1", "missing-3"]))).toBe(true);
    expect(late).toBe(ignored);
  });

  it("records an event id even when its sequence is stale", () => {
    const current = stream(
      { ...event("message.start", { turn_id: "t1", message_id: "m1" }), seq: 3 },
    );
    const stale = reduceNativeChatTimelineEvent(current, {
      ...event("message.delta", { turn_id: "t1", message_id: "m1", text: "old", event_id: "old-2" }),
      event_id: "old-2",
      seq: 2,
    });

    expect(stale).not.toBe(current);
    expect(stale.lastSeqBySession.s1).toBe(3);
    expect(stale.seenEventIds.has(JSON.stringify(["s1", "old-2"]))).toBe(true);
    expect(stale.entries[0]?.text).toBe("");
  });

  it("deduplicates event ids without duplicating deltas", () => {
    const first = reduceNativeChatTimelineEvent(initialNativeChatTimeline, { ...event("message.start", { turn_id: "t1", message_id: "m1" }), event_id: "e1" });
    const second = reduceNativeChatTimelineEvent(first, { ...event("message.delta", { turn_id: "t1", message_id: "m1", text: "x" }), event_id: "e2", seq: 2 });
    const duplicate = reduceNativeChatTimelineEvent(second, { ...event("message.delta", { turn_id: "t1", message_id: "m1", text: "x" }), event_id: "e2", seq: 2 });

    expect(duplicate).toBe(second);
    expect(duplicate.entries[0].text).toBe("x");
  });

  it("does not treat a message id as an event id and ignores unrelated events", () => {
    const started = stream(event("message.start", { turn_id: "t1", message_id: "m1" }));
    const updated = reduceNativeChatTimelineEvent(started, event("message.delta", { turn_id: "t1", message_id: "m1", text: "x" }));

    expect(updated.entries[0].text).toBe("x");
    expect(reduceNativeChatTimelineEvent(updated, event("status.update", { text: "Ready" }))).toBe(updated);
  });

  it("ignores late sequenced events for the same session", () => {
    const current = stream(
      { ...event("message.start", { turn_id: "t1", message_id: "m1" }), seq: 1 },
      { ...event("message.delta", { turn_id: "t1", message_id: "m1", text: "new" }), seq: 4 },
    );
    const late = reduceNativeChatTimelineEvent(current, { ...event("message.delta", { turn_id: "t1", message_id: "m1", text: "old" }), seq: 3 });

    expect(late).toBe(current);
    expect(late.entries[0].text).toBe("new");
  });

  it("isolates sessions and turns even when ids are reused", () => {
    const state = stream(
      event("message.start", { turn_id: "t1", message_id: "same" }, "s1"),
      event("message.start", { turn_id: "t1", message_id: "same" }, "s2"),
      event("message.start", { turn_id: "t2", message_id: "same" }, "s1"),
    );
    const updated = reduceNativeChatTimelineEvent(state, event("message.delta", { turn_id: "t2", message_id: "same", text: "only t2" }, "s1"));

    expect(updated.entries.map((entry) => [entry.sessionId, entry.turnId, entry.text])).toEqual([
      ["s1", "t1", ""], ["s2", "t1", ""], ["s1", "t2", "only t2"],
    ]);
  });

  it("updates an explicit entry after runtime session adoption", () => {
    const started = reduceNativeChatTimeline(initialNativeChatTimeline, {
      type: "append",
      event: event("message.start", { turn_id: "t1", message_id: "m1" }, "runtime-1"),
      entryId: "assistant-local",
    });
    const rebound = reduceNativeChatTimeline(started, { type: "rebind-session", fromSession: "runtime-1", toSession: "runtime-2" });
    const updated = reduceNativeChatTimeline(rebound, {
      type: "update",
      event: event("message.delta", { turn_id: "t1", text: "after reconnect" }, "runtime-2"),
      entryId: "assistant-local",
    });

    expect(updated.entries).toHaveLength(1);
    expect(updated.entries[0]).toMatchObject({ id: "assistant-local", sessionId: "runtime-2", text: "after reconnect" });
  });

  it("deduplicates event ids across runtime sessions with the same session key", () => {
    const first = reduceNativeChatTimelineEvent(initialNativeChatTimeline, {
      type: "message.start",
      session_id: "runtime-1",
      session_key: "stable-session",
      payload: { turn_id: "t1", message_id: "m1", event_id: "event-1" },
    });
    const replay = reduceNativeChatTimelineEvent(first, {
      type: "message.start",
      session_id: "runtime-2",
      session_key: "stable-session",
      payload: { turn_id: "t1", message_id: "m1", event_id: "event-1" },
    });

    expect(replay).toBe(first);
  });

  it("records structured errors and preserves prior text", () => {
    const state = stream(event("message.start", { turn_id: "t1", message_id: "m1" }), event("message.delta", { turn_id: "t1", message_id: "m1", text: "partial" }));
    const errored = reduceNativeChatTimelineEvent(state, event("error", { turn_id: "t1", message_id: "m1", error: "disconnected" }));

    expect(errored.entries[0]).toMatchObject({ text: "partial", status: "error", error: "disconnected" });
  });

  it("supports explicit append/update/complete/error actions", () => {
    const startEvent = event("message.start", { turn_id: "t1", message_id: "m1" });
    let state = reduceNativeChatTimeline(initialNativeChatTimeline, { type: "append", event: startEvent });
    state = reduceNativeChatTimeline(state, { type: "update", event: event("message.delta", { turn_id: "t1", message_id: "m1", text: "x" }) });
    state = reduceNativeChatTimeline(state, { type: "complete", event: event("message.complete", { turn_id: "t1", message_id: "m1" }) });
    state = reduceNativeChatTimeline(state, { type: "error", event: event("error", { turn_id: "t1", message_id: "m1", message: "failed" }) });

    expect(state.entries[0]).toMatchObject({ text: "x", status: "error", error: "failed" });
  });

  it("resets a runtime sequence watermark without clearing entries", () => {
    const state = stream({ ...event("message.start", { turn_id: "t1", message_id: "m1" }), seq: 4 });
    const reset = reduceNativeChatTimeline(state, { type: "reset-sequence", session: "s1" });

    expect(reset.entries).toHaveLength(1);
    expect(reset.lastSeqBySession).toEqual({});
  });

  it("resets all timeline entries, identity, and sequence watermarks immutably", () => {
    const state = stream(
      { ...event("message.start", { turn_id: "t1", message_id: "m1" }), event_id: "e1", seq: 4 },
    );
    const reset = reduceNativeChatTimeline(state, { type: "reset" });

    expect(reset).toEqual(initialNativeChatTimeline);
    expect(reset).not.toBe(state);
    expect(reset.entries).not.toBe(state.entries);
    expect(reset.seenEventIds).not.toBe(state.seenEventIds);
  });

  it("projects explicit identity metadata and leaves unbound events without a default turn", () => {
    const identified = stream(
      event("message.start", { turn_id: "t1", message_id: "m1" }),
      event("message.delta", { turn_id: "t1", message_id: "m1", text: "partial" }),
      event("message.complete", { turn_id: "t1", message_id: "m1" }),
    );
    expect(projectTimelineEntries(identified.entries)).toEqual([
      { id: "m1", role: "assistant", text: "partial", streaming: false, turnId: "t1", messageId: "m1" },
    ]);

    const unbound = stream(event("message.start"));
    expect(unbound.entries[0]?.turnId).toBeUndefined();
    expect(projectTimelineEntries(unbound.entries)[0]).not.toHaveProperty("turnId");
  });
});
