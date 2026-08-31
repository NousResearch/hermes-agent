import { describe, expect, it } from "vitest";

import {
  initialNativeChatTimeline,
  reduceNativeChatTimeline,
  reduceNativeChatTimelineEvent,
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
});
