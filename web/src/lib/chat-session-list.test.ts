import { describe, expect, it } from "vitest";

import {
  getChatSessionPinsStorageKey,
  partitionPinnedSessions,
  readPinnedSessionIds,
  togglePinnedSessionIds,
  writePinnedSessionIds,
} from "./chat-session-list";

const session = (id: string) => ({
  id,
  source: "dashboard",
  model: null,
  title: id,
  started_at: 1,
  ended_at: null,
  last_active: 1,
  is_active: false,
  message_count: 0,
  tool_call_count: 0,
  input_tokens: 0,
  output_tokens: 0,
  preview: null,
});

describe("chat session pin helpers", () => {
  it("gives the empty profile its own storage key", () => {
    expect(getChatSessionPinsStorageKey()).not.toBe(
      getChatSessionPinsStorageKey("writer"),
    );
    expect(getChatSessionPinsStorageKey("")).toBe(getChatSessionPinsStorageKey());
  });

  it("reads and writes profile-scoped pins without accepting malformed storage", () => {
    const values = new Map<string, string>();
    const storage = {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
    };

    writePinnedSessionIds(storage, "writer", ["one", "one", "two"]);

    expect(readPinnedSessionIds(storage, "writer")).toEqual(["one", "two"]);
    expect(readPinnedSessionIds(storage, "other")).toEqual([]);

    values.set(getChatSessionPinsStorageKey("writer"), "not-json");
    expect(readPinnedSessionIds(storage, "writer")).toEqual([]);
  });

  it("toggles one session id without changing the other pins", () => {
    expect(togglePinnedSessionIds(["one", "two"], "two")).toEqual(["one"]);
    expect(togglePinnedSessionIds(["one"], "two")).toEqual(["one", "two"]);
  });

  it("partitions pinned sessions before recent sessions without duplicates", () => {
    const result = partitionPinnedSessions(
      [session("one"), session("two"), session("three")],
      ["three", "two", "two", "missing"],
    );

    expect(result.pinned.map((item) => item.id)).toEqual(["two", "three"]);
    expect(result.recent.map((item) => item.id)).toEqual(["one"]);
    expect(
      [...result.pinned, ...result.recent].map((item) => item.id),
    ).toEqual(["two", "three", "one"]);
  });

  it("drops duplicate backend rows while preserving their first-seen order", () => {
    const result = partitionPinnedSessions(
      [session("one"), session("one"), session("two"), session("two")],
      ["two"],
    );

    expect(result.pinned.map((item) => item.id)).toEqual(["two"]);
    expect(result.recent.map((item) => item.id)).toEqual(["one"]);
  });
});
