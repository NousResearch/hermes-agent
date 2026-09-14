import { describe, expect, it } from "vitest";

import {
  appendPendingUserMessage,
  createStructuredChatState,
  isTranscriptPinnedToBottom,
  parseStructuredGatewayEvent,
  reduceStructuredChatEvent,
  replaceStructuredChatHistory,
} from "./structured-chat";

describe("structured chat timeline", () => {
  it("reconciles durable history with sequenced live assistant output for the selected session", () => {
    const history = replaceStructuredChatHistory(createStructuredChatState("runtime-1"), [
      { role: "user", text: "Bitte prüfen" },
      { role: "assistant", text: "Ich beginne." },
    ]);

    const started = reduceStructuredChatEvent(history, {
      session_id: "runtime-1",
      seq: 7,
      type: "message.start",
      payload: {},
    });
    const streamed = reduceStructuredChatEvent(started, {
      session_id: "runtime-1",
      seq: 8,
      type: "message.delta",
      payload: { text: "Datei gelesen." },
    });
    const duplicate = reduceStructuredChatEvent(streamed, {
      session_id: "runtime-1",
      seq: 8,
      type: "message.delta",
      payload: { text: "Datei gelesen." },
    });
    const foreign = reduceStructuredChatEvent(duplicate, {
      session_id: "runtime-2",
      seq: 9,
      type: "message.delta",
      payload: { text: "fremd" },
    });

    expect(foreign.items.map((item) => [item.kind, item.text])).toEqual([
      ["message", "Bitte prüfen"],
      ["message", "Ich beginne."],
      ["message", "Datei gelesen."],
    ]);
    expect(foreign.lastSeq).toBe(8);
  });

  it("keeps typed tool, interaction, subagent, and unknown events addressable by stable ids", () => {
    let state = createStructuredChatState("runtime-1");
    const apply = (seq: number, type: string, payload: Record<string, unknown>) => {
      state = reduceStructuredChatEvent(state, { session_id: "runtime-1", seq, type, payload });
    };

    apply(1, "tool.start", { tool_id: "tool-1", name: "read_file", args: { path: "README.md" } });
    apply(2, "tool.progress", { tool_id: "tool-1", text: "reading" });
    apply(3, "tool.complete", { tool_id: "tool-1", result: { content: "ok" } });
    apply(4, "clarify.request", { request_id: "request-1", question: "Fortfahren?" });
    apply(5, "subagent.started", { subagent_id: "agent-1", name: "Scout", role: "research" });
    apply(6, "subagent.completed", { subagent_id: "agent-1", result: "fertig" });
    apply(7, "future.event", { detail: "opaque" });

    expect(state.items.filter((item) => item.id === "agent-1")).toHaveLength(1);
    expect(state.items).toEqual(expect.arrayContaining([
      expect.objectContaining({ id: "tool-1", kind: "tool", status: "complete", title: "read_file", text: '{\n  "content": "ok"\n}' }),
      expect.objectContaining({ id: "request-1", kind: "interaction", status: "pending", text: "Fortfahren?" }),
      expect.objectContaining({ id: "agent-1", kind: "subagent", status: "complete", title: "Scout", text: "fertig" }),
      expect.objectContaining({ id: "event-7", kind: "diagnostic", title: "future.event" }),
    ]));
  });

  it("renders gateway error events with their payload message instead of the unknown-event placeholder", () => {
    let state = createStructuredChatState("runtime-1");
    state = reduceStructuredChatEvent(state, {
      session_id: "runtime-1",
      seq: 1,
      type: "error",
      payload: { message: "Turn cancelled before the agent was ready" },
    });
    state = reduceStructuredChatEvent(state, {
      session_id: "runtime-1",
      seq: 2,
      type: "turn.error",
      payload: { message: "session busy" },
    });

    expect(state.items.map((item) => [item.kind, item.status, item.title, item.text])).toEqual([
      ["diagnostic", "error", "error", "Turn cancelled before the agent was ready"],
      ["diagnostic", "error", "turn.error", "session busy"],
    ]);
    expect(state.items.some((item) => item.text.includes("Unbekanntes strukturiertes Ereignis"))).toBe(false);
  });

  it("consumes session lifecycle events without unknown diagnostic cards while advancing seq", () => {
    let state = createStructuredChatState("runtime-1");
    for (const [seq, type] of [
      [1, "session.info"],
      [2, "status.update"],
      [3, "thinking.delta"],
      [4, "turn.end"],
      [5, "gateway.ready"],
    ] as const) {
      state = reduceStructuredChatEvent(state, {
        session_id: "runtime-1",
        seq,
        type,
        payload: { text: "x" },
      });
    }

    expect(state.items).toEqual([]);
    expect(state.lastSeq).toBe(5);
  });

  it("applies live TUI frames that use the durable session id as an alias", () => {
    const history = replaceStructuredChatHistory(
      createStructuredChatState("runtime-1", ["durable-1"]),
      [{ role: "assistant", text: "Alt" }],
    );
    const parsed = parseStructuredGatewayEvent(JSON.stringify({
      jsonrpc: "2.0",
      method: "event",
      params: {
        type: "message.delta",
        session_id: "durable-1",
        seq: 12,
        payload: { text: "Neu von der TUI" },
      },
    }));
    const live = reduceStructuredChatEvent(history, parsed!);
    expect(live.items.map((item) => item.text)).toEqual(["Alt", "Neu von der TUI"]);
  });

  it("treats a transcript near the bottom as pinned and otherwise not", () => {
    expect(isTranscriptPinnedToBottom({ scrollTop: 900, scrollHeight: 1000, clientHeight: 100 })).toBe(true);
    expect(isTranscriptPinnedToBottom({ scrollTop: 0, scrollHeight: 1000, clientHeight: 100 })).toBe(false);
  });

  it("keeps a pending user send visible until history includes it", () => {
    const pending = appendPendingUserMessage(createStructuredChatState("runtime-1"), "Hallo aus dem Composer");
    expect(pending.items.at(-1)).toEqual(expect.objectContaining({
      kind: "message",
      role: "user",
      text: "Hallo aus dem Composer",
      status: "pending",
    }));
    const stillPending = replaceStructuredChatHistory(pending, [{ role: "assistant", text: "Alt" }]);
    expect(stillPending.items.map((item) => item.text)).toEqual(["Alt", "Hallo aus dem Composer"]);
    const arrived = replaceStructuredChatHistory(stillPending, [
      { role: "assistant", text: "Alt" },
      { role: "user", text: "Hallo aus dem Composer" },
    ]);
    expect(arrived.items.filter((item) => item.role === "user")).toHaveLength(1);
  });
});
