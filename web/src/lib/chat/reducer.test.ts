/**
 * Reducer tests — pure, no react/jsdom, no GatewayClient.
 *
 * Each case resets the in-memory id caches so test ordering can't leak
 * synthetic ids between scenarios.
 *
 * The brief's twelve event-handling checkpoints map onto these tests:
 *   6. message.delta updates streaming assistant text
 *   7. message.complete finalizes the message
 *   8. tool.start creates a tool card
 *   9. tool.progress updates the correct tool
 *  10. tool.complete finalizes the tool
 *  11. reasoning.delta updates reasoning
 *  (+ extras for /resume seeding, user-side echoes, and submit lifecycle.)
 */
import { beforeEach, describe, expect, it } from "vitest";

import {
  type ChatUIAction,
} from "@/components/chat/types";
import {
  chatUIReducer,
  describeModel,
  initialChatUIStore,
  isStreaming,
  lastAssistantTurn,
  resetChatUIIdCounters,
} from "@/lib/chat/reducer";
import type { TranscriptMessage } from "@hermes/shared";

function reduce(
  seq: ChatUIAction[],
  state = initialChatUIStore,
): ReturnType<typeof chatUIReducer> {
  return seq.reduce(
    (acc, action) => chatUIReducer(acc, action),
    state,
  );
}

beforeEach(() => {
  resetChatUIIdCounters();
});

describe("chatUIReducer", () => {
  it("seeds the store from session.resume history", () => {
    const history: TranscriptMessage[] = [
      {
        role: "user",
        text: "hi",
        row_id: 1,
        timestamp: 1000,
        display_kind: null,
        display_metadata: null,
        name: null,
        context: null,
        args: null,
        labels: null,
        reasoning: null,
      },
      {
        role: "assistant",
        text: "hello!",
        row_id: 2,
        timestamp: 1001,
        display_kind: null,
        display_metadata: null,
        name: null,
        context: null,
        args: null,
        labels: null,
        reasoning: null,
      },
    ];
    const result = chatUIReducer(initialChatUIStore, {
      type: "session/resumed",
      sessionId: "s1",
      storedSessionId: "stored-1",
      title: "Greet",
      model: "anthropic/claude-sonnet",
      provider: "anthropic",
      reasoningEffort: "medium",
      history,
      historySeq: 7,
    });
    expect(result.sessionId).toBe("s1");
    expect(result.storedSessionId).toBe("stored-1");
    expect(result.title).toBe("Greet");
    expect(result.model).toBe("anthropic/claude-sonnet");
    expect(result.reasoningEffort).toBe("medium");
    expect(result.turns).toHaveLength(2);
    expect(result.turns[0]).toMatchObject({ role: "user", text: "hi" });
    expect(result.turns[1]).toMatchObject({
      role: "assistant",
      text: "hello!",
      streaming: false,
    });
    expect(result.lastSeenSeq).toBe(7);
    expect(result.hydrated).toBe(true);
    expect(describeModel(result)).toContain("anthropic");
  });

  it("appends message.delta text to the streaming draft", () => {
    const seeded = chatUIReducer(initialChatUIStore, {
      type: "message/start",
    });
    expect(isStreaming(seeded)).toBe(true);
    const t1 = chatUIReducer(seeded, {
      type: "message/delta",
      payload: { text: "Hel" },
    });
    const t2 = chatUIReducer(t1, {
      type: "message/delta",
      payload: { text: "lo " },
    });
    const t3 = chatUIReducer(t2, {
      type: "message/delta",
      payload: { text: "world" },
    });
    const draft = lastAssistantTurn(t3);
    expect(draft?.text).toBe("Hello world");
    expect(isStreaming(t3)).toBe(true);
  });

  it("message.complete finalizes the assistant turn", () => {
    const seeded = chatUIReducer(initialChatUIStore, {
      type: "message/start",
    });
    const streamed = chatUIReducer(seeded, {
      type: "message/delta",
      payload: { text: "Hi" },
    });
    const finalized = chatUIReducer(streamed, {
      type: "message/complete",
      payload: {
        text: "Hi",
        status: "complete",
        usage: {
          input_tokens: 10,
          output_tokens: 5,
          total_tokens: 15,
        },
        reasoning: null,
        warning: null,
        response_previewed: null,
        billing: null,
        failure_reason: null,
        rendered: null,
        error: null,
        recoverable: null,
        error_surface: null,
        partial: null,
      },
    });
    const draft = lastAssistantTurn(finalized);
    expect(draft?.streaming).toBe(false);
    expect(draft?.status).toBe("complete");
    expect(draft?.finalizedAt).toBeTypeOf("number");
    expect(draft?.usage?.total_tokens).toBe(15);
    expect(isStreaming(finalized)).toBe(false);
  });

  it("tool.start creates a new tool card by tool_id", () => {
    const seeded = chatUIReducer(initialChatUIStore, {
      type: "message/start",
    });
    const started = chatUIReducer(seeded, {
      type: "tool/start",
      payload: {
        tool_id: "call_42",
        name: "bash",
        context: "ls -la",
        args: { command: "ls -la" },
        args_text: null,
        preview: null,
        labels: null,
      },
    });
    const draft = lastAssistantTurn(started);
    expect(draft?.tools["call_42"]).toMatchObject({
      toolId: "call_42",
      name: "bash",
      status: "running",
    });
  });

  it("tool.progress updates the correct card by tool_id", () => {
    const seeded = chatUIReducer(initialChatUIStore, { type: "message/start" });
    const started = chatUIReducer(seeded, {
      type: "tool/start",
      payload: {
        tool_id: "call_42",
        name: "bash",
        context: null,
        args: { command: "ls" },
        args_text: null,
        preview: null,
        labels: null,
      },
    });
    const updated = chatUIReducer(started, {
      type: "tool/progress",
      payload: {
        tool_id: "call_42",
        tail: "out 1\nout 2\nout 3",
      },
    });
    const draft = lastAssistantTurn(updated);
    expect(draft?.tools["call_42"].status).toBe("running");
    expect(draft?.tools["call_42"].progress?.tail).toBe(
      "out 1\nout 2\nout 3",
    );
  });

  it("tool.complete finalizes the tool card by tool_id", () => {
    const seeded = chatUIReducer(initialChatUIStore, { type: "message/start" });
    const started = chatUIReducer(seeded, {
      type: "tool/start",
      payload: {
        tool_id: "call_99",
        name: "read_file",
        context: null,
        args: { path: "/tmp/x" },
        args_text: null,
        preview: null,
        labels: null,
      },
    });
    const finalized = chatUIReducer(started, {
      type: "tool/complete",
      payload: {
        tool_id: "call_99",
        name: "read_file",
        args: { path: "/tmp/x" },
        duration_s: 0.42,
        result: "ok",
        summary: "ok",
        result_text: "ok",
        inline_diff: null,
        todos: null,
        revision: null,
        labels: null,
      },
    });
    const draft = lastAssistantTurn(finalized);
    expect(draft?.tools["call_99"].status).toBe("success");
    expect(draft?.tools["call_99"].result).toBe("ok");
    expect(draft?.tools["call_99"].durationS).toBe(0.42);
  });

  it("reasoning.delta updates reasoning text", () => {
    const seeded = chatUIReducer(initialChatUIStore, { type: "message/start" });
    const t1 = chatUIReducer(seeded, {
      type: "reasoning/delta",
      payload: { text: "thinking " },
    });
    const t2 = chatUIReducer(t1, {
      type: "reasoning/delta",
      payload: { text: "about it" },
    });
    const draft = lastAssistantTurn(t2);
    expect(draft?.reasoning).toBe("thinking about it");
  });

  it("user/submitted appends a user turn", () => {
    const result = chatUIReducer(initialChatUIStore, {
      type: "user/submitted",
      text: "Ping",
    });
    expect(result.turns).toHaveLength(1);
    expect(result.turns[0]).toMatchObject({ role: "user", text: "Ping" });
  });

  it("submit/started toggles busy", () => {
    const a = chatUIReducer(initialChatUIStore, { type: "submit/started" });
    expect(a.submitting).toBe(true);
    const b = chatUIReducer(a, { type: "submit/done", ok: true });
    expect(b.submitting).toBe(false);
  });

  it("session/info updates model + title without touching turns", () => {
    const seeded = chatUIReducer(initialChatUIStore, {
      type: "session/resumed",
      sessionId: "s2",
      storedSessionId: undefined,
      title: "Old",
      model: "x/y",
      provider: null,
      reasoningEffort: null,
      history: [],
      historySeq: 0,
    });
    const updated = chatUIReducer(seeded, {
      type: "session/info",
      title: "New",
      model: "anthropic/claude",
      provider: "anthropic",
      reasoningEffort: "low",
    });
    expect(updated.title).toBe("New");
    expect(updated.model).toBe("anthropic/claude");
    expect(updated.reasoningEffort).toBe("low");
    expect(updated.turns).toHaveLength(0);
  });

  it("reset returns to a clean store", () => {
    const seeded = chatUIReducer(initialChatUIStore, {
      type: "user/submitted",
      text: "hi",
    });
    const reset = chatUIReducer(seeded, { type: "reset" });
    expect(reset.turns).toHaveLength(0);
    expect(reset.sessionId).toBeNull();
    expect(reset.hydrated).toBe(false);
  });

  it("message.start is idempotent if a draft is already streaming", () => {
    const a = chatUIReducer(initialChatUIStore, { type: "message/start" });
    const b = chatUIReducer(a, { type: "message/start" });
    expect(b.turns.filter((t) => t.role === "assistant")).toHaveLength(1);
  });

  it("a full streaming sequence yields a finalized assistant turn", () => {
    const seq: ChatUIAction[] = [
      { type: "user/submitted", text: "go" },
      { type: "message/start" },
      { type: "message/delta", payload: { text: "Work" } },
      { type: "message/delta", payload: { text: "ing" } },
      {
        type: "reasoning/delta",
        payload: { text: "step by step" },
      },
      {
        type: "tool/start",
        payload: {
          tool_id: "t1",
          name: "bash",
          context: null,
          args: { command: "echo hi" },
          args_text: null,
          preview: null,
          labels: null,
        },
      },
      {
        type: "tool/complete",
        payload: {
          tool_id: "t1",
          name: "bash",
          args: { command: "echo hi" },
          duration_s: 0.1,
          result: "hi",
          summary: null,
          result_text: null,
          inline_diff: null,
          todos: null,
          revision: null,
          labels: null,
        },
      },
      {
        type: "message/complete",
        payload: {
          text: "Working",
          status: "complete",
          usage: null,
          reasoning: null,
          warning: null,
          response_previewed: null,
          billing: null,
          failure_reason: null,
          rendered: null,
          error: null,
          recoverable: null,
          error_surface: null,
          partial: null,
        },
      },
    ];
    const result = reduce(seq);
    const draft = lastAssistantTurn(result);
    expect(draft?.text).toBe("Working");
    expect(draft?.tools["t1"].status).toBe("success");
    expect(draft?.streaming).toBe(false);
    expect(isStreaming(result)).toBe(false);
  });
});
