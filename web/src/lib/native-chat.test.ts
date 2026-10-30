import { describe, expect, it } from "vitest";

import {
  EMPTY_CHAT,
  adviserSeverity,
  applyEvent,
  rowsFromHistory,
  type ChatEvent,
} from "./native-chat";

function event<T extends ChatEvent["type"]>(type: T, payload: unknown, seq = 1): ChatEvent {
  return { type, payload, seq } as ChatEvent;
}

describe("rowsFromHistory", () => {
  it("maps an adopted adviser row to a card, not a user bubble", () => {
    const rows = rowsFromHistory([
      {
        role: "user",
        display_kind: "adviser",
        text: "[adviser — advisory only]\n- [concern · reviewer] check the migration",
      },
    ]);
    expect(rows).toHaveLength(1);
    expect(rows[0].kind).toBe("adviser");
    if (rows[0].kind === "adviser") expect(rows[0].severity).toBe("concern");
  });

  it("keeps tool rows and drops empty assistant rows", () => {
    const rows = rowsFromHistory([
      { role: "assistant", text: "done" },
      { role: "assistant", text: "" },
      { role: "tool", name: "read_file", args: { path: "a.py" }, content: "ok" },
    ]);
    expect(rows.map((row) => row.kind)).toEqual(["assistant", "tool"]);
  });
});

describe("applyEvent", () => {
  it("streams one assistant row and seals it on complete", () => {
    let state = applyEvent(EMPTY_CHAT, event("message.start", {}, 1));
    state = applyEvent(state, event("message.delta", { text: "hel" }, 2));
    state = applyEvent(state, event("message.delta", { text: "lo" }, 3));
    expect(state.rows).toHaveLength(1);
    expect(state.rows[0].kind).toBe("assistant");
    if (state.rows[0].kind === "assistant") expect(state.rows[0].text).toBe("hello");

    state = applyEvent(state, event("message.complete", { text: "hello" }, 4));
    if (state.rows[0].kind === "assistant") {
      expect(state.rows[0].streaming).toBe(false);
    }
    expect(state.streamingId).toBeNull();
  });

  it("adds an adviser card at the turn end and dedupes by id", () => {
    const payload = {
      id: "ad-1",
      text: "[adviser] check it",
      severity: "blocker",
      label: "adviser · blocker",
      display_kind: "adviser",
    };
    let state = applyEvent(EMPTY_CHAT, event("message.adviser", payload));
    state = applyEvent(state, event("message.adviser", payload, 2));
    expect(state.rows).toHaveLength(1);
    expect(state.rows[0].kind).toBe("adviser");
  });

  it("updates a tool row in place", () => {
    let state = applyEvent(
      EMPTY_CHAT,
      event("tool.start", { tool_id: "t1", name: "read_file", args: { path: "a.py" } }),
    );
    state = applyEvent(
      state,
      event("tool.complete", { tool_id: "t1", name: "read_file", result_text: "file body" }),
    );
    expect(state.rows).toHaveLength(1);
    if (state.rows[0].kind === "tool") {
      expect(state.rows[0].status).toBe("done");
      expect(state.rows[0].output).toBe("file body");
    }
  });
});

describe("adviserSeverity", () => {
  it("reads the per-note header, then falls back to the block header", () => {
    expect(adviserSeverity("- [blocker · reviewer] stop")).toBe("blocker");
    expect(adviserSeverity("[adviser — concern] note")).toBe("concern");
    expect(adviserSeverity("no marker")).toBe("advisory");
  });
});
