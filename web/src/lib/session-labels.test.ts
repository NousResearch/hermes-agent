import { describe, expect, it } from "vitest";

import { telegramThreadLabel } from "@/lib/session-labels";
import type { SessionInfo } from "@/lib/api";

function session(overrides: Partial<SessionInfo>): SessionInfo {
  return {
    id: "s1",
    source: null,
    model: null,
    title: null,
    started_at: 0,
    ended_at: null,
    last_active: 0,
    is_active: false,
    message_count: 0,
    tool_call_count: 0,
    input_tokens: 0,
    output_tokens: 0,
    preview: null,
    ...overrides,
  };
}

describe("telegramThreadLabel", () => {
  it("ignores non-Telegram sessions", () => {
    expect(telegramThreadLabel(session({ source: "cli" }))).toBeNull();
  });

  it("shows Telegram DM display context", () => {
    expect(
      telegramThreadLabel(
        session({
          source: "telegram",
          display_name: "Frank van Puffelen",
          chat_type: "dm",
        }),
      ),
    ).toBe("telegram · Frank van Puffelen");
  });

  it("prefers topic names from origin metadata", () => {
    expect(
      telegramThreadLabel(
        session({
          source: "telegram",
          display_name: "Home",
          thread_id: 123,
          chat_type: "group",
          origin_json: JSON.stringify({ chat_topic: "Daily video jobs" }),
        }),
      ),
    ).toBe("telegram · Home · Daily video jobs");
  });

  it("falls back to thread id when topic metadata is absent", () => {
    expect(
      telegramThreadLabel(
        session({
          source: "telegram",
          display_name: "Home",
          thread_id: 456,
          chat_type: "group",
        }),
      ),
    ).toBe("telegram · Home · thread 456");
  });

  it("does not throw on malformed origin metadata", () => {
    expect(
      telegramThreadLabel(
        session({
          source: "telegram",
          display_name: "Home",
          chat_type: "group",
          origin_json: "not json",
        }),
      ),
    ).toBe("telegram · Home");
  });

  it("shows only the platform when no display or thread context exists", () => {
    expect(
      telegramThreadLabel(
        session({
          source: "telegram",
          chat_type: "group",
        }),
      ),
    ).toBe("telegram");
  });
});
