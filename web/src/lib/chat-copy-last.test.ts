import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { SessionMessage } from "@/lib/api";

import { copyLastResponse, lastAssistantText } from "./chat-copy-last";

const getSessionMessages = vi.fn();
const fetchJSON = vi.fn();
const copyTextToClipboard = vi.fn();

vi.mock("@/lib/api", () => ({
  api: {
    getSessionMessages: (...args: unknown[]) => getSessionMessages(...args),
  },
  fetchJSON: (...args: unknown[]) => fetchJSON(...args),
}));

vi.mock("@/lib/clipboard", () => ({
  copyTextToClipboard: (...args: unknown[]) => copyTextToClipboard(...args),
}));

const transcript = (...rows: SessionMessage[]) => ({
  messages: rows,
  pagination: { limit: 500, offset: 0, order: "latest" as const, returned: rows.length },
  session_id: "sess-1",
});

beforeEach(() => {
  vi.clearAllMocks();
  copyTextToClipboard.mockResolvedValue(true);
});

afterEach(() => {
  vi.useRealTimers();
});

describe("lastAssistantText", () => {
  it("takes the newest assistant reply and skips tool-call-only rows", () => {
    const rows: SessionMessage[] = [
      { content: "an older answer", role: "assistant" },
      { content: null, role: "assistant" },
      { content: "tool output", role: "tool", tool_name: "terminal" },
      { content: "the newest answer", role: "assistant" },
      { content: null, role: "assistant" },
    ];

    expect(lastAssistantText(rows)).toBe("the newest answer");
  });

  it("returns null when the transcript holds no assistant text", () => {
    expect(lastAssistantText([{ content: null, role: "assistant" }])).toBeNull();
  });
});

describe("copyLastResponse", () => {
  it("copies the routed session's last reply without asking for the active session", async () => {
    getSessionMessages.mockResolvedValue(
      transcript({ content: "the answer", role: "assistant" }),
    );

    await expect(
      copyLastResponse({ channel: "chan-1", sessionId: "sess-1" }),
    ).resolves.toBe("the answer");

    expect(getSessionMessages).toHaveBeenCalledWith("sess-1", undefined);
    expect(fetchJSON).not.toHaveBeenCalled();
    expect(copyTextToClipboard).toHaveBeenCalledWith("the answer");
  });

  it("falls back to the chat's active session for a fresh chat", async () => {
    fetchJSON.mockResolvedValue({ session_id: "sess-fresh" });
    getSessionMessages.mockResolvedValue(
      transcript({ content: "fresh answer", role: "assistant" }),
    );

    await expect(
      copyLastResponse({ channel: "chan-fresh", sessionId: null }),
    ).resolves.toBe("fresh answer");

    expect(fetchJSON).toHaveBeenCalledWith(
      "/api/chat/active-session?channel=chan-fresh",
    );
    expect(getSessionMessages).toHaveBeenCalledWith("sess-fresh", undefined);
  });

  it("copies nothing when no session is known yet", async () => {
    fetchJSON.mockResolvedValue({ session_id: null });

    await expect(
      copyLastResponse({ channel: "chan-1", sessionId: null }),
    ).resolves.toBeNull();

    expect(getSessionMessages).not.toHaveBeenCalled();
    expect(copyTextToClipboard).not.toHaveBeenCalled();
  });

  it("reports failure instead of claiming a copy the clipboard refused", async () => {
    getSessionMessages.mockResolvedValue(
      transcript({ content: "the answer", role: "assistant" }),
    );
    copyTextToClipboard.mockResolvedValue(false);

    await expect(
      copyLastResponse({ channel: "chan-1", sessionId: "sess-1" }),
    ).resolves.toBeNull();
  });
});
