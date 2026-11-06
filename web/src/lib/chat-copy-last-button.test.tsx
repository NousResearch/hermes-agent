// @vitest-environment jsdom
// The dashboard's floating "copy last response" button: it must copy without
// touching the chat PTY composer, and only claim "copied" once the clipboard
// write resolved (#149751).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

import type { SessionMessage } from "@/lib/api";

import { CopyLastButton } from "./chat-copy-last-button";

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

let container: HTMLDivElement;
let root: Root;

const label = () => container.querySelector("button")?.textContent ?? "";

async function render(props: {
  channel: string;
  sessionId: string | null;
  profile?: string;
}) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root.render(<CopyLastButton color="#fff" {...props} />);
  });
}

async function click() {
  const button = container.querySelector("button");
  await act(async () => {
    button?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
  await act(async () => {
    await Promise.resolve();
  });
}

beforeEach(() => {
  vi.clearAllMocks();
  copyTextToClipboard.mockResolvedValue(true);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("CopyLastButton", () => {
  it("copies the last assistant reply over HTTP and says so", async () => {
    getSessionMessages.mockResolvedValue(
      transcript({ content: "the answer", role: "assistant" }),
    );
    await render({ channel: "chan-1", sessionId: "sess-1" });

    expect(label()).toContain("copy last response");
    expect(copyTextToClipboard).not.toHaveBeenCalled();

    await click();

    expect(copyTextToClipboard).toHaveBeenCalledWith("the answer");
    expect(label()).toContain("copied");
  });

  it("resolves a fresh chat's session before copying", async () => {
    fetchJSON.mockResolvedValue({ session_id: "sess-fresh" });
    getSessionMessages.mockResolvedValue(
      transcript({ content: "fresh answer", role: "assistant" }),
    );
    await render({ channel: "chan-fresh", sessionId: null });

    await click();

    expect(fetchJSON).toHaveBeenCalledWith(
      "/api/chat/active-session?channel=chan-fresh",
    );
    expect(copyTextToClipboard).toHaveBeenCalledWith("fresh answer");
    expect(label()).toContain("copied");
  });

  it("does not claim a copy the browser clipboard refused", async () => {
    getSessionMessages.mockResolvedValue(
      transcript({ content: "the answer", role: "assistant" }),
    );
    copyTextToClipboard.mockResolvedValue(false);
    await render({ channel: "chan-1", sessionId: "sess-1" });

    await click();

    expect(label()).toContain("copy failed");
    expect(label()).not.toContain("copied");
  });

  it("says nothing was copied when the chat has no session yet", async () => {
    fetchJSON.mockResolvedValue({ session_id: null });
    await render({ channel: "chan-1", sessionId: null });

    await click();

    expect(getSessionMessages).not.toHaveBeenCalled();
    expect(copyTextToClipboard).not.toHaveBeenCalled();
    expect(label()).toContain("copy failed");
  });
});
