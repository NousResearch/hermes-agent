// @vitest-environment jsdom
import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const api = vi.hoisted(() => ({
  getSessions: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api }));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      common: { loading: "Loading", retry: "Retry", refresh: "Refresh" },
      sessions: { title: "Sessions", newChat: "New chat", noSessions: "No sessions", untitledSession: "Untitled session" },
    },
  }),
}));
vi.mock("@nous-research/ui/ui/components/button", () => ({
  Button: ({ children, ...props }: { children?: ReactNode; [key: string]: unknown }) => createElement("button", props, children),
}));
vi.mock("@nous-research/ui/ui/components/list-item", () => ({
  ListItem: ({ children, ...props }: { children?: ReactNode; [key: string]: unknown }) => createElement("button", { type: "button", ...props }, children),
}));
vi.mock("@nous-research/ui/ui/components/spinner", () => ({ Spinner: () => createElement("span", null, "spinner") }));
vi.mock("lucide-react", () => ({ AlertCircle: () => null, MessageSquarePlus: () => null, RefreshCw: () => null }));

import { ChatSessionList, type SessionActivityStatus } from "./ChatSessionList";

const session = (id: string) => ({
  id, source: "dashboard", model: null, title: id, started_at: 1, ended_at: null,
  last_active: 1, is_active: false, message_count: 0, tool_call_count: 0,
  input_tokens: 0, output_tokens: 0, preview: null,
});

let root: Root;
let host: HTMLDivElement;

beforeEach(() => {
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  api.getSessions.mockResolvedValue({ sessions: [session("one"), session("two")], total: 2, limit: 30, offset: 0 });
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.clearAllMocks();
});

async function render(statuses?: Record<string, SessionActivityStatus>) {
  await act(async () => root.render(createElement(MemoryRouter, null,
    createElement(ChatSessionList, { activeSessionId: "one", sessionStatuses: statuses }),
  )));
  await act(async () => { await Promise.resolve(); });
}

describe("ChatSessionList activity status", () => {
  it("does not add native title attributes to its action buttons", async () => {
    await render();
    expect(host.querySelectorAll("button[title]")).toHaveLength(0);
  });

  it("renders the supported labels for each session status", async () => {
    await render({ one: "ready", two: "working" });
    expect(host.textContent).toContain("Ready");
    expect(host.textContent).toContain("Working");
  });

  it("isolates status updates by session id and falls back to Unknown/Offline", async () => {
    await render({ one: "error" });
    const rows = Array.from(host.querySelectorAll("[data-session-id]"));
    expect(rows[0]?.textContent).toContain("Error");
    expect(rows[1]?.textContent).toContain("Unknown/Offline");
    expect(rows[1]?.textContent).not.toContain("Error");
  });

  it("renders waiting for input distinctly from working", async () => {
    await render({ one: "waiting" });
    expect(host.textContent).toContain("Waiting for input");
    expect(host.textContent).not.toContain("Working");
  });
});
