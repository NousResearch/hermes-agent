// @vitest-environment jsdom
import { act, createElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useSearchParams } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getChatSessionPinsStorageKey } from "@/lib/chat-session-list";

const api = vi.hoisted(() => ({
  getSessions: vi.fn(),
  searchSessions: vi.fn(),
  deleteSession: vi.fn(),
  renameSession: vi.fn(),
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
vi.mock("@nous-research/ui/ui/components/input", () => ({
  Input: (props: React.InputHTMLAttributes<HTMLInputElement>) => createElement("input", props),
}));
vi.mock("@/components/DeleteConfirmDialog", () => ({
  DeleteConfirmDialog: ({
    open,
    onCancel,
    onConfirm,
    title,
  }: {
    open: boolean;
    onCancel: () => void;
    onConfirm: () => void;
    title: string;
  }) => open
    ? createElement(
        "div",
        { role: "dialog", "data-delete-dialog": true },
        createElement("span", null, title),
        createElement("button", { type: "button", "data-delete-cancel": true, onClick: onCancel }, "Cancel"),
        createElement("button", { type: "button", "data-delete-confirm": true, onClick: onConfirm }, "Delete"),
      )
    : null,
}));
vi.mock("lucide-react", () => ({
  AlertCircle: () => null,
  Check: () => null,
  MessageSquarePlus: () => null,
  Pin: () => null,
  PinOff: () => null,
  Pencil: () => null,
  RefreshCw: () => null,
  Search: () => null,
  Trash2: () => null,
  X: () => null,
}));

import { ChatSessionList, type SessionActivityStatus } from "./ChatSessionList";

const session = (id: string) => ({
  id, source: "dashboard", model: null, title: id, started_at: 1, ended_at: null,
  last_active: 1, is_active: false, message_count: 0, tool_call_count: 0,
  input_tokens: 0, output_tokens: 0, preview: null,
});

type SearchFixture = ReturnType<typeof session> & {
  session_id: string;
  snippet: string;
};

const searchFixture = (id: string): SearchFixture => ({
  ...session(id),
  session_id: id,
  snippet: id,
});

let root: Root;
let host: HTMLDivElement;

beforeEach(() => {
  localStorage.clear();
  host = document.createElement("div");
  document.body.appendChild(host);
  root = createRoot(host);
  api.getSessions.mockResolvedValue({ sessions: [session("one"), session("two")], total: 2, limit: 30, offset: 0 });
  api.searchSessions.mockResolvedValue({ results: [] });
  api.deleteSession.mockResolvedValue({ ok: true });
  api.renameSession.mockResolvedValue({ ok: true, title: "Renamed session" });
});

afterEach(() => {
  act(() => root.unmount());
  host.remove();
  vi.clearAllMocks();
});

async function render(
  statuses?: Record<string, SessionActivityStatus>,
  props: { profile?: string } = {},
  withLocationProbe = false,
  initialEntries?: string[],
) {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries },
    createElement(ChatSessionList, {
      activeSessionId: "one",
      sessionStatuses: statuses,
      ...props,
    }),
    withLocationProbe ? createElement(LocationProbe) : null,
  )));
  await act(async () => { await Promise.resolve(); });
}

function LocationProbe() {
  const [params] = useSearchParams();
  return createElement("span", { "data-testid": "resume-param" }, params.get("resume") ?? "");
}

async function flushPromises() {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

function setSearchValue(value: string) {
  const input = host.querySelector("input[type=search]") as HTMLInputElement | null;
  if (!input) throw new Error("search input not found");
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
  act(() => {
    setter?.call(input, value);
    input.dispatchEvent(new InputEvent("input", {
      bubbles: true,
      inputType: "insertText",
      data: value,
    }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  });
}

describe("ChatSessionList activity status", () => {
  it("does not add native title attributes to its action buttons", async () => {
    await render();
    expect(host.querySelectorAll("button[title]")).toHaveLength(0);
  });

  it("keeps the mobile session search field at a 16px-safe size", async () => {
    await render();
    const search = host.querySelector<HTMLInputElement>("input[type=search]");
    expect(search?.className).toContain("text-[16px]");
    expect(search?.className).toContain("sm:text-xs");
  });

  it("renders the supported labels for each session status", async () => {
    await render({ one: "ready", two: "working" });
    expect(host.textContent).toContain("Ready");
    expect(host.textContent).toContain("Working");
  });

  it("isolates status updates by session id and falls back to Unknown/Offline", async () => {
    await render({ one: "error" });
    const rows = Array.from(host.querySelectorAll("[data-session-id]:not([data-session-action])"));
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

describe("ChatSessionList pinning and search", () => {
  it("loads profile-scoped pins and renders them before each unpinned session once", async () => {
    localStorage.setItem(
      getChatSessionPinsStorageKey("writer"),
      JSON.stringify(["two"]),
    );
    localStorage.setItem(
      getChatSessionPinsStorageKey("reviewer"),
      JSON.stringify(["one"]),
    );

    await render(undefined, { profile: "writer" });
    expect(
      Array.from(host.querySelectorAll("[data-session-id]:not([data-session-action])"), (row) => row.getAttribute("data-session-id")),
    ).toEqual(["two", "one"]);
    expect(host.querySelector("[data-session-section=pinned]")).not.toBeNull();
    expect(host.querySelector("[data-session-section=recent]")).not.toBeNull();

    await render(undefined, { profile: "reviewer" });
    await flushPromises();
    expect(
      Array.from(host.querySelectorAll("[data-session-id]:not([data-session-action])"), (row) => row.getAttribute("data-session-id")),
    ).toEqual(["one", "two"]);
  });

  it("persists pin and unpin in the selected profile's local store", async () => {
    await render(undefined, { profile: "writer" });

    const pin = host.querySelector(
      '[data-session-action="pin"][data-session-id="one"]',
    ) as HTMLButtonElement | null;
    expect(pin).not.toBeNull();
    act(() => pin?.click());
    expect(JSON.parse(localStorage.getItem(getChatSessionPinsStorageKey("writer")) ?? "[]")).toEqual(["one"]);

    const unpin = host.querySelector(
      '[data-session-action="unpin"][data-session-id="one"]',
    ) as HTMLButtonElement | null;
    expect(unpin).not.toBeNull();
    act(() => unpin?.click());
    expect(JSON.parse(localStorage.getItem(getChatSessionPinsStorageKey("writer")) ?? "[]")).toEqual([]);
  });

  it("keeps row selection and actions as sibling controls", async () => {
    await render();

    const row = host.querySelector("[data-session-id]:not([data-session-action])");
    expect(row?.querySelector("[data-session-select]" )).not.toBeNull();
    expect(row?.querySelector('[data-session-action="pin"]')).not.toBeNull();
    expect(row?.querySelector('[data-session-action="delete"]')).not.toBeNull();
    expect(row?.querySelector("button button")).toBeNull();
  });

  it("renders debounced search results and returns to recent sessions when cleared", async () => {
    vi.useFakeTimers();
    try {
      const result = { ...session("match"), session_id: "match", snippet: "needle" };
      api.searchSessions.mockResolvedValue({ results: [result] });
      await render();
      setSearchValue("needle");
      await flushPromises();

      await act(async () => {
        vi.advanceTimersByTime(279);
      });
      expect(api.searchSessions).not.toHaveBeenCalled();

      await act(async () => {
        vi.advanceTimersByTime(1);
        await Promise.resolve();
      });
      await flushPromises();
      expect(api.searchSessions).toHaveBeenCalledWith("needle", "");
      expect(host.textContent).toContain("match");
      expect(host.textContent).not.toContain("one");

      setSearchValue("");
      await flushPromises();
      expect(host.textContent).toContain("one");
      expect(host.textContent).not.toContain("match");
    } finally {
      vi.useRealTimers();
    }
  });

  it("uses the existing resume URL when a search result is selected", async () => {
    vi.useFakeTimers();
    try {
      api.searchSessions.mockResolvedValue({
        results: [{ ...session("match"), session_id: "match", snippet: "needle" }],
      });
      await render(undefined, {}, true);
      setSearchValue("needle");
      await flushPromises();
      await act(async () => { vi.advanceTimersByTime(280); });
      await flushPromises();

      const resultButton = host.querySelector<HTMLButtonElement>(
        '[data-session-select="match"]',
      );
      expect(resultButton).not.toBeNull();
      act(() => resultButton?.click());
      expect(host.querySelector("[data-testid='resume-param']")?.textContent).toBe("match");
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows a search error with retry and then exposes the no-results state", async () => {
    vi.useFakeTimers();
    try {
      api.searchSessions
        .mockRejectedValueOnce(new Error("Search unavailable"))
        .mockResolvedValueOnce({ results: [] });
      await render();
      setSearchValue("missing");
      await flushPromises();
      await act(async () => { vi.advanceTimersByTime(280); });
      await flushPromises();

      expect(host.querySelector("[role=alert]")?.textContent).toContain("Search unavailable");
      const retry = host.querySelector<HTMLButtonElement>("[role=alert] button");
      expect(retry).not.toBeNull();
      act(() => retry?.click());
      await act(async () => { vi.advanceTimersByTime(280); });
      await flushPromises();
      expect(host.textContent).toContain("No sessions match your search");
      expect(api.searchSessions).toHaveBeenCalledTimes(2);
    } finally {
      vi.useRealTimers();
    }
  });

  it("ignores a stale search response", async () => {
    vi.useFakeTimers();
    try {
      const resolvers = new Map<string, (value: { results: SearchFixture[] }) => void>();
      api.searchSessions.mockImplementation((query: string) => new Promise((resolve) => {
        resolvers.set(query, resolve);
      }));
      await render();

      setSearchValue("old");
      await flushPromises();
      await act(async () => { vi.advanceTimersByTime(300); });
      setSearchValue("new");
      await flushPromises();
      await act(async () => { vi.advanceTimersByTime(300); });
      expect(api.searchSessions).toHaveBeenCalledTimes(2);

      await act(async () => {
        resolvers.get("new")?.({ results: [searchFixture("new")] });
      });
      await act(async () => {
        resolvers.get("old")?.({ results: [searchFixture("old")] });
      });
      await flushPromises();
      expect(host.textContent).toContain("new");
      expect(host.textContent).not.toContain("old");
    } finally {
      vi.useRealTimers();
    }
  });

  it("shows a safe delete confirmation and deletes only after confirmation", async () => {
    await render();
    const deleteButton = host.querySelector(
      '[data-session-action="delete"][data-session-id="two"]',
    ) as HTMLButtonElement | null;
    expect(deleteButton).not.toBeNull();
    act(() => deleteButton?.click());
    expect(host.querySelector("[data-delete-dialog]")).not.toBeNull();
    expect(api.deleteSession).not.toHaveBeenCalled();

    const confirm = host.querySelector("[data-delete-confirm]") as HTMLButtonElement | null;
    await act(async () => {
      confirm?.click();
      await Promise.resolve();
    });
    expect(api.deleteSession).toHaveBeenCalledWith("two", "");
    expect(host.querySelector('[data-session-id="two"]')).toBeNull();
  });

  it("clears the active resume target after a confirmed delete", async () => {
    await render(undefined, {}, true, ["/chat?resume=one"]);
    const deleteButton = host.querySelector(
      '[data-session-action="delete"][data-session-id="one"]',
    ) as HTMLButtonElement | null;
    act(() => deleteButton?.click());
    const confirm = host.querySelector("[data-delete-confirm]") as HTMLButtonElement | null;
    await act(async () => {
      confirm?.click();
      await Promise.resolve();
    });

    expect(api.deleteSession).toHaveBeenCalledWith("one", "");
    expect(host.querySelector("[data-testid='resume-param']")?.textContent).toBe("");
  });

  it("renames a session inline only after the backend confirms", async () => {
    await render();
    const rename = host.querySelector<HTMLButtonElement>(
      '[data-session-action="rename"][data-session-id="one"]',
    );
    expect(rename).not.toBeNull();
    act(() => rename?.click());

    const input = host.querySelector<HTMLInputElement>("input[aria-label='Rename session']");
    expect(input).not.toBeNull();
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")?.set;
    await act(async () => {
      setter?.call(input, "Renamed session");
      input?.dispatchEvent(new Event("input", { bubbles: true }));
      host.querySelector<HTMLButtonElement>("button[aria-label='Save session name']")?.click();
      await Promise.resolve();
    });

    expect(api.renameSession).toHaveBeenCalledWith("one", "Renamed session", "");
    expect(host.querySelector('[data-session-id="one"]')?.textContent).toContain("Renamed session");
  });
});
