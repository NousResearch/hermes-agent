// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { ChatSessionList } from "./ChatSessionList";

const getSessionsMock = vi.fn(async () => ({
  sessions: [
    {
      id: "session-1",
      title: "First conversation",
      last_active: "2026-08-10T12:00:00Z",
      message_count: 5,
    },
    {
      id: "session-2",
      title: "Second conversation",
      last_active: "2026-08-10T13:00:00Z",
      message_count: 10,
    },
  ],
  total: 2,
}));

vi.mock("@/lib/api", () => ({
  api: {
    getSessions: (...args: unknown[]) => getSessionsMock(...args),
  },
}));

vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      common: { loading: "Loading...", retry: "Retry", refresh: "Refresh" },
      sessions: {
        title: "Sessions",
        newChat: "New Chat",
        noSessions: "No sessions",
        untitledSession: "Untitled",
      },
    },
  }),
}));

function LocationSpy({ onLocation }: { onLocation: (loc: { pathname: string; search: string }) => void }) {
  const location = useLocation();
  onLocation(location);
  return null;
}

describe("ChatSessionList navigation", () => {
  let container: HTMLDivElement | null = null;
  let root: Root | null = null;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    getSessionsMock.mockClear();
  });

  afterEach(() => {
    if (root && container) {
      act(() => {
        root?.unmount();
      });
    }
    container?.remove();
    container = null;
    root = null;
  });

  it("navigates to /chat?resume=<id> when clicking a session from a settings/capabilities route", async () => {
    let currentLocation = { pathname: "/models", search: "" };

    await act(async () => {
      root?.render(
        <MemoryRouter initialEntries={["/models"]}>
          <LocationSpy onLocation={(loc) => { currentLocation = loc; }} />
          <Routes>
            <Route
              path="/models"
              element={<ChatSessionList activeSessionId="session-1" />}
            />
            <Route
              path="/chat"
              element={<ChatSessionList activeSessionId="session-1" />}
            />
          </Routes>
        </MemoryRouter>,
      );
    });

    // Wait for session list to render
    const listItems = container?.querySelectorAll("button") || [];
    expect(listItems.length).toBeGreaterThan(0);

    // Click session-1 (which is activeSessionId) while on /models
    const sessionText = Array.from(container?.querySelectorAll("span") || []).find(
      (el) => el.textContent?.trim() === "First conversation",
    );
    expect(sessionText).toBeTruthy();

    await act(async () => {
      (sessionText as HTMLElement).click();
    });

    expect(currentLocation.pathname).toBe("/chat");
    expect(currentLocation.search).toBe("?resume=session-1");
  });

  it("does not trigger navigation when re-clicking active session while already on /chat", async () => {
    let currentLocation = { pathname: "/chat", search: "?resume=session-1" };

    await act(async () => {
      root?.render(
        <MemoryRouter initialEntries={["/chat?resume=session-1"]}>
          <LocationSpy onLocation={(loc) => { currentLocation = loc; }} />
          <Routes>
            <Route
              path="/chat"
              element={<ChatSessionList activeSessionId="session-1" />}
            />
          </Routes>
        </MemoryRouter>,
      );
    });

    const sessionText = Array.from(container?.querySelectorAll("span") || []).find(
      (el) => el.textContent?.trim() === "First conversation",
    );
    expect(sessionText).toBeTruthy();

    // Click session-1 while already on /chat?resume=session-1
    await act(async () => {
      (sessionText as HTMLElement).click();
    });

    // Remains unchanged
    expect(currentLocation.pathname).toBe("/chat");
    expect(currentLocation.search).toBe("?resume=session-1");
  });

  it("navigates to /chat when clicking New Chat from a non-chat route", async () => {
    let currentLocation = { pathname: "/models", search: "" };

    await act(async () => {
      root?.render(
        <MemoryRouter initialEntries={["/models"]}>
          <LocationSpy onLocation={(loc) => { currentLocation = loc; }} />
          <Routes>
            <Route
              path="/models"
              element={<ChatSessionList activeSessionId="session-1" />}
            />
            <Route
              path="/chat"
              element={<div>Chat Page</div>}
            />
          </Routes>
        </MemoryRouter>,
      );
    });

    const buttons = container?.querySelectorAll("button") || [];
    const newChatBtn = Array.from(buttons).find((b) => b.textContent?.includes("New Chat"));
    expect(newChatBtn).toBeTruthy();

    await act(async () => {
      (newChatBtn as HTMLElement).click();
    });

    expect(currentLocation.pathname).toBe("/chat");
    expect(currentLocation.search).toBe("");
  });
});
