// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  deleteSession: vi.fn(),
  getSessions: vi.fn(),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function waitFor(condition: () => boolean, timeoutMs = 5000) {
  const start = Date.now();
  while (!condition()) {
    if (Date.now() - start > timeoutMs) throw new Error("waitFor: condition never became true");
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
  }
}

function click(element: Element | null) {
  if (!element) throw new Error("element not rendered");
  element.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

function LocationProbe() {
  const location = useLocation();
  return <output data-testid="location">{location.pathname}{location.search}</output>;
}

async function renderList() {
  const [{ ChatSessionList }, { I18nProvider }] = await Promise.all([
    import("./ChatSessionList"),
    import("@/i18n"),
  ]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <I18nProvider>
        <MemoryRouter initialEntries={["/chat?resume=active-session"]}>
          <ChatSessionList activeSessionId="active-session" profile="selected-profile" />
          <LocationProbe />
        </MemoryRouter>
      </I18nProvider>,
    ),
  );
}

beforeEach(() => {
  apiMocks.deleteSession.mockReset();
  apiMocks.getSessions.mockReset();
  apiMocks.getSessions.mockResolvedValue({
    sessions: [
      { id: "active-session", title: "Live session", preview: null, last_active: 1, message_count: 1, source: "cli" },
      { id: "old-session", title: "Old session", preview: null, last_active: 1, message_count: 1, source: "cli", profile: "owning-profile" },
    ],
  });
  apiMocks.deleteSession.mockResolvedValue({ ok: true });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ChatSessionList deletion", () => {
  it("confirms a non-active delete without navigating, routes it to the owning profile, and removes it while keeping the active row undeletable", async () => {
    await renderList();
    await waitFor(() => Boolean(document.querySelector('button[aria-label="Delete session"]')));

    const deleteButton = document.querySelector('button[aria-label="Delete session"]');
    expect(document.querySelectorAll('button[aria-label="Delete session"]')).toHaveLength(1);
    await act(async () => click(deleteButton));

    expect(document.querySelector('[data-testid="location"]')?.textContent).toBe("/chat?resume=active-session");
    await waitFor(() => Boolean(document.querySelector('[role="alertdialog"]')));
    expect(document.querySelector('[role="alertdialog"]')?.textContent).toContain("Old session");

    const confirm = Array.from(document.querySelectorAll('[role="alertdialog"] button')).find(
      (button) => button.textContent?.trim() === "Delete",
    );
    await act(async () => click(confirm ?? null));

    await waitFor(() => apiMocks.deleteSession.mock.calls.length === 1);
    expect(apiMocks.deleteSession).toHaveBeenCalledWith("old-session", "owning-profile");
    await waitFor(() => !container.textContent?.includes("Old session"));
    expect(document.querySelector('button[aria-label="Delete session"]')).toBeNull();
  });
});
