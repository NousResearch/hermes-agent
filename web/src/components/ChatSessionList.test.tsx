// @vitest-environment jsdom
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({ getSessions: vi.fn() }));

vi.mock("@/lib/api", () => ({ api: apiMocks }));

let container: HTMLDivElement;
let root: Root;
let search = "";
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function LocationProbe() {
  const location = useLocation();
  useEffect(() => {
    search = location.search;
  }, [location.search]);
  return null;
}

const row = (id: string, title: string) => ({
  id, title, preview: "", source: "cli", model: null, started_at: 1, ended_at: null, last_active: 1,
  is_active: false, message_count: 1, tool_call_count: 0, input_tokens: 0, output_tokens: 0,
});

beforeEach(() => {
  apiMocks.getSessions.mockReset();
  apiMocks.getSessions.mockResolvedValue({ sessions: [row("sess-A", "Alpha"), row("sess-B", "Beta")], total: 2 });
  search = "";
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ChatSessionList row pick", () => {
  it("re-points the URL at the highlighted session when the TUI moved away from ?resume= (#94716)", async () => {
    const [{ ChatSessionList }, { I18nProvider }] = await Promise.all([import("./ChatSessionList"), import("@/i18n")]);
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
    // The URL names sess-B; the PTY runs sess-A, so the rail highlights sess-A.
    await act(async () =>
      root.render(
        <I18nProvider>
          <MemoryRouter initialEntries={["/chat?resume=sess-B"]}>
            <LocationProbe />
            <ChatSessionList activeSessionId="sess-A" />
          </MemoryRouter>
        </I18nProvider>,
      ),
    );
    await vi.waitFor(() => expect(container.textContent).toContain("Alpha"));
    const alpha = [...container.querySelectorAll('[aria-current="true"]')];
    expect(alpha.map((el) => el.textContent)).toEqual([expect.stringContaining("Alpha")]);

    await act(async () => {
      alpha[0].dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
    });

    expect(new URLSearchParams(search).get("resume")).toBe("sess-A");
  });
});
