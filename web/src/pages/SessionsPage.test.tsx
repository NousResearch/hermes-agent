// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getSessions: vi.fn(),
  getSessionMessages: vi.fn(),
  getEmptySessionsCount: vi.fn(),
  getStatus: vi.fn(),
  searchSessions: vi.fn(),
  importSessions: vi.fn(),
  exportSessionUrl: vi.fn(),
  renameSession: vi.fn(),
  pruneSessions: vi.fn(),
  deleteSession: vi.fn(),
  deleteEmptySessions: vi.fn(),
  bulkDeleteSessions: vi.fn(),
  getProfiles: vi.fn(),
  getActiveProfile: vi.fn(),
  getSessionStats: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: apiMocks,
  // ProfileProvider mirrors its selection into the api module.
  setManagementProfile: vi.fn(),
  getManagementProfile: vi.fn(() => ""),
}));
vi.mock("@/components/PlatformsCard", () => ({ PlatformsCard: () => null }));
vi.mock("@/components/Markdown", () => ({ Markdown: () => null }));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function waitFor(cond: () => boolean, timeoutMs = 5000) {
  const start = Date.now();
  while (!cond()) {
    if (Date.now() - start > timeoutMs) throw new Error("waitFor: condition never became true");
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
  }
}

function click(el: Element | null) {
  if (!el) throw new Error("element not rendered");
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

const button = (label: string) => document.querySelector(`button[aria-label="${label}"]`);

async function renderSessionsPage(rows: Record<string, unknown>[]) {
  // Page list uses limit 20; the overview tab's recent-cards fetch uses 50 —
  // keep the overview empty so the list view (with row actions) renders.
  apiMocks.getSessions.mockImplementation(async (limit: number) => ({
    sessions: limit >= 50 ? [] : rows,
    total: limit >= 50 ? 0 : rows.length,
    limit,
    offset: 0,
  }));
  const [{ default: SessionsPage }, { I18nProvider }, { SystemActionsProvider }, { ProfileProvider }, { PageHeaderProvider }] =
    await Promise.all([
      import("./SessionsPage"),
      import("@/i18n"),
      import("@/contexts/SystemActions"),
      import("@/contexts/ProfileProvider"),
      import("@/contexts/PageHeaderProvider"),
    ]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <I18nProvider>
        <MemoryRouter>
          <SystemActionsProvider>
            <ProfileProvider>
              <PageHeaderProvider pluginTabs={[]}>
                <SessionsPage />
              </PageHeaderProvider>
            </ProfileProvider>
          </SystemActionsProvider>
        </MemoryRouter>
      </I18nProvider>,
    ),
  );
  await waitFor(() => Boolean(button("Delete session")));
}

beforeEach(() => {
  for (const fn of Object.values(apiMocks)) fn.mockReset();
  apiMocks.getStatus.mockResolvedValue({});
  apiMocks.getEmptySessionsCount.mockResolvedValue({ count: 0 });
  apiMocks.getProfiles.mockResolvedValue({ profiles: [] });
  // active === current keeps the management profile "" — the precondition
  // under which an unstamped request hits the process's own store.
  apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
  apiMocks.getSessionStats.mockResolvedValue({ by_source: {} });
  apiMocks.getSessionMessages.mockResolvedValue({ messages: [] });
  apiMocks.deleteSession.mockResolvedValue({ ok: true });
  apiMocks.renameSession.mockResolvedValue({ ok: true, title: "Renamed" });
  apiMocks.exportSessionUrl.mockReturnValue("/api/sessions/x/export");
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 500 })));
  vi.stubGlobal("ResizeObserver", class { disconnect() {} observe() {} unobserve() {} });
  // gsap ticks through rAF; a synchronous callback recurses to death.
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => setTimeout(() => cb(0), 0) as unknown as number);
  vi.stubGlobal("cancelAnimationFrame", (id: number) => clearTimeout(id));
  vi.stubGlobal("matchMedia", () => ({ addEventListener() {}, matches: false, media: "", removeEventListener() {} }));
  sessionStorage.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("SessionsPage per-row profile routing (#99387)", () => {
  it("sends every per-row request to the row's owning profile, not the management default", async () => {
    await renderSessionsPage([
      { id: "sid-guanli", profile: "guanli", source: "cli", model: null, title: "Managed", started_at: 1, ended_at: null,
        last_active: 1, is_active: false, message_count: 2, tool_call_count: 0, input_tokens: 1, output_tokens: 1, preview: "hi" },
    ]);

    // expand → transcript read
    await act(async () => click(button("Delete session")!.closest("div.cursor-pointer")));
    await waitFor(() => apiMocks.getSessionMessages.mock.calls.length > 0);
    expect(apiMocks.getSessionMessages).toHaveBeenCalledWith("sid-guanli", "guanli");

    await act(async () => click(button("Export session")));
    expect(apiMocks.exportSessionUrl).toHaveBeenCalledWith("sid-guanli", "guanli");

    await act(async () => click(button("Rename session")));
    const input = document.querySelector<HTMLInputElement>('input[placeholder="Session title"]');
    if (!input) throw new Error("rename input not rendered");
    await act(async () => {
      Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!.call(input, "Renamed");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => click(button("Save title")));
    expect(apiMocks.renameSession).toHaveBeenCalledWith("sid-guanli", "Renamed", "guanli");

    await act(async () => click(button("Delete session")));
    await waitFor(() => Boolean(document.querySelector('[role="alertdialog"]')));
    const confirm = Array.from(document.querySelectorAll('[role="alertdialog"] button')).find(
      (b) => b.textContent?.trim() === "Delete",
    );
    await act(async () => click(confirm ?? null));
    expect(apiMocks.deleteSession).toHaveBeenCalledWith("sid-guanli", "guanli");
  });
});
