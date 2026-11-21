// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * T4.6 — transcript paging. The REST endpoint already pages (`order=latest`
 * anchors at the newest end, `offset` skips back), but the client only ever
 * fetched the newest 500 rows, so long transcripts silently lost their head.
 * The row now offers a "Load earlier messages" button that walks back by
 * `offset` and prepends each page (chronological), hiding when a page
 * returns fewer rows than the limit.
 */

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
  setManagementProfile: vi.fn(),
  getManagementProfile: vi.fn(() => ""),
}));
vi.mock("@/components/PlatformsCard", () => ({ PlatformsCard: () => null }));
vi.mock("@/components/Markdown", () => ({
  Markdown: ({ content }: { content: string }) => <div>{content}</div>,
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

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

beforeEach(() => {
  for (const fn of Object.values(apiMocks)) fn.mockReset();
  apiMocks.getSessions.mockImplementation(async (limit: number) => ({
    sessions: limit >= 50 ? [] : [{ id: "sid-long", title: "Long transcript", profile: "" }],
    total: limit >= 50 ? 0 : 1,
    limit,
    offset: 0,
  }));
  apiMocks.getEmptySessionsCount.mockResolvedValue({ count: 0 });
  apiMocks.getProfiles.mockResolvedValue({ profiles: [] });
  apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
  apiMocks.getSessionStats.mockResolvedValue({ by_source: {} });
  apiMocks.getStatus.mockResolvedValue({});
  vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 500 })));
  vi.stubGlobal("ResizeObserver", class {
    disconnect() {}
    observe() {}
    unobserve() {}
  });
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

const msg = (content: string) => ({ role: "user", content, timestamp: 1000 });
const newerPage = Array.from({ length: 500 }, (_, i) => msg(`newer-${i}`));
const olderPage = [msg("older-0"), msg("older-1")];

async function mountPage() {
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
                <Routes>
                  <Route path="/" element={<SessionsPage />} />
                </Routes>
              </PageHeaderProvider>
            </ProfileProvider>
          </SystemActionsProvider>
        </MemoryRouter>
      </I18nProvider>,
    ),
  );
  await waitFor(() => Boolean(button("Delete session")));
}

describe("transcript paging (T4.6)", () => {
  it("offers a load-earlier button when the first page is full", async () => {
    apiMocks.getSessionMessages
      .mockResolvedValueOnce({
        messages: newerPage,
        pagination: { limit: 500, offset: 0, order: "latest", returned: 500 },
      })
      .mockResolvedValueOnce({
        messages: olderPage,
        pagination: { limit: 500, offset: 500, order: "latest", returned: 2 },
      });
    await mountPage();
    await act(async () => click(button("Delete session")!.closest("div.cursor-pointer")));
    await waitFor(() => apiMocks.getSessionMessages.mock.calls.length > 0);

    const load = button("Load earlier messages");
    expect(load).not.toBeNull();

    await act(async () => click(load));
    expect(apiMocks.getSessionMessages).toHaveBeenLastCalledWith("sid-long", "", { offset: 500, order: "latest" });
    await waitFor(() => container.innerHTML.includes("older-0"));
    // Chronological order: older rows before the newer page.
    expect(container.innerHTML.indexOf("older-0")).toBeLessThan(container.innerHTML.indexOf("newer-0"));
    expect(container.innerHTML.indexOf("older-1")).toBeLessThan(container.innerHTML.indexOf("newer-0"));
    // Last page returned fewer than the limit — the button is gone.
    expect(button("Load earlier messages")).toBeNull();
  });

  it("shows no load-earlier button when the transcript fits in one page", async () => {
    apiMocks.getSessionMessages.mockResolvedValue({
      messages: [msg("only")],
      pagination: { limit: 500, offset: 0, order: "latest", returned: 1 },
    });
    await mountPage();
    await act(async () => click(button("Delete session")!.closest("div.cursor-pointer")));
    await waitFor(() => apiMocks.getSessionMessages.mock.calls.length > 0);
    expect(button("Load earlier messages")).toBeNull();
  });
});
