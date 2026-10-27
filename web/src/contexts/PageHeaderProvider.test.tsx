// @vitest-environment jsdom
import { act, useLayoutEffect, useState } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes, useNavigate } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { PageHeaderProvider } from "./PageHeaderProvider";
import { usePageHeader } from "./usePageHeader";
import ChannelsPage from "@/pages/ChannelsPage";

vi.mock("@/i18n", () => ({ useI18n: () => ({ t: {} }) }));
vi.mock("@/lib/resolve-page-title", () => ({ resolvePageTitle: (path: string) => path }));
vi.mock("@/lib/api", () => ({
  api: { getMessagingPlatforms: vi.fn(async () => ({ platforms: [] })) },
}));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
let container: HTMLDivElement;
let root: Root;

function Navigate() {
  const navigate = useNavigate();
  return <nav>
    <button onClick={() => navigate("/channels")}>Channels</button>
    <button onClick={() => navigate("/other")}>Other</button>
    <button onClick={() => navigate("/empty")}>Empty</button>
    <button onClick={() => navigate("/delayed")}>Delayed</button>
  </nav>;
}

function OtherPage() {
  const { setTitle, setAfterTitle, setEnd } = usePageHeader();
  useLayoutEffect(() => {
    setTitle("Other page");
    setAfterTitle(<span>Other detail</span>);
    setEnd(<button>Other action</button>);
    return () => { setTitle(null); setAfterTitle(null); setEnd(null); };
  }, [setTitle, setAfterTitle, setEnd]);
  return null;
}

function DelayedPage() {
  const [visible, setVisible] = useState(false);
  return <><button onClick={() => setVisible(true)}>Mount page</button>{visible && <OtherPage />}</>;
}

async function renderAt(path: string) {
  await act(async () => root.render(
    <MemoryRouter initialEntries={[path]}>
      <Navigate />
      <PageHeaderProvider pluginTabs={[]}>
        <Routes>
          <Route path="/channels" element={<ChannelsPage />} />
          <Route path="/other" element={<OtherPage />} />
          <Route path="/empty" element={<div>Empty page</div>} />
          <Route path="/delayed" element={<DelayedPage />} />
        </Routes>
      </PageHeaderProvider>
    </MemoryRouter>,
  ));
}

function header() { return container.querySelector("header")!; }
async function click(label: string) {
  const button = Array.from(container.querySelectorAll("button")).find((b) => b.textContent === label);
  if (!button) throw new Error(`Missing button: ${label}`);
  await act(async () => button.click());
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});
afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe("page header ownership", () => {
  it("keeps the actual Channels restart action on a direct load", async () => {
    await renderAt("/channels");
    expect(header().textContent).toContain("Restart gateway");
  });

  it("replaces all slots on navigation and clears them for an empty route", async () => {
    await renderAt("/channels");
    await click("Other");
    expect(header().textContent).toContain("Other page");
    expect(header().textContent).toContain("Other detail");
    expect(header().textContent).toContain("Other action");
    expect(header().textContent).not.toContain("Restart gateway");
    await click("Empty");
    expect(header().textContent).toBe("/empty");
    await click("Channels");
    expect(header().textContent).toContain("Restart gateway");
    expect(header().textContent).not.toContain("Other action");
  });

  it("accepts a page that mounts after the route changes", async () => {
    await renderAt("/channels");
    await click("Delayed");
    expect(header().textContent).toBe("/delayed");
    await click("Mount page");
    expect(header().textContent).toContain("Other action");
  });
});
