// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { DebugShareResponse } from "@/lib/api";

const mocks = vi.hoisted(() => ({
  runDebugShare: vi.fn(),
  showToast: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: {
    checkHermesUpdate: vi.fn(async () => ({ update_available: false })),
    getCheckpoints: vi.fn(async () => ({ sessions: [], total_bytes: 0 })),
    getCredentialPool: vi.fn(async () => ({ providers: [] })),
    getCurator: vi.fn(async () => null),
    getHooks: vi.fn(async () => ({ hooks: [], valid_events: [] })),
    getMemory: vi.fn(async () => null),
    getPortal: vi.fn(async () => null),
    getStatus: vi.fn(async () => ({ gateway_running: false })),
    getSystemStats: vi.fn(async () => null),
    runDebugShare: mocks.runDebugShare,
  },
}));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ showToast: mocks.showToast, toast: null }),
}));
vi.mock("@/lib/clipboard", () => ({ copyTextToClipboard: vi.fn() }));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function waitFor(condition: () => boolean, timeoutMs = 5000) {
  const started = Date.now();
  while (!condition()) {
    if (Date.now() - started > timeoutMs) throw new Error("condition did not become true");
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 20));
    });
  }
}

function button(label: string): HTMLButtonElement {
  const match = Array.from(container.querySelectorAll("button")).find(
    (candidate) => candidate.textContent?.trim() === label,
  );
  if (!(match instanceof HTMLButtonElement)) throw new Error(`missing button: ${label}`);
  return match;
}

function exactText(text: string): Element | undefined {
  return Array.from(container.querySelectorAll("*")).find(
    (element) => element.children.length === 0 && element.textContent?.trim() === text,
  );
}

function debugShareResponse(urls: Record<string, string>): DebugShareResponse {
  return {
    ok: true,
    auto_delete_seconds: 10800,
    failures: [],
    redacted: true,
    urls,
  };
}

async function renderAndGenerateShare(expectedUrl: string) {
  const { default: SystemPage } = await import("./SystemPage");
  await act(async () => {
    root.render(
      <MemoryRouter>
        <SystemPage />
      </MemoryRouter>,
    );
  });
  await waitFor(() => container.textContent?.includes("Share debug report") === true);

  await act(async () => button("Generate share link").click());
  await waitFor(() => mocks.runDebugShare.mock.calls.length === 1);
  await waitFor(() => container.textContent?.includes(expectedUrl) === true);
}

beforeEach(() => {
  mocks.runDebugShare.mockReset();
  mocks.showToast.mockReset();
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
  mocks.runDebugShare.mockResolvedValue(
    debugShareResponse({
      Report: "https://paste.rs/report",
      Logs: "https://dpaste.com/logs",
    }),
  );
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  vi.restoreAllMocks();
});

describe("SystemPage debug-share retention notice", () => {
  it("qualifies the six-hour claim when a result includes a dpaste fallback", async () => {
    await renderAndGenerateShare("https://dpaste.com/logs");

    const preUploadNotice =
      "paste.rs links auto-delete in 6h; dpaste.com fallback links are kept for 1 day and cannot be deleted.";
    const resultNotice =
      "paste.rs links auto-delete in 3h; dpaste.com fallback links are kept for 1 day and cannot be deleted.";
    expect(exactText("Share debug report")?.nextElementSibling?.textContent).toContain(
      preUploadNotice,
    );
    expect(exactText("uploaded")?.parentElement?.textContent).toContain(resultNotice);
    expect(container.textContent).not.toContain("Pastes auto-delete after 6 hours.");
  });

  it("omits the paste.rs policy from a dpaste-only result notice", async () => {
    mocks.runDebugShare.mockResolvedValue(
      debugShareResponse({ Report: "https://dpaste.com/report" }),
    );
    await renderAndGenerateShare("https://dpaste.com/report");

    const resultNotice = exactText("uploaded")?.parentElement?.textContent ?? "";
    expect(resultNotice).toContain(
      "dpaste.com fallback links are kept for 1 day and cannot be deleted.",
    );
    expect(resultNotice).not.toContain("paste.rs links auto-delete");
  });
});
