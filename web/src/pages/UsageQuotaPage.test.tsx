// @vitest-environment jsdom
import { act, type ReactElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getUsageQuota: vi.fn(),
}));
const headerMocks = vi.hoisted(() => ({
  setAfterTitle: vi.fn(),
  setEnd: vi.fn(),
}));

vi.mock("@/lib/api", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api")>("@/lib/api");
  return { ...actual, api: { ...actual.api, ...apiMocks } };
});
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => headerMocks,
}));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

const response = {
  providers: [
    {
      provider: "openai-codex",
      source: "codex_usage_api",
      fetched_at: "2026-08-30T00:00:00Z",
      title: "Account limits",
      plan: null,
      windows: [
        { label: "Current session", used_percent: 0, reset_at: null, detail: null },
        { label: "Current week", used_percent: null, reset_at: null, detail: "Reset not provided" },
      ],
      details: [],
      unavailable_reason: null,
      available: true,
    },
    {
      provider: "anthropic",
      source: "unavailable",
      fetched_at: "2026-08-30T00:00:00Z",
      title: "Account limits",
      plan: null,
      windows: [],
      details: [],
      unavailable_reason: "Anthropic account limits are unavailable for this credential.",
      available: false,
    },
  ],
};

beforeEach(() => {
  apiMocks.getUsageQuota.mockReset();
  apiMocks.getUsageQuota.mockResolvedValue(response);
  headerMocks.setAfterTitle.mockReset();
  headerMocks.setEnd.mockReset();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("UsageQuotaPage", () => {
  it("keeps zero distinct from unknown and exposes accessible progress semantics", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    await render(
      <MemoryRouter>
        <UsageQuotaPage />
      </MemoryRouter>,
    );

    await vi.waitFor(() => expect(container.textContent).toContain("100% remaining"));
    expect(container.textContent).toContain("Usage unknown");
    expect(container.textContent).toContain("Reset not provided");
    expect(container.querySelector('[role="progressbar"]')).toMatchObject({
      ariaValueNow: "100",
      ariaValueMin: "0",
      ariaValueMax: "100",
    });
    expect(container.textContent).toContain("not local analytics or billing");
    expect(container.textContent).toContain("Unavailable");
    expect(container.textContent).toContain("Anthropic account limits are unavailable");
  });

  it("refreshes through the page-header action and keeps the prior data visible", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    let resolveRefresh!: (value: typeof response) => void;
    apiMocks.getUsageQuota
      .mockResolvedValueOnce(response)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveRefresh = resolve; }));
    await render(
      <MemoryRouter>
        <UsageQuotaPage />
      </MemoryRouter>,
    );
    await vi.waitFor(() => expect(apiMocks.getUsageQuota).toHaveBeenCalledTimes(1));

    await act(async () => root.render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>));
    // The page-header callback receives a button; invoke its click handler via
    // the rendered page after the initial effect has installed it.
    const button = headerMocks.setEnd.mock.calls.at(-1)?.[0] as ReactElement<{
      "aria-label"?: string;
      onClick?: () => void;
    }>;
    expect(button.props["aria-label"]).toBe("Refresh quota");
    await act(async () => button.props.onClick?.());
    expect(apiMocks.getUsageQuota).toHaveBeenCalledTimes(2);
    expect(container.textContent).toContain("100% remaining");
    resolveRefresh(response);
    await act(async () => { await Promise.resolve(); });
    expect(container.textContent).toContain("Updated");

  });

  it("documents that backend metadata is optional rather than fabricating scope", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);
    await vi.waitFor(() => expect(container.textContent).toContain("Source: codex_usage_api"));
    expect(container.textContent).not.toContain("Scope:");
    // The current backend contract provides source/fetched_at only; the UI must
    // not invent a scope, stale flag, or partial-data claim when absent.
  });
});
