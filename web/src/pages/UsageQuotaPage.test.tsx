// @vitest-environment jsdom
import { act, type ReactElement, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { registerSlot, unregisterPluginSlots } from "@/plugins/slots";

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

  it("preserves non-9Router provider percentage window baseline semantics with out-of-range used_percent and reset_at", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [
        {
          provider: "openai-codex",
          source: "codex_usage_api",
          fetched_at: "2026-08-30T00:00:00Z",
          title: "Account limits",
          plan: null,
          windows: [
            {
              label: "Current session",
              used_percent: -20,
              reset_at: "2026-09-13T12:00:00Z",
              detail: null,
            },
          ],
          details: [],
          unavailable_reason: null,
          available: true,
        },
      ],
    });
    await render(
      <MemoryRouter>
        <UsageQuotaPage />
      </MemoryRouter>,
    );

    await vi.waitFor(() => expect(container.textContent).toContain("120% remaining"));
    expect(container.textContent).toContain("-20% used");
    expect(container.textContent).toContain("Resets ");
    expect(container.textContent).not.toContain("Resets in ");
    expect(container.querySelector('[role="progressbar"]')).toMatchObject({
      ariaValueNow: "120",
      ariaValueMin: "0",
      ariaValueMax: "100",
    });
  });

  it("renders 9Router route rows without inventing a combined total", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: true,
        scope: "profile:quota-test",
        routes: [
          {
            route: "daily",
            provider: "route-provider",
            account: "account-a",
            usage: 12,
            limit: 100,
            remaining: 88,
            remaining_percent: 88,
            unit: "requests",
            reset_at: "2026-09-14T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "weekly",
            provider: "route-provider",
            account: "account-b",
            usage: null,
            limit: null,
            remaining: null,
            unit: null,
            reset_at: null,
            status: "unavailable",
            source: "9router_management_api",
            detail: "9Router management authentication is required for this route.",
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("daily"));
    expect(container.textContent).toContain("account-a");
    expect(container.textContent).toContain("1 quota");
    expect(container.textContent).toContain("Used 12 requests");
    expect(container.textContent).toContain("Limit 100 requests");
    expect(container.textContent).toContain("Remaining 88 requests");
    expect(container.textContent).toContain("88% remaining");
    expect(container.textContent).toContain("Resets");
    expect(container.textContent).toContain("weekly");
    expect(container.textContent).toContain("account-b");
    expect(container.textContent).toContain("Unavailable");
    expect(container.textContent).toContain("authentication is required");
    expect(container.textContent).toContain("No combined total");
    expect(container.textContent).not.toContain("Combined total:");
    const accountCards = Array.from(container.querySelectorAll<HTMLDetailsElement>('[data-testid="usage-quota-account-card"]'));
    expect(accountCards).toHaveLength(2);
    expect(accountCards.every((card) => card.open)).toBe(true);
    await act(async () => accountCards[0]?.querySelector("summary")?.click());
    expect(accountCards[0]?.open).toBe(false);
    expect(accountCards[1]?.open).toBe(true);
    await act(async () => accountCards[1]?.querySelector("summary")?.click());
    expect(accountCards[0]?.open).toBe(false);
    expect(accountCards[1]?.open).toBe(false);
    expect(container.querySelectorAll('[data-testid="usage-quota-route"]').length).toBe(2);
    expect(container.querySelectorAll('[role="progressbar"]').length).toBe(1);
    expect(container.querySelector('[role="progressbar"]')).toMatchObject({
      ariaValueNow: "88",
      ariaValueMin: "0",
      ariaValueMax: "100",
    });
  });

  it("collapses Antigravity model quotas into Gemini/GPT+Claude 5 hr rows and keeps weekly quotas", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    const modelRoute = (route: string) => ({
      route,
      provider: "antigravity",
      account: "antigravity-account",
      usage: 120,
      limit: 1000,
      remaining: 880,
      remaining_percent: 88,
      unit: null,
      reset_at: "2026-09-13T10:52:41Z",
      status: "reported" as const,
      source: "9router_management_api",
      detail: null,
    });
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          modelRoute("gemini-3.6-flash-high"),
          modelRoute("gemini-3.7-flash-low"),
          modelRoute("claude-opus-4-6-thinking"),
          modelRoute("gpt-oss-120b-medium"),
          {
            ...modelRoute("gemini_weekly"),
            usage: 12,
            remaining: 988,
            remaining_percent: 98.8,
            reset_at: "2026-09-20T00:00:00Z",
          },
          {
            ...modelRoute("claude_gpt_weekly"),
            usage: 20,
            remaining: 980,
            remaining_percent: 98,
            reset_at: "2026-09-20T00:00:00Z",
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));
    expect(container.textContent).toContain("claude_gpt_5hr");
    expect(container.textContent).not.toContain("gemini-3.6-flash-high");
    expect(container.textContent).not.toContain("gemini-3.7-flash-low");
    expect(container.textContent).not.toContain("claude-opus-4-6-thinking");
    expect(container.textContent).not.toContain("gpt-oss-120b-medium");
    expect(container.textContent).toContain("gemini_weekly");
    expect(container.textContent).toContain("claude_gpt_weekly");
    expect(container.textContent).toContain("4 quotas");
    expect(container.querySelectorAll('[data-testid="usage-quota-route"]').length).toBe(4);
    expect(container.querySelectorAll('[data-testid="usage-quota-route"]:nth-child(1)')[0]?.textContent).toContain("gemini_5hr");
    expect(container.querySelectorAll('[data-testid="usage-quota-route"]:nth-child(2)')[0]?.textContent).toContain("claude_gpt_5hr");
  });

  it("keeps an inconsistent Antigravity 5 hr window unknown instead of aggregating models", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    const modelRoute = (route: string, usage: number) => ({
      route,
      provider: "antigravity",
      account: "antigravity-account",
      usage,
      limit: 1000,
      remaining: 1000 - usage,
      remaining_percent: 100 - usage / 10,
      unit: null,
      reset_at: "2026-09-13T10:52:41Z",
      status: "reported" as const,
      source: "9router_management_api",
      detail: null,
    });
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          modelRoute("gemini-3.6-flash-high", 120),
          modelRoute("gemini-3.7-flash-low", 240),
          modelRoute("claude-opus-4-6-thinking", 120),
          modelRoute("gpt-oss-120b-medium", 120),
          {
            ...modelRoute("gemini_weekly", 12),
            reset_at: "2026-09-20T00:00:00Z",
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));
    const fiveHourRow = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'))
      .find((row) => row.textContent?.includes("gemini_5hr"));
    expect(fiveHourRow?.textContent).toContain("Unknown");
    expect(fiveHourRow?.textContent).toContain("inconsistent");
    expect(fiveHourRow?.textContent).not.toContain("Used 120");
    expect(container.textContent).toContain("claude_gpt_5hr");
    expect(container.textContent).toContain("Used 120");
    expect(container.textContent).not.toContain("gemini-3.6-flash-high");
    expect(container.textContent).not.toContain("gemini-3.7-flash-low");
    expect(container.textContent).not.toContain("claude-opus-4-6-thinking");
    expect(container.textContent).not.toContain("gpt-oss-120b-medium");
    expect(container.textContent).toContain("gemini_weekly");
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

  it("does not render external provider cards on the built-in quota page", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    function DiagnosticsCard() {
      return <div data-testid="provider-diagnostics-slot">external provider card</div>;
    }
    registerSlot("test-provider-plugin", "test-provider-cards", DiagnosticsCard);
    try {
      await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);
      await vi.waitFor(() => expect(container.textContent).toContain("Source: codex_usage_api"));
      expect(container.querySelector("[data-testid=provider-diagnostics-slot]")).toBeNull();
    } finally {
      unregisterPluginSlots("test-provider-plugin");
    }
  });

  it("renders independent account cards when duplicate account labels exist across providers", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        routes: [
          {
            route: "daily",
            provider: "provider-alpha",
            account: "shared-account",
            usage: 10,
            limit: 100,
            remaining: 90,
            remaining_percent: 90,
            unit: "requests",
            reset_at: "2026-09-14T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "weekly",
            provider: "provider-beta",
            account: "shared-account",
            usage: 25,
            limit: 50,
            remaining: 25,
            remaining_percent: 50,
            unit: "requests",
            reset_at: "2026-09-21T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("daily"));
    expect(container.textContent).toContain("weekly");

    const accountCards = Array.from(
      container.querySelectorAll<HTMLDetailsElement>('[data-testid="usage-quota-account-card"]')
    );
    expect(accountCards).toHaveLength(2);

    expect(accountCards[0]?.textContent).toContain("shared-account");
    expect(accountCards[0]?.textContent).toContain("provider-alpha");
    expect(accountCards[0]?.textContent).toContain("daily");

    expect(accountCards[1]?.textContent).toContain("shared-account");
    expect(accountCards[1]?.textContent).toContain("provider-beta");
    expect(accountCards[1]?.textContent).toContain("weekly");

    expect(accountCards[0]?.open).toBe(true);
    expect(accountCards[1]?.open).toBe(true);
    await act(async () => accountCards[0]?.querySelector("summary")?.click());
    expect(accountCards[0]?.open).toBe(false);
    expect(accountCards[1]?.open).toBe(true);
  });

  it("fails closed to unknown with no usable quota when explicit gemini_5hr conflicts with a gemini model row", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    const modelRoute = (route: string, usage: number) => ({
      route,
      provider: "antigravity",
      account: "antigravity-account",
      usage,
      limit: 1000,
      remaining: 1000 - usage,
      remaining_percent: 100 - usage / 10,
      unit: null,
      reset_at: "2026-09-13T10:52:41Z",
      status: "reported" as const,
      source: "9router_management_api",
      detail: null,
    });
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          modelRoute("gemini_5hr", 100),
          modelRoute("gemini-3.6-flash-high", 120),
          modelRoute("claude_gpt_5hr", 120),
          {
            ...modelRoute("gemini_weekly", 12),
            reset_at: "2026-09-20T00:00:00Z",
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));
    const geminiRow = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'))
      .find((row) => row.textContent?.includes("gemini_5hr"));
    expect(geminiRow).toBeDefined();
    expect(geminiRow?.textContent).toContain("Unknown");
    expect(geminiRow?.textContent).toContain("inconsistent");
    expect(geminiRow?.textContent).not.toContain("Used 100");
    expect(geminiRow?.textContent).not.toContain("Used 120");
    expect(geminiRow?.textContent).toContain("Used Unknown");
    expect(geminiRow?.textContent).toContain("Limit Unknown");
    expect(geminiRow?.textContent).toContain("Remaining Unknown");
    expect(geminiRow?.querySelector('[role="progressbar"]')).toBeNull();
    expect(container.textContent).not.toContain("gemini-3.6-flash-high");
    expect(container.textContent).toContain("claude_gpt_5hr");
    expect(container.textContent).toContain("Used 120");
  });

  it("fails closed to exactly one unknown family row when duplicate explicit family rows exist", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    const explicitRow = (usage: number) => ({
      route: "gemini_5hr",
      provider: "antigravity",
      account: "antigravity-account",
      usage,
      limit: 1000,
      remaining: 1000 - usage,
      remaining_percent: 100 - usage / 10,
      unit: null,
      reset_at: "2026-09-13T10:52:41Z",
      status: "reported" as const,
      source: "9router_management_api",
      detail: null,
    });
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          explicitRow(120),
          explicitRow(120),
          {
            route: "claude_gpt_5hr",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 50,
            limit: 1000,
            remaining: 950,
            remaining_percent: 95,
            unit: null,
            reset_at: "2026-09-13T10:52:41Z",
            status: "reported" as const,
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "gemini_weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 12,
            limit: 1000,
            remaining: 988,
            remaining_percent: 98.8,
            unit: null,
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported" as const,
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));
    const geminiRows = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'))
      .filter((row) => row.textContent?.includes("gemini_5hr"));
    expect(geminiRows).toHaveLength(1);
    expect(geminiRows[0]?.textContent).toContain("Unknown");
    expect(geminiRows[0]?.textContent).toContain("Multiple explicit quota windows");
    expect(geminiRows[0]?.textContent).not.toContain("Used 120");
    expect(geminiRows[0]?.textContent).toContain("Used Unknown");
    expect(geminiRows[0]?.querySelector('[role="progressbar"]')).toBeNull();
  });

  it("displays a valid explicit family row alone and preserves the four-family contract", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          {
            route: "gemini_5hr",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 120,
            limit: 1000,
            remaining: 880,
            remaining_percent: 88,
            unit: "requests",
            reset_at: "2026-09-13T10:52:41Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "claude-opus-4-6-thinking",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 200,
            limit: 1000,
            remaining: 800,
            remaining_percent: 80,
            unit: "requests",
            reset_at: "2026-09-13T10:52:41Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "gemini_weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 12,
            limit: 1000,
            remaining: 988,
            remaining_percent: 98.8,
            unit: "requests",
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "claude_gpt_weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 20,
            limit: 1000,
            remaining: 980,
            remaining_percent: 98,
            unit: "requests",
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));
    expect(container.textContent).toContain("claude_gpt_5hr");
    expect(container.textContent).toContain("gemini_weekly");
    expect(container.textContent).toContain("claude_gpt_weekly");
    expect(container.textContent).not.toContain("claude-opus-4-6-thinking");

    const routeRows = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'));
    expect(routeRows).toHaveLength(4);
    expect(container.textContent).toContain("4 quotas");

    const geminiRow = routeRows.find((row) => row.textContent?.includes("gemini_5hr"));
    expect(geminiRow).toBeDefined();
    expect(geminiRow?.textContent).toContain("Reported");
    expect(geminiRow?.textContent).toContain("Used 120 requests");
    expect(geminiRow?.textContent).toContain("Limit 1,000 requests");
    expect(geminiRow?.textContent).toContain("Remaining 880 requests");
    expect(geminiRow?.textContent).toContain("88% remaining");
  });

  it("treats malformed explicit family data as unknown rather than trusting it", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          {
            route: "gemini_5hr",
            provider: "antigravity",
            account: "antigravity-account",
            usage: -10,
            limit: 1000,
            remaining: 1010,
            remaining_percent: 101,
            unit: "requests",
            reset_at: "2026-09-13T10:52:41Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));
    const geminiRow = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'))
      .find((row) => row.textContent?.includes("gemini_5hr"));
    expect(geminiRow).toBeDefined();
    expect(geminiRow?.textContent).toContain("Unknown");
    expect(geminiRow?.textContent).toContain("unusable quota data");
    expect(geminiRow?.textContent).not.toContain("Used -10");
    expect(geminiRow?.textContent).toContain("Used Unknown");
  });

  it("enforces the exact-four Antigravity contract in order, synthesizing missing families and ignoring generic weekly or model labels", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          {
            route: "gemini_5hr",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 10,
            limit: 100,
            remaining: 90,
            remaining_percent: 90,
            unit: "requests",
            reset_at: "2026-09-13T10:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 5,
            limit: 100,
            remaining: 95,
            remaining_percent: 95,
            unit: "requests",
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "gemini-1.5-pro",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 10,
            limit: 100,
            remaining: 90,
            remaining_percent: 90,
            unit: "requests",
            reset_at: "2026-09-13T10:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          {
            route: "arbitrary_weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 2,
            limit: 100,
            remaining: 98,
            remaining_percent: 98,
            unit: "requests",
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));

    const routeRows = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'));
    // Exactly four rows per account
    expect(routeRows).toHaveLength(4);
    expect(container.textContent).toContain("4 quotas");

    // Exact order: gemini_5hr, claude_gpt_5hr, gemini_weekly, claude_gpt_weekly
    expect(routeRows[0]?.textContent).toContain("gemini_5hr");
    expect(routeRows[0]?.textContent).toContain("Reported");

    expect(routeRows[1]?.textContent).toContain("claude_gpt_5hr");
    expect(routeRows[1]?.textContent).toContain("Unknown");
    expect(routeRows[1]?.textContent).toContain("did not report the claude_gpt_5hr quota window");

    expect(routeRows[2]?.textContent).toContain("gemini_weekly");
    expect(routeRows[2]?.textContent).toContain("Unknown");
    expect(routeRows[2]?.textContent).toContain("did not report the gemini_weekly quota window");

    expect(routeRows[3]?.textContent).toContain("claude_gpt_weekly");
    expect(routeRows[3]?.textContent).toContain("Unknown");
    expect(routeRows[3]?.textContent).toContain("did not report the claude_gpt_weekly quota window");

    // Generic weekly and model labels ignored from display output
    expect(container.textContent).not.toContain("arbitrary_weekly");
    expect(container.textContent).not.toContain("gemini-1.5-pro");
    const weeklyLabels = routeRows.map(r => r.querySelector("span.truncate")?.textContent);
    expect(weeklyLabels).toEqual(["gemini_5hr", "claude_gpt_5hr", "gemini_weekly", "claude_gpt_weekly"]);
  });

  it("collapses duplicate explicit weekly rows to exactly one unknown row", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    const weeklyRow = {
      route: "gemini_weekly",
      provider: "antigravity",
      account: "antigravity-account",
      usage: 10,
      limit: 100,
      remaining: 90,
      remaining_percent: 90,
      unit: "requests",
      reset_at: "2026-09-20T00:00:00Z",
      status: "reported" as const,
      source: "9router_management_api",
      detail: null,
    };
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          weeklyRow,
          { ...weeklyRow },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_weekly"));
    const weeklyRows = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'))
      .filter((row) => row.textContent?.includes("gemini_weekly"));
    expect(weeklyRows).toHaveLength(1);
    expect(weeklyRows[0]?.textContent).toContain("Unknown");
    expect(weeklyRows[0]?.textContent).toContain("Multiple explicit quota windows were reported for gemini_weekly");
    expect(weeklyRows[0]?.textContent).toContain("Used Unknown");
    expect(weeklyRows[0]?.querySelector('[role="progressbar"]')).toBeNull();
  });

  it("applies the invariant validator fail-closed to weekly and malformed numeric rows", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          // gemini_5hr: usage + remaining does not equal limit
          {
            route: "gemini_5hr",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 20,
            limit: 100,
            remaining: 50,
            remaining_percent: 50,
            unit: "requests",
            reset_at: "2026-09-13T10:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          // claude_gpt_5hr: unparseable reset_at
          {
            route: "claude_gpt_5hr",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 20,
            limit: 100,
            remaining: 80,
            remaining_percent: 80,
            unit: "requests",
            reset_at: "invalid-date-string",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          // gemini_weekly: remaining_percent contradicts remaining/limit
          {
            route: "gemini_weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 10,
            limit: 100,
            remaining: 90,
            remaining_percent: 25,
            unit: "requests",
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          // claude_gpt_weekly: usage > limit
          {
            route: "claude_gpt_weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 150,
            limit: 100,
            remaining: 0,
            remaining_percent: 0,
            unit: "requests",
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));

    const routeRows = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'));
    expect(routeRows).toHaveLength(4);

    for (const row of routeRows) {
      expect(row.textContent).toContain("Unknown");
      expect(row.textContent).toContain("unusable quota data");
      expect(row.querySelector('[role="progressbar"]')).toBeNull();
      expect(row.textContent).toContain("Used Unknown");
      expect(row.textContent).toContain("Limit Unknown");
      expect(row.textContent).toContain("Remaining Unknown");
    }
  });

  it("prevents gemini_weekly from being reclassified as a model row for gemini_5hr", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          {
            route: "gemini_weekly",
            provider: "antigravity",
            account: "antigravity-account",
            usage: 50,
            limit: 100,
            remaining: 50,
            remaining_percent: 50,
            unit: "requests",
            reset_at: "2026-09-20T00:00:00Z",
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_weekly"));

    const routeRows = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'));
    const gemini5hrRow = routeRows.find((r) => r.textContent?.includes("gemini_5hr"));
    const geminiWeeklyRow = routeRows.find((r) => r.textContent?.includes("gemini_weekly"));

    // gemini_5hr must NOT take gemini_weekly's values: it should be synthesized unknown!
    expect(gemini5hrRow?.textContent).toContain("Unknown");
    expect(gemini5hrRow?.textContent).toContain("did not report the gemini_5hr quota window");
    expect(gemini5hrRow?.textContent).not.toContain("Used 50");

    // gemini_weekly is reported
    expect(geminiWeeklyRow?.textContent).toContain("Reported");
    expect(geminiWeeklyRow?.textContent).toContain("Used 50");
  });

  it("groups accounts and generates React keys without delimiter collision", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          // Account A: provider="alpha:beta", account="gamma"
          {
            route: "daily",
            provider: "alpha:beta",
            account: "gamma",
            usage: 1,
            limit: 10,
            remaining: 9,
            remaining_percent: 90,
            unit: null,
            reset_at: null,
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          // Account B: provider="alpha", account="beta:gamma" (would collide under a simple colon delimiter)
          {
            route: "daily",
            provider: "alpha",
            account: "beta:gamma",
            usage: 2,
            limit: 10,
            remaining: 8,
            remaining_percent: 80,
            unit: null,
            reset_at: null,
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
          // Account C: provider="alpha", account="beta\u0000gamma" (would collide under NUL delimiter)
          {
            route: "daily",
            provider: "alpha",
            account: "beta\u0000gamma",
            usage: 3,
            limit: 10,
            remaining: 7,
            remaining_percent: 70,
            unit: null,
            reset_at: null,
            status: "reported",
            source: "9router_management_api",
            detail: null,
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("alpha:beta"));

    const cards = Array.from(
      container.querySelectorAll<HTMLDetailsElement>('[data-testid="usage-quota-account-card"]')
    );
    // All 3 accounts must remain separate distinct cards without merging
    expect(cards).toHaveLength(3);
    const cardTexts = cards.map((c) => c.textContent);
    expect(cardTexts.some((t) => t?.includes("alpha:beta") && t?.includes("gamma"))).toBe(true);
    expect(cardTexts.some((t) => t?.includes("alpha") && t?.includes("beta:gamma"))).toBe(true);
    expect(cardTexts.some((t) => t?.includes("alpha") && t?.includes("beta\u0000gamma"))).toBe(true);
  });

  it("permits Antigravity family routes with remaining: null when remaining_percent is valid and consistent", async () => {
    const { default: UsageQuotaPage } = await import("./UsageQuotaPage");
    const modelRoute = (route: string) => ({
      route,
      provider: "antigravity",
      account: "antigravity-account",
      usage: 120,
      limit: 1000,
      remaining: null,
      remaining_percent: 88,
      unit: null,
      reset_at: "2026-09-13T10:52:41Z",
      status: "reported" as const,
      source: "9router_management_api",
      detail: null,
    });
    apiMocks.getUsageQuota.mockResolvedValueOnce({
      providers: [{
        provider: "9router",
        source: "9router_management_api",
        fetched_at: "2026-09-13T00:00:00Z",
        title: "9Router usage & quota",
        plan: null,
        windows: [],
        details: [],
        unavailable_reason: null,
        available: true,
        partial: false,
        scope: "profile:quota-test",
        stale: false,
        routes: [
          modelRoute("gemini-3.6-flash-high"),
          modelRoute("claude-opus-4-6-thinking"),
          {
            ...modelRoute("gemini_weekly"),
            usage: 12,
            remaining: null,
            remaining_percent: 98.8,
            reset_at: "2026-09-20T00:00:00Z",
          },
          {
            ...modelRoute("claude_gpt_weekly"),
            usage: 20,
            remaining: null,
            remaining_percent: 98,
            reset_at: "2026-09-20T00:00:00Z",
          },
        ],
      }],
    });
    await render(<MemoryRouter><UsageQuotaPage /></MemoryRouter>);

    await vi.waitFor(() => expect(container.textContent).toContain("gemini_5hr"));
    expect(container.textContent).toContain("claude_gpt_5hr");
    expect(container.textContent).toContain("gemini_weekly");
    expect(container.textContent).toContain("claude_gpt_weekly");

    const routeRows = Array.from(container.querySelectorAll('[data-testid="usage-quota-route"]'));
    expect(routeRows).toHaveLength(4);

    for (const row of routeRows) {
      expect(row.textContent).toContain("Reported");
      expect(row.querySelector('[role="progressbar"]')).not.toBeNull();
      expect(row.textContent).not.toContain("unusable quota data");
    }

    const gemini5hr = routeRows.find((r) => r.textContent?.includes("gemini_5hr"));
    expect(gemini5hr?.textContent).toContain("Used 120");
    expect(gemini5hr?.textContent).toContain("88% remaining");
    expect(gemini5hr?.textContent).toContain("Remaining Unknown");
  });
});
