// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { AnalyticsResponse } from "@/lib/api-analytics";
import { I18nProvider } from "@/i18n";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import AnalyticsPage from "./AnalyticsPage";

const apiMocks = vi.hoisted(() => ({ getConfig: vi.fn(), getAnalytics: vi.fn() }));
vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const provider = (name = "openrouter", overrides: Record<string, number> = {}) => ({
  provider: name,
  input_tokens: 2400,
  output_tokens: 300,
  cache_read_tokens: 100,
  cache_write_tokens: 50,
  reasoning_tokens: 20,
  estimated_cost: 0.125,
  actual_cost: 0.25,
  sessions: 2,
  api_calls: 8,
  ...overrides,
});

function response(extra: Record<string, unknown> = {}): AnalyticsResponse {
  return {
    daily: [],
    by_model: [],
    totals: {
      total_input: 0,
      total_output: 0,
      total_cache_read: 0,
      total_reasoning: 0,
      total_estimated_cost: 0,
      total_actual_cost: 0,
      total_sessions: 0,
      total_api_calls: 0,
    },
    skills: {
      summary: { total_skill_loads: 0, total_skill_edits: 0, total_skill_actions: 0, distinct_skills_used: 0 },
      top_skills: [],
    },
    ...extra,
  };
}

let container: HTMLDivElement;
let root: Root;

async function renderPage() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(
    <I18nProvider>
      <MemoryRouter initialEntries={["/analytics"]}>
        <PageHeaderProvider pluginTabs={[]}>
          <AnalyticsPage />
        </PageHeaderProvider>
      </MemoryRouter>
    </I18nProvider>,
  ));
}

async function clickButton(label: string) {
  const target = Array.from(container.querySelectorAll("button")).find(
    (button) => button.textContent?.trim() === label || button.getAttribute("aria-label") === label,
  );
  expect(target, `Button ${label} should be available`).toBeDefined();
  await act(async () => target!.click());
}

function providerTable() {
  return Array.from(container.querySelectorAll("table")).find(
    (table) => table.querySelector("th")?.textContent === "Provider",
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  apiMocks.getConfig.mockReset().mockResolvedValue({ dashboard: { show_token_analytics: true } });
  apiMocks.getAnalytics.mockReset().mockResolvedValue(response());
  localStorage.clear();
  vi.stubGlobal("matchMedia", () => ({ addEventListener() {}, matches: false, removeEventListener() {} }));
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("provider usage analytics", () => {
  it.each([false, undefined, "true"])("does not fetch or expose usage without explicit opt-in (%s)", async (setting) => {
    apiMocks.getConfig.mockResolvedValue({ dashboard: { show_token_analytics: setting } });
    apiMocks.getAnalytics.mockResolvedValue(response({ by_provider: [provider()] }));
    await renderPage();
    expect(apiMocks.getAnalytics).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Token analytics hidden");
    expect(providerTable()).toBeUndefined();
    expect(container.querySelector('button[aria-label="Refresh"]')).toBeNull();
  });

  it("keeps analytics hidden when configuration cannot be read", async () => {
    apiMocks.getConfig.mockRejectedValue(new Error("config unavailable"));
    await renderPage();
    expect(apiMocks.getAnalytics).not.toHaveBeenCalled();
    expect(container.textContent).toContain("Token analytics hidden");
  });

  it("renders old responses without by_provider and keeps existing model data", async () => {
    apiMocks.getAnalytics.mockResolvedValue(response({ by_model: [{
      model: "legacy-model", input_tokens: 120, output_tokens: 15, estimated_cost: 0.01, sessions: 1, api_calls: 2,
    }] }));
    await renderPage();
    expect(apiMocks.getAnalytics).toHaveBeenCalledWith(30);
    expect(container.textContent).toContain("legacy-model");
    expect(providerTable()).toBeUndefined();
    expect(container.textContent).not.toContain("No usage data for this period");
  });

  it.each([{}, { by_provider: [] }])("shows the empty state with no recorded usage (%j)", async (extra) => {
    apiMocks.getAnalytics.mockResolvedValue(response(extra));
    await renderPage();
    expect(container.textContent).toContain("No usage data for this period");
    expect(providerTable()).toBeUndefined();
  });

  it("renders auxiliary-only provider rows without deriving them from unchanged summary totals", async () => {
    apiMocks.getAnalytics.mockResolvedValue(response({ by_provider: [provider()] }));
    await renderPage();
    const table = providerTable();
    expect(table).toBeDefined();
    expect(Array.from(table!.querySelectorAll("tbody td"), (cell) => cell.textContent)).toEqual([
      "openrouter", "2", "8", "2.4K / 300", "$0.1250", "$0.2500",
    ]);
    expect(container.textContent).toContain("Local records, not authoritative provider billing.");
    expect(container.textContent).toContain("recorded primary and auxiliary calls");
    expect(container.textContent).toContain("Each session counts once per provider");
    expect(container.textContent).not.toContain("No usage data for this period");
  });

  it("preserves per-provider session counts and sorts rows by logged cost or selected column", async () => {
    apiMocks.getAnalytics.mockResolvedValue(response({ by_provider: [
      provider("unknown", { sessions: 1, actual_cost: 0 }),
      provider("anthropic", { sessions: 1, actual_cost: 0.8 }),
      provider("openrouter", { sessions: 1, actual_cost: 0.3 }),
    ] }));
    await renderPage();
    const rows = () => Array.from(providerTable()!.querySelectorAll("tbody tr"), (row) =>
      Array.from(row.querySelectorAll("td"), (cell) => cell.textContent));
    expect(rows().map((row) => row[0])).toEqual(["anthropic", "openrouter", "unknown"]);
    expect(rows().map((row) => row[1])).toEqual(["1", "1", "1"]);
    const header = Array.from(providerTable()!.querySelectorAll("th")).find((cell) => cell.textContent === "Provider")!;
    await act(async () => header.click());
    expect(rows().map((row) => row[0])).toEqual(["unknown", "openrouter", "anthropic"]);
    await act(async () => header.click());
    expect(rows().map((row) => row[0])).toEqual(["anthropic", "openrouter", "unknown"]);
  });

  it("refreshes after an initial error and retains loaded data on a failed refresh", async () => {
    apiMocks.getAnalytics.mockRejectedValueOnce(new Error("usage unavailable"));
    await renderPage();
    expect(container.textContent).toContain("usage unavailable");
    expect(providerTable()).toBeUndefined();
    apiMocks.getAnalytics.mockResolvedValueOnce(response({ by_provider: [provider()] }));
    await clickButton("Refresh");
    expect(providerTable()).toBeDefined();
    expect(container.textContent).not.toContain("usage unavailable");
    apiMocks.getAnalytics.mockRejectedValueOnce(new Error("refresh failed"));
    await clickButton("Refresh");
    expect(container.textContent).toContain("refresh failed");
    expect(providerTable()!.textContent).toContain("openrouter");
    apiMocks.getAnalytics.mockResolvedValueOnce(response({ by_provider: [] }));
    await clickButton("Refresh");
    expect(container.textContent).not.toContain("refresh failed");
    expect(providerTable()).toBeUndefined();
    expect(container.textContent).toContain("No usage data for this period");
  });

  it("reloads provider rows for the selected range and refreshes the same range", async () => {
    apiMocks.getAnalytics.mockImplementation(async (days: number) => response({
      by_provider: days === 90 ? [] : [provider(`provider-${days}`)],
    }));
    await renderPage();
    expect(providerTable()!.textContent).toContain("provider-30");
    await clickButton("7d");
    expect(apiMocks.getAnalytics).toHaveBeenLastCalledWith(7);
    expect(providerTable()!.textContent).toContain("provider-7");
    expect(providerTable()!.textContent).not.toContain("provider-30");
    await clickButton("Refresh");
    expect(apiMocks.getAnalytics).toHaveBeenLastCalledWith(7);
    await clickButton("90d");
    expect(apiMocks.getAnalytics).toHaveBeenLastCalledWith(90);
    expect(providerTable()).toBeUndefined();
    expect(container.textContent).toContain("No usage data for this period");
  });

  it("localizes the provider table and scope warning", async () => {
    localStorage.setItem("hermes-locale", "ja");
    apiMocks.getAnalytics.mockResolvedValue(response({ by_provider: [provider()] }));
    await renderPage();
    expect(container.textContent).toContain("openrouter");
    expect(container.textContent).not.toContain("Local records, not authoritative provider billing.");
    expect(container.querySelector("table th")?.textContent).toBe("プロバイダー");
  });
});
