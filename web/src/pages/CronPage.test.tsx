// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { I18nProvider } from "@/i18n";
import type { CronJobAnalytics } from "@/lib/api";
import CronPage from "./CronPage";

const mocks = vi.hoisted(() => ({
  api: {
    getCronJobs: vi.fn(),
    getCronAnalytics: vi.fn(),
    getProfiles: vi.fn(),
    getCronDeliveryTargets: vi.fn(),
    getSkills: vi.fn(),
    getToolsets: vi.fn(),
    getModelOptions: vi.fn(),
  },
  setEnd: vi.fn(),
}));
vi.mock("@/lib/api", () => ({ api: mocks.api }));
vi.mock("@/contexts/usePageHeader", () => ({ usePageHeader: () => ({ setEnd: mocks.setEnd }) }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));
vi.mock("@/components/AutomationBlueprints", () => ({ AutomationBlueprints: () => null }));
vi.mock("@/components/DeleteConfirmDialog", () => ({ DeleteConfirmDialog: () => null }));

let container: HTMLDivElement;
let root: Root;
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function usage(jobId: string, historyLimited: boolean): CronJobAnalytics {
  return {
    job_id: jobId,
    profile: "default",
    period_days: 30,
    usage_runs: 2,
    total_tokens: 1000,
    avg_tokens_per_run: 500,
    total_cost_usd: 0.2,
    avg_cost_usd_per_run: 0.1,
    actual_cost_runs: 2,
    estimated_cost_runs: 0,
    unknown_cost_runs: 0,
    incomplete_runs: 0,
    attempts: 2,
    completed: 1,
    failed: 1,
    unknown: 0,
    running: 0,
    manual_runs: 1,
    scheduled_runs: 1,
    retry_attempts: 1,
    history_limited: historyLimited,
  };
}

async function renderPage() {
  await act(async () => root.render(<I18nProvider><CronPage /></I18nProvider>));
}

beforeEach(() => {
  vi.clearAllMocks();
  const storage = new Map<string, string>();
  vi.stubGlobal("localStorage", {
    getItem: (key: string) => storage.get(key) ?? null,
    setItem: (key: string, value: string) => storage.set(key, value),
    clear: () => storage.clear(),
  });
  mocks.api.getCronJobs.mockResolvedValue([
    { id: "limited", profile: "default", name: "Retained job", schedule: { kind: "interval", minutes: 60 } },
    { id: "complete", profile: "default", name: "Complete job", schedule: { kind: "interval", minutes: 60 } },
  ]);
  mocks.api.getCronAnalytics.mockResolvedValue({ period_days: 30, jobs: [usage("limited", true), usage("complete", false)] });
  mocks.api.getProfiles.mockResolvedValue({ profiles: [] });
  mocks.api.getCronDeliveryTargets.mockResolvedValue({ targets: [] });
  mocks.api.getSkills.mockResolvedValue([]);
  mocks.api.getToolsets.mockResolvedValue([]);
  mocks.api.getModelOptions.mockResolvedValue(null);
  vi.stubGlobal("ResizeObserver", class { disconnect() {} observe() {} unobserve() {} });
  vi.stubGlobal("matchMedia", () => ({ addEventListener() {}, matches: false, media: "", removeEventListener() {} }));
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
  localStorage.clear();
  vi.unstubAllGlobals();
});

describe("cron analytics coverage", () => {
  it("warns only on the limited job about execution outcome, manual and retry counts, not usage totals", async () => {
    await renderPage();
    const warning = container.querySelector('[data-testid="cron-history-limited-default:limited"]');
    expect(warning).not.toBeNull();
    expect(warning?.textContent).toMatch(/execution outcome, manual, and retry counts may be incomplete/i);
    expect(warning?.textContent).not.toMatch(/token|cost/i);
    expect(container.querySelector('[data-testid="cron-history-limited-default:complete"]')).toBeNull();
    expect(container.querySelector('[data-testid="cron-usage-default:limited"]')?.textContent).toContain("Total tokens: 1,000");
    expect(container.querySelector('[data-testid="cron-usage-default:limited"]')?.textContent).toContain("Total cost: $0.20");
  });

  it("renders analytics and the coverage warning with the real Turkish provider", async () => {
    localStorage.setItem("hermes-locale", "tr");
    await renderPage();
    expect(container.textContent).toContain("Zamanlanmış Görevler");
    expect(container.textContent).toContain("Usage period");
    expect(container.querySelector('[data-testid="cron-history-limited-default:limited"]')?.textContent).toMatch(/counts may be incomplete/i);
    expect(document.documentElement.lang).toBe("tr");
  });

  it("keeps jobs usable when analytics is unavailable", async () => {
    mocks.api.getCronAnalytics.mockRejectedValue(new Error("analytics unavailable"));
    await renderPage();
    expect(container.textContent).toContain("Retained job");
    expect(container.querySelector('[data-testid^="cron-usage-"]')).toBeNull();
    expect(container.querySelector('[data-testid^="cron-history-limited-"]')).toBeNull();
  });
});
