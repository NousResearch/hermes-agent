// @vitest-environment jsdom
/**
 * Render test for the Cron page's category grouping.
 *
 * The grouping helper (cron-groups.test.ts) is pure and cannot catch the bug this
 * file exists for: a hook declared *after* the page's `loading` early return, which
 * made the hook count differ between renders and crashed the page with
 * "Rendered more hooks than during the previous render". Only rendering the
 * component finds that.
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getCronJobs: vi.fn(),
  getProfiles: vi.fn(async () => ({ profiles: [{ name: "default" }] })),
  getSkills: vi.fn(async () => []),
  getToolsets: vi.fn(async () => []),
  getModelOptions: vi.fn(async () => ({ models: [], providers: [] })),
  getCronDeliveryTargets: vi.fn(async () => ({ targets: [{ id: "local", name: "Local" }] })),
  createCronJob: vi.fn(async () => ({})),
  updateCronJob: vi.fn(async () => ({})),
  deleteCronJob: vi.fn(async () => ({})),
  pauseCronJob: vi.fn(async () => ({})),
  resumeCronJob: vi.fn(async () => ({})),
  triggerCronJob: vi.fn(async () => ({})),
}));

vi.mock("@/lib/api", () => ({ api: apiMocks }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));
vi.mock("@nous-research/ui/hooks/use-toast", () => ({
  useToast: () => ({ toast: null, showToast: vi.fn() }),
}));

import { MemoryRouter } from "react-router";
import { I18nProvider } from "@/i18n";
import { PageHeaderProvider } from "@/contexts/PageHeaderProvider";
import CronPage from "./CronPage";

const JOBS = [
  { id: "a", name: "piano-practice", prompt: "p", schedule: "every 1d", category: "Family" },
  { id: "b", name: "restic-full", prompt: "p", schedule: "every 1d", category: "Backups" },
  { id: "c", name: "loose-job", prompt: "p", schedule: "every 1d" },
];

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
  apiMocks.getCronJobs.mockResolvedValue(JOBS);
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
  vi.clearAllMocks();
});

/** Render the page the way the app does (the providers it expects), let the api
 *  promises settle, and return the rendered text. */
async function renderCronPage(): Promise<string> {
  await act(async () => {
    root.render(
      <MemoryRouter>
        <I18nProvider>
          <PageHeaderProvider pluginTabs={[]}>
            <CronPage />
          </PageHeaderProvider>
        </I18nProvider>
      </MemoryRouter>,
    );
  });
  return container.textContent ?? "";
}

function headings(): string[] {
  return Array.from(container.querySelectorAll('[data-testid="cron-category-header"]')).map(
    (el) => el.textContent ?? "",
  );
}

function headerFor(category: string): HTMLElement | undefined {
  return Array.from(
    container.querySelectorAll<HTMLElement>('[data-testid="cron-category-header"]'),
  ).find((el) => (el.textContent ?? "").includes(category));
}

async function click(el: HTMLElement): Promise<void> {
  await act(async () => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  });
}

/** Flip the Ungrouped | By category control to the grouped view. */
async function switchToGrouped(): Promise<void> {
  const toggle = Array.from(container.querySelectorAll("button")).find((b) =>
    (b.textContent ?? "").includes("By category"),
  );
  expect(toggle).toBeTruthy();
  expect(headings()).toHaveLength(0); // ungrouped to start with
  await click(toggle!);
}

describe("CronPage category grouping", () => {
  it("renders the job list without crashing", async () => {
    const text = await renderCronPage();
    expect(text).toContain("piano-practice");
    expect(text).toContain("loose-job");
  });

  it("offers the grouping toggle once a job carries a label, and renders headings when switched on", async () => {
    await renderCronPage();
    await switchToGrouped();

    const rendered = headings();
    expect(rendered.length).toBe(3);
    // labelled groups sort alphabetically, the unlabelled bucket always last
    expect(rendered[0]).toContain("Backups");
    expect(rendered[1]).toContain("Family");
    expect(rendered[2]).toContain("Uncategorised");
    expect(rendered[2]).toContain("(1)");
  });

  it("folds a category away when its header is clicked, and unfolds it again", async () => {
    await renderCronPage();
    await switchToGrouped();

    const family = headerFor("Family");
    expect(family?.getAttribute("aria-expanded")).toBe("true");
    expect(container.textContent).toContain("piano-practice");

    await click(family!);
    expect(headerFor("Family")?.getAttribute("aria-expanded")).toBe("false");
    expect(container.textContent).not.toContain("piano-practice");
    // other groups are untouched, and the header stays visible with its count
    expect(container.textContent).toContain("restic-full");
    expect(headerFor("Family")?.textContent).toContain("(1)");

    await click(headerFor("Family")!);
    expect(headerFor("Family")?.getAttribute("aria-expanded")).toBe("true");
    expect(container.textContent).toContain("piano-practice");
  });
});
