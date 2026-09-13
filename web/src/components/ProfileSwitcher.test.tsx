// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  getProfiles: vi.fn(),
  getActiveProfile: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: apiMocks,
  // ProfileProvider mirrors its selection into the api module.
  setManagementProfile: vi.fn(),
  getManagementProfile: vi.fn(() => ""),
}));

function profileFixture(overrides: Record<string, unknown> = {}) {
  return {
    name: "default",
    path: "/tmp/hermes-home/default",
    is_default: true,
    model: null,
    provider: null,
    has_env: false,
    skill_count: 0,
    gateway_running: false,
    description: "",
    description_auto: false,
    display_name: "",
    distribution_name: null,
    distribution_version: null,
    distribution_source: null,
    has_alias: false,
    ...overrides,
  };
}

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

const trigger = () => document.querySelector('button[role="combobox"]');

async function renderSwitcher() {
  const [{ ProfileSwitcher }, { I18nProvider }, { ProfileProvider }] = await Promise.all([
    import("./ProfileSwitcher"),
    import("@/i18n"),
    import("@/contexts/ProfileProvider"),
  ]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <I18nProvider>
        <MemoryRouter>
          <ProfileProvider>
            <ProfileSwitcher />
          </ProfileProvider>
        </MemoryRouter>
      </I18nProvider>,
    ),
  );
}

function click(el: Element | null) {
  if (!el) throw new Error("element not rendered");
  el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
}

const optionTexts = () =>
  Array.from(document.querySelectorAll('[role="option"]')).map((el) =>
    el.textContent?.replace("✓", "").trim(),
  );

describe("ProfileSwitcher display_name labels (#103251)", () => {
  beforeEach(() => {
    for (const fn of Object.values(apiMocks)) fn.mockReset();
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 500 })));
    vi.stubGlobal("ResizeObserver", class { disconnect() {} observe() {} unobserve() {} });
    // The listbox scroll path touches Element.prototype.scrollIntoView, absent in jsdom.
    Element.prototype.scrollIntoView = () => {};
  });

  afterEach(async () => {
    await act(async () => root?.unmount());
    container?.remove();
    vi.unstubAllGlobals();
  });

  it("labels the dashboard's own profile with its display name, not the raw id", async () => {
    apiMocks.getProfiles.mockResolvedValue({
      profiles: [
        profileFixture({ name: "default", display_name: "Legal writing" }),
        profileFixture({ name: "work", is_default: false }),
      ],
    });
    apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
    await renderSwitcher();
    await waitFor(() => Boolean(trigger()));

    expect(trigger()?.textContent).toContain("this dashboard (Legal writing (default))");
  });

  it("falls back to the bare id when the current profile has no display name", async () => {
    apiMocks.getProfiles.mockResolvedValue({
      profiles: [profileFixture(), profileFixture({ name: "work", is_default: false })],
    });
    apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
    await renderSwitcher();
    await waitFor(() => Boolean(trigger()));

    expect(trigger()?.textContent).toContain("this dashboard (default)");
  });

  it("labels other profiles' listbox options with their display names", async () => {
    apiMocks.getProfiles.mockResolvedValue({
      profiles: [
        profileFixture(),
        profileFixture({ name: "work", is_default: false, display_name: "Work", path: "/tmp/hermes-home/work" }),
      ],
    });
    apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
    await renderSwitcher();
    await waitFor(() => Boolean(trigger()));

    await act(async () => {
      click(trigger());
    });
    expect(optionTexts()).toContain("Work (work)");
  });
});
