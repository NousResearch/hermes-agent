// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  createProfile: vi.fn(),
  getActiveProfile: vi.fn(),
  getModelOptions: vi.fn(),
  getProfiles: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: apiMocks,
  // ProfileProvider mirrors its selection into the api module.
  setManagementProfile: vi.fn(),
  getManagementProfile: vi.fn(() => ""),
}));

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

async function click(el: Element | null | undefined) {
  if (!el) throw new Error("element not rendered");
  await act(async () => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  });
}

const byText = (scope: ParentNode, selector: string, text: string) =>
  [...scope.querySelectorAll(selector)].find((el) => el.textContent?.trim() === text);

async function renderProfilesPage() {
  const [{ default: ProfilesPage }, { I18nProvider }, { ProfileProvider }, { PageHeaderProvider }] =
    await Promise.all([
      import("./ProfilesPage"),
      import("@/i18n"),
      import("@/contexts/ProfileProvider"),
      import("@/contexts/PageHeaderProvider"),
    ]);
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <I18nProvider>
        <MemoryRouter initialEntries={["/profiles"]}>
          <ProfileProvider>
            <PageHeaderProvider pluginTabs={[]}>
              <ProfilesPage />
            </PageHeaderProvider>
          </ProfileProvider>
        </MemoryRouter>
      </I18nProvider>,
    ),
  );
  await waitFor(() => apiMocks.getProfiles.mock.calls.length > 0 && !document.querySelector('[aria-busy="true"]'));
}

beforeEach(() => {
  for (const fn of Object.values(apiMocks)) fn.mockReset();
  apiMocks.getProfiles.mockResolvedValue({ profiles: [] });
  apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
  apiMocks.getModelOptions.mockResolvedValue({
    providers: [{ slug: "anthropic", name: "Anthropic", models: ["claude-sonnet-4-5"] }],
  });
  vi.stubGlobal("ResizeObserver", class { disconnect() {} observe() {} unobserve() {} });
  vi.stubGlobal("matchMedia", () => ({ addEventListener() {}, matches: false, media: "", removeEventListener() {} }));
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("ProfilesPage create modal", () => {
  it("says why the picked model was not set", async () => {
    const reason = "Anthropic is not connected: no API key or login was found for it.";
    apiMocks.createProfile.mockResolvedValue({
      ok: true,
      name: "coder",
      path: "/x/profiles/coder",
      model_set: false,
      model_error: reason,
    });
    await renderProfilesPage();

    await click(byText(document, "button", "Create"));
    const dialog = document.querySelector('[role="dialog"]')!;
    const input = dialog.querySelector("#profile-name") as HTMLInputElement;
    await act(async () => {
      const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
      setValue.call(input, "coder");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
    const modelPicker = dialog.querySelector('#profile-model [role="combobox"]') as HTMLButtonElement;
    await waitFor(() => !modelPicker.disabled);
    await click(modelPicker);
    await click(byText(dialog, '[role="option"]', "Anthropic · claude-sonnet-4-5"));
    await click(byText(dialog, "button", "Create"));

    expect(apiMocks.createProfile.mock.calls[0][0]).toMatchObject({
      provider: "anthropic",
      model: "claude-sonnet-4-5",
    });
    await waitFor(() => Boolean(document.querySelector('[role="status"][aria-live="polite"]')));
    expect(document.querySelector('[role="status"][aria-live="polite"]')!.textContent).toContain(reason);
  });
});
