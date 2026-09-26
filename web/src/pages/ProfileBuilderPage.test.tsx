// @vitest-environment jsdom
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, Route, Routes, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  createProfile: vi.fn(),
  getActiveProfile: vi.fn(),
  getModelOptions: vi.fn(),
  getProfiles: vi.fn(),
  getSkills: vi.fn(),
}));

vi.mock("@/lib/api", () => ({
  api: apiMocks,
  // ProfileProvider mirrors its selection into the api module.
  setManagementProfile: vi.fn(),
  getManagementProfile: vi.fn(() => ""),
}));

const MODEL_ERROR =
  "Anthropic is not connected: no API key or login was found for it.";

let container: HTMLDivElement;
let root: Root;
let routeState: unknown;
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

function buttonByText(text: string) {
  const el = [...document.querySelectorAll("button")].find(
    (b) => b.textContent?.trim() === text,
  );
  if (!el) throw new Error(`no button "${text}"`);
  return el;
}

async function click(el: Element) {
  await act(async () => {
    el.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  });
}

function LocationProbe() {
  const { state } = useLocation();
  useEffect(() => {
    routeState = state;
  }, [state]);
  return null;
}

async function renderBuilder() {
  const [
    { default: ProfileBuilderPage },
    { default: ProfilesPage },
    { I18nProvider },
    { ProfileProvider },
    { PageHeaderProvider },
  ] = await Promise.all([
    import("./ProfileBuilderPage"),
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
        <MemoryRouter initialEntries={["/profiles/new"]}>
          <ProfileProvider>
            <PageHeaderProvider pluginTabs={[]}>
              <LocationProbe />
              <Routes>
                <Route path="/profiles" element={<ProfilesPage />} />
                <Route path="/profiles/new" element={<ProfileBuilderPage />} />
              </Routes>
            </PageHeaderProvider>
          </ProfileProvider>
        </MemoryRouter>
      </I18nProvider>,
    ),
  );
}

async function createFromBuilder({ pickModel }: { pickModel: boolean }) {
  await renderBuilder();
  const input = document.querySelector("#pb-name") as HTMLInputElement;
  await act(async () => {
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    setValue.call(input, "coder");
    input.dispatchEvent(new Event("input", { bubbles: true }));
  });
  if (pickModel) {
    await click(buttonByText("2. Model"));
    await waitFor(() =>
      [...document.querySelectorAll("button")].some(
        (b) => b.textContent?.trim() === "Anthropic · claude-sonnet-4-5",
      ),
    );
    await click(buttonByText("Anthropic · claude-sonnet-4-5"));
  }
  await click(buttonByText("5. Review"));
  await click(buttonByText("Create profile"));
  // The builder navigates to /profiles once the create resolves.
  await waitFor(() => !document.querySelector("#pb-name") && apiMocks.getProfiles.mock.calls.length > 0);
}

const toastText = () => document.querySelector('[role="status"][aria-live="polite"]')?.textContent ?? "";

beforeEach(() => {
  for (const fn of Object.values(apiMocks)) fn.mockReset();
  apiMocks.getProfiles.mockResolvedValue({ profiles: [] });
  apiMocks.getActiveProfile.mockResolvedValue({ current: "default", active: "default" });
  apiMocks.getSkills.mockResolvedValue([]);
  apiMocks.getModelOptions.mockResolvedValue({
    providers: [{ slug: "anthropic", name: "Anthropic", models: ["claude-sonnet-4-5"] }],
  });
  vi.stubGlobal("ResizeObserver", class { disconnect() {} observe() {} unobserve() {} });
  vi.stubGlobal("matchMedia", () => ({ addEventListener() {}, matches: false, media: "", removeEventListener() {} }));
  routeState = undefined;
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
  delete window.__HERMES_DASHBOARD_PROFILE__;
});

describe("ProfileBuilderPage create result", () => {
  it("tells the user on the Profiles page why the picked model was not set", async () => {
    // A dashboard that names its own profile: ProfileProvider re-asserts
    // ?profile= on the bare /profiles navigation.
    window.__HERMES_DASHBOARD_PROFILE__ = "default";
    apiMocks.createProfile.mockResolvedValue({
      ok: true,
      name: "coder",
      path: "/x/profiles/coder",
      model_set: false,
      model_error: MODEL_ERROR,
      hub_installs: [],
    });

    await createFromBuilder({ pickModel: true });

    expect(apiMocks.createProfile.mock.calls[0][0]).toMatchObject({
      provider: "anthropic",
      model: "claude-sonnet-4-5",
    });
    await waitFor(() => toastText() !== "");
    expect(toastText()).toContain('Profile "coder" created');
    expect(toastText()).toContain(MODEL_ERROR);
  });

  it("shows the success toast on the Profiles page it lands on, once", async () => {
    apiMocks.createProfile.mockResolvedValue({
      ok: true,
      name: "coder",
      path: "/x/profiles/coder",
      model_set: false,
      model_error: "",
      hub_installs: [{ identifier: "a/b", pid: 42 }],
    });

    await createFromBuilder({ pickModel: false });

    await waitFor(() => toastText() !== "");
    expect(toastText()).toBe('Profile "coder" created — 1 hub skill installing');
    // Back/forward onto this entry must not replay a stale "created" toast.
    expect(routeState ?? null).toBeNull();
  });
});
