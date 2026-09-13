// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useProfileScope } from "@/contexts/useProfileScope";

/**
 * Scope-alignment regression tests.
 *
 * The provider's on-load alignment (switcher follows the sticky active
 * profile) must only run on the MACHINE dashboard (current === "default").
 * On an isolated per-profile dashboard (`<profile> dashboard --isolated`),
 * the sticky active profile belongs to a different agent — aligning to it
 * silently retargets every chat to that profile (banner, TUI env, and model
 * all resolve from the wrong profile).
 */

const apiMocks = vi.hoisted(() => ({
  getProfiles: vi.fn(async () => ({
    profiles: [{ name: "default" }, { name: "priyanshi" }],
  })),
  getActiveProfile: vi.fn(async () => ({ active: "default", current: "priyanshi" })),
  setManagementProfile: vi.fn(),
}));

const routerMocks = vi.hoisted(() => {
  const state = { searchParams: new URLSearchParams() };
  const setSearchParams = vi.fn(
    (
      update: (prev: URLSearchParams) => URLSearchParams,
    ) => {
      state.searchParams = update(state.searchParams);
      return state.searchParams;
    },
  );
  return { state, setSearchParams };
});

vi.mock("@/lib/api", () => ({
  api: {
    getProfiles: apiMocks.getProfiles,
    getActiveProfile: apiMocks.getActiveProfile,
  },
  setManagementProfile: apiMocks.setManagementProfile,
}));

vi.mock("react-router", () => ({
  useSearchParams: () => [routerMocks.state.searchParams, routerMocks.setSearchParams],
  useLocation: () => ({ pathname: "/" }),
}));

let container: HTMLDivElement | undefined;
let root: Root | undefined;

function Probe() {
  const scope = useProfileScope();
  return (
    <div
      id="probe"
      data-profile={scope.profile}
      data-current={scope.currentProfile}
      data-profiles={scope.profiles.join(",")}
    />
  );
}

function renderTree(tree: ReactNode) {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  act(() => {
    root!.render(tree);
  });
}

async function flushEffects() {
  // Effects fire after mount; the provider's Promise.all chain settles on
  // microtasks. Two act rounds are enough for fetch -> then -> setState.
  await act(async () => {
    await Promise.resolve();
  });
  await act(async () => {
    await Promise.resolve();
  });
}

function probeData(): { profile: string; current: string } {
  const el = document.getElementById("probe");
  if (!el) throw new Error("probe not mounted");
  return {
    profile: el.getAttribute("data-profile") ?? "",
    current: el.getAttribute("data-current") ?? "",
  };
}

describe("ProfileProvider scope alignment", () => {
  beforeEach(() => {
    routerMocks.state.searchParams = new URLSearchParams();
    routerMocks.setSearchParams.mockClear();
    apiMocks.setManagementProfile.mockClear();
  });

  afterEach(() => {
    act(() => {
      root?.unmount();
    });
    container?.remove();
    container = undefined;
    root = undefined;
  });

  it("keeps an isolated profile dashboard scoped to its own profile even when the sticky active profile differs", async () => {
    apiMocks.getActiveProfile.mockResolvedValue({
      active: "default",
      current: "priyanshi",
    });

    const { ProfileProvider } = await import("@/contexts/ProfileProvider");
    renderTree(
      <ProfileProvider>
        <Probe />
      </ProfileProvider>,
    );
    await flushEffects();

    const { profile, current } = probeData();
    expect(current).toBe("priyanshi");
    // "" = the dashboard's own profile. Alignment to the machine's sticky
    // active profile ("default") must NOT happen on an isolated dashboard.
    expect(profile).toBe("");
    // No retargeting call may fire for the wrong profile either.
    expect(apiMocks.setManagementProfile).not.toHaveBeenCalledWith("default");
  });

  it("does not align a dashboard running on a nonstandard HERMES_HOME (current 'custom'), even though launch routing groups it with machine dashboards", async () => {
    // Launch routing treats ("default", "custom") as machine-dashboard
    // profile names, but a dashboard whose HERMES_HOME is a nonstandard
    // path serves that home's own data — aligning its scope to the
    // machine-global sticky active profile would retarget it the same way
    // isolated dashboards were retargeted (see issue #96712). The guard
    // deliberately excludes "custom"; this test pins that exclusion so
    // widening the guard back to the launch-routing tuple becomes a
    // conscious decision.
    apiMocks.getActiveProfile.mockResolvedValue({
      active: "default",
      current: "custom",
    });

    const { ProfileProvider } = await import("@/contexts/ProfileProvider");
    renderTree(
      <ProfileProvider>
        <Probe />
      </ProfileProvider>,
    );
    await flushEffects();

    const { profile, current } = probeData();
    expect(current).toBe("custom");
    expect(profile).toBe("");
    expect(apiMocks.setManagementProfile).not.toHaveBeenCalledWith("default");
  });

  it("still aligns the switcher to the sticky active profile on the machine dashboard", async () => {
    apiMocks.getActiveProfile.mockResolvedValue({
      active: "priyanshi",
      current: "default",
    });

    const { ProfileProvider } = await import("@/contexts/ProfileProvider");
    renderTree(
      <ProfileProvider>
        <Probe />
      </ProfileProvider>,
    );
    await flushEffects();

    const { profile, current } = probeData();
    expect(current).toBe("default");
    expect(profile).toBe("priyanshi");
  });

  it("lets a ?profile= deep link win on an isolated dashboard", async () => {
    apiMocks.getActiveProfile.mockResolvedValue({
      active: "default",
      current: "priyanshi",
    });
    routerMocks.state.searchParams = new URLSearchParams("profile=default");

    const { ProfileProvider } = await import("@/contexts/ProfileProvider");
    renderTree(
      <ProfileProvider>
        <Probe />
      </ProfileProvider>,
    );
    await flushEffects();

    const { profile } = probeData();
    expect(profile).toBe("default");
  });
});
