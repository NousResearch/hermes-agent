// @vitest-environment jsdom
// Regression for #146434: visible names must never become management identities.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useLocation } from "react-router";

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...original,
    api: {
      ...original.api,
      getProfiles: vi.fn(),
      getActiveProfile: vi.fn(),
    },
  };
});

import { api, getManagementProfile, setManagementProfile, type ProfileInfo } from "@/lib/api";
import { ProfileProvider } from "./ProfileProvider";
import { useProfileScope } from "./useProfileScope";
import { ProfileSwitcher } from "@/components/ProfileSwitcher";
import { ProfileScopeBanner } from "@/components/ProfileScopeBanner";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

function info(name: string, metadata: Record<string, unknown> = {}): ProfileInfo {
  return {
    name, path: `/profiles/${name}`, is_default: name === "default", model: null,
    provider: null, has_env: false, skill_count: 0, gateway_running: false,
    description: "", description_auto: false, distribution_name: null,
    distribution_version: null, distribution_source: null, has_alias: false,
    ...metadata,
  };
}

function Probe() {
  const { profile, profiles } = useProfileScope();
  const { search } = useLocation();
  return <output data-profile={profile} data-profiles={JSON.stringify(profiles)} data-search={search} />;
}

async function mount(entry = "/config") {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(
    <MemoryRouter initialEntries={[entry]}>
      <ProfileProvider>
        <ProfileSwitcher collapsed />
        <ProfileScopeBanner />
        <Probe />
      </ProfileProvider>
    </MemoryRouter>,
  ));
}

async function options() {
  await act(async () => container.querySelector<HTMLButtonElement>('[role="combobox"]')!.click());
  return Array.from(container.querySelectorAll<HTMLElement>('[role="option"]'));
}

beforeEach(() => {
  vi.mocked(api.getActiveProfile).mockResolvedValue({ current: "default", active: "default" });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  setManagementProfile("");
  vi.clearAllMocks();
});

describe("profile visible names", () => {
  it("preserves unnamed dashboard and profile fallbacks without humanizing IDs", async () => {
    vi.mocked(api.getProfiles).mockResolvedValue({ profiles: [
      info("default", { bot_title: " ", display_name: "\t" }),
      info("mybot-test-1"),
    ] });
    await mount();

    expect(container.querySelector('[role="combobox"]')?.textContent).toBe("this dashboard (default)");
    const choices = await options();
    expect(choices.map((choice) => choice.textContent)).toEqual([
      "this dashboard (default)", "mybot-test-1",
    ]);
    await act(async () => choices[1].click());
    expect(container.textContent).toContain("Managing profile “mybot-test-1”");
    expect(getManagementProfile()).toBe("mybot-test-1");
  });

  it.each(["default", "u5357-u5bab"])(
    "shows the dashboard title for an explicit own-profile URL (%s)",
    async (current) => {
      vi.mocked(api.getActiveProfile).mockResolvedValue({ current, active: current });
      vi.mocked(api.getProfiles).mockResolvedValue({ profiles: [
        info(current, { bot_title: "My agent" }), info("other"),
      ] });
      await mount(`/config?profile=${current}`);

      expect(container.querySelector('[role="combobox"]')?.textContent).toBe("this dashboard (My agent)");
      expect(container.textContent).not.toContain("Managing profile “");
      expect(getManagementProfile()).toBe(current);
      expect(container.querySelector("output")?.dataset.search).toBe(`?profile=${current}`);
    },
  );

  it("uses title metadata in switcher and scope banner while selecting canonical IDs", async () => {
    vi.mocked(api.getProfiles).mockResolvedValue({ profiles: [
      info("default", { bot_title: "Home agent" }),
      info("u5357-u5bab", { bot_title: "南宫婉", display_name: "Legacy name" }),
      info("work", { display_name: "Work agent" }),
    ] });
    await mount();

    expect(container.querySelector('[role="combobox"]')?.textContent).toBe("this dashboard (Home agent)");
    const choices = await options();
    expect(choices.map((choice) => choice.textContent)).toEqual([
      "this dashboard (Home agent)", "南宫婉", "Work agent",
    ]);
    await act(async () => choices[1].click());

    expect(container.querySelector("output")?.dataset).toMatchObject({
      profile: "u5357-u5bab", search: "?profile=u5357-u5bab",
      profiles: JSON.stringify(["default", "u5357-u5bab", "work"]),
    });
    expect(getManagementProfile()).toBe("u5357-u5bab");
    expect(container.textContent).toContain("Managing profile “南宫婉”");
    expect(container.querySelector(".sr-only")?.textContent).toBe("南宫婉");
  });

  it("refreshes titles on focus without reselecting a profile with the same visible name", async () => {
    vi.mocked(api.getProfiles).mockResolvedValue({ profiles: [
      info("default"),
      info("alpha", { bot_title: "Shared name" }),
      info("beta", { bot_title: "Shared name" }),
    ] });
    await mount();
    const choices = await options();
    await act(async () => choices[2].click());
    expect(getManagementProfile()).toBe("beta");

    vi.mocked(api.getProfiles).mockResolvedValue({ profiles: [
      info("default"),
      info("alpha", { bot_title: "Shared name" }),
      info("beta", { bot_title: "Renamed agent" }),
    ] });
    await act(async () => window.dispatchEvent(new Event("focus")));

    expect(container.querySelector('[role="combobox"]')?.textContent).toBe("Renamed agent");
    expect(container.textContent).toContain("Managing profile “Renamed agent”");
    expect(container.querySelector("output")?.dataset).toMatchObject({
      profile: "beta", search: "?profile=beta",
    });
    expect(getManagementProfile()).toBe("beta");
    expect(api.getActiveProfile).toHaveBeenCalledTimes(1);
  });
});
