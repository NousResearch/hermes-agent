// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const profileScope = vi.hoisted(() => ({
  profile: "",
  currentProfile: "default",
  profiles: ["default", "mapautogen", "microtwin", "scenarioautogen", "realtosim"],
  setProfile: vi.fn(),
}));

vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => profileScope,
}));

let container: HTMLDivElement;
let root: Root;

async function render(initialEntry = "/chat?resume=old-session&learn=keep-me") {
  const { ProjectBotLauncher } = await import("./ProjectBotLauncher");
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root.render(
      <MemoryRouter initialEntries={[initialEntry]}>
        <ProjectBotLauncher />
      </MemoryRouter>,
    );
  });
}

beforeEach(() => {
  profileScope.profile = "";
  profileScope.currentProfile = "default";
  profileScope.profiles = ["default", "mapautogen", "microtwin", "scenarioautogen", "realtosim"];
  profileScope.setProfile.mockReset();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ProjectBotLauncher", () => {
  it("marks Main CTO selected when the dashboard profile has no explicit scope", async () => {
    await render("/chat");

    const launch = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Open Main CTO"]',
    );
    expect(launch?.getAttribute("aria-current")).toBe("true");
  });

  it("uses the dashboard process profile when there is no explicit scope", async () => {
    profileScope.currentProfile = "realtosim";
    await render("/chat");

    const launch = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Open Real-to-Sim PM"]',
    );
    expect(launch?.getAttribute("aria-current")).toBe("true");
  });

  it("opens the Map Autogen PM profile workspace without a stale resume id", async () => {
    await render();

    const launch = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Open Map Autogen PM"]',
    );
    expect(launch).not.toBeNull();

    await act(async () => launch!.click());

    expect(profileScope.setProfile).toHaveBeenCalledWith("mapautogen", {
      clearResume: true,
    });
  });

  it("launches MicroTwin PM and marks the selected profile", async () => {
    profileScope.profile = "microtwin";
    await render("/chat?profile=microtwin");

    const launch = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Open MicroTwin PM"]',
    );
    expect(launch?.getAttribute("aria-current")).toBe("true");

    await act(async () => launch!.click());
    expect(profileScope.setProfile).toHaveBeenCalledWith("microtwin", {
      clearResume: true,
    });
  });

  it.each([
    ["Main CTO", "default"],
    ["Scenario Autocreation PM", "scenarioautogen"],
    ["Real-to-Sim PM", "realtosim"],
  ])("opens the %s workspace", async (name, profileId) => {
    await render();

    const launch = container.querySelector<HTMLButtonElement>(
      `button[aria-label="Open ${name}"]`,
    );
    expect(launch).not.toBeNull();

    await act(async () => launch!.click());
    expect(profileScope.setProfile).toHaveBeenCalledWith(profileId, {
      clearResume: true,
    });
  });

  it("shows an unavailable pilot without launching a missing profile", async () => {
    profileScope.profiles = ["default", "mapautogen"];
    await render("/chat");

    const launch = container.querySelector<HTMLButtonElement>(
      'button[aria-label="Open MicroTwin PM"]',
    );
    expect(launch?.disabled).toBe(true);
    expect(container.textContent).toContain("Not configured");
  });
});
