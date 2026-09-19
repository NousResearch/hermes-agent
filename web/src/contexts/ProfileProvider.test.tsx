// @vitest-environment jsdom
import { act, useLayoutEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useLocation, useNavigate } from "react-router";
import { afterEach, describe, expect, it, vi } from "vitest";

import { setManagementProfile } from "@/lib/api";
import { ProfileProvider } from "./ProfileProvider";
import { useProfileScope } from "./useProfileScope";

vi.mock("@/lib/api", () => ({
  api: {
    getProfiles: vi.fn().mockResolvedValue({ profiles: [] }),
    getActiveProfile: vi.fn().mockResolvedValue({
      current: "default",
      active: "default",
    }),
  },
  setManagementProfile: vi.fn(),
}));

interface CommittedScope {
  url: string;
  profile: string;
  managementProfile: string | undefined;
}

function Harness({ commits }: { commits: CommittedScope[] }) {
  const { profile, setProfile } = useProfileScope();
  const location = useLocation();
  const navigate = useNavigate();

  useLayoutEffect(() => {
    const managementCalls = vi.mocked(setManagementProfile).mock.calls;
    commits.push({
      url: `${location.pathname}${location.search}`,
      profile,
      managementProfile: managementCalls[managementCalls.length - 1]?.[0],
    });
  }, [commits, location, profile]);

  return (
    <>
      <button
        type="button"
        onClick={() => setProfile("mapautogen", { clearResume: true })}
      >
        Open Map Autogen PM
      </button>
      <button type="button" onClick={() => navigate("/skills")}>
        Open Skills
      </button>
      <output data-testid="location">{location.pathname}{location.search}</output>
      <output data-testid="profile">{profile}</output>
    </>
  );
}

let container: HTMLDivElement;
let root: Root;

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ProfileProvider", () => {
  it("commits a clear-resume switch without stale profile scope", async () => {
    const commits: CommittedScope[] = [];
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);

    await act(async () => {
      root.render(
        <MemoryRouter
          initialEntries={["/chat?profile=default&resume=old-session&learn=keep-me"]}
        >
          <ProfileProvider>
            <Harness commits={commits} />
          </ProfileProvider>
        </MemoryRouter>,
      );
    });

    const button = container.querySelector<HTMLButtonElement>("button");
    await act(async () => button!.click());

    expect(container.querySelector('[data-testid="location"]')?.textContent).toBe(
      "/chat?profile=mapautogen&learn=keep-me",
    );
    expect(container.querySelector('[data-testid="profile"]')?.textContent).toBe(
      "mapautogen",
    );
    expect(commits).toContainEqual({
      url: "/chat?profile=mapautogen&learn=keep-me",
      profile: "mapautogen",
      managementProfile: "mapautogen",
    });
    const skillsButton = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((candidate) => candidate.textContent?.includes("Skills"));
    await act(async () => skillsButton!.click());

    expect(container.querySelector('[data-testid="location"]')?.textContent).toBe(
      "/skills?profile=mapautogen",
    );
    expect(container.querySelector('[data-testid="profile"]')?.textContent).toBe(
      "mapautogen",
    );

    for (const commit of commits) {
      const urlProfile = new URLSearchParams(commit.url.split("?")[1]).get("profile");
      if (urlProfile !== null) expect(commit.profile).toBe(urlProfile);
      expect(commit.managementProfile).toBe(commit.profile);
    }
  });
});
