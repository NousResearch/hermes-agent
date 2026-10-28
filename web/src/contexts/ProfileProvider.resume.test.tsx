// @vitest-environment jsdom
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Link, MemoryRouter, useLocation } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const backend = vi.hoisted(() => ({
  current: "default",
  active: "default",
  info: null as Promise<{ current: string; active: string }> | null,
}));
vi.mock("@/lib/api", () => ({
  api: {
    getProfiles: async () => ({
      profiles: [{ name: "default" }, { name: "research" }],
    }),
    getActiveProfile: () =>
      backend.info ??
      Promise.resolve({ current: backend.current, active: backend.active }),
  },
  setManagementProfile: vi.fn(),
}));

import { ProfileProvider } from "./ProfileProvider";
import { useProfileScope } from "./useProfileScope";

(
  globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }
).IS_REACT_ACT_ENVIRONMENT = true;
let container: HTMLDivElement;
let root: Root;
const committedScopes: { profile: string; resume: string | null }[] = [];

interface ProbeProps {
  selection: string;
}

function Probe({ selection }: ProbeProps) {
  const { profile, setProfile } = useProfileScope();
  const { search } = useLocation();
  useEffect(() => {
    committedScopes.push({
      profile,
      resume: new URLSearchParams(search).get("resume"),
    });
  }, [profile, search]);
  return (
    <>
      <button
        data-profile={profile}
        data-search={search}
        onClick={() => setProfile(selection)}
      />
      <Link to="/chat?profile=research&resume=research-session">
        Research session
      </Link>
    </>
  );
}

async function mount(previous: string | null, selection: string) {
  window.__HERMES_DASHBOARD_PROFILE__ = backend.current;
  const query = previous === null ? "" : `profile=${previous}&`;
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => {
    root.render(
      <MemoryRouter
        initialEntries={[
          `/chat?${query}resume=old-session&panel=models&channel=chat-1`,
        ]}
      >
        <ProfileProvider>
          <Probe selection={selection} />
        </ProfileProvider>
      </MemoryRouter>,
    );
  });
}

beforeEach(() => {
  backend.current = "default";
  backend.active = "default";
  backend.info = null;
  committedScopes.length = 0;
});
afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  delete window.__HERMES_DASHBOARD_PROFILE__;
});

describe("profile selection with a resumed chat", () => {
  it.each([
    ["default", "research"],
    ["research", ""],
  ])(
    "drops the old resume when switching %s to %s",
    async (previous, selection) => {
      await mount(previous, selection);
      const button = container.querySelector("button")!;
      await act(async () => button.click());
      const params = new URLSearchParams(button.dataset.search);
      expect(button.dataset.profile).toBe(selection);
      expect(params.get("profile")).toBe(selection || null);
      expect(params.has("resume")).toBe(false);
      expect(params.get("panel")).toBe("models");
      expect(params.get("channel")).toBe("chat-1");
      expect(committedScopes).not.toContainEqual({
        profile: selection,
        resume: "old-session",
      });
    },
  );

  it.each([
    ["default", "default", ""],
    ["research", "research", ""],
    ["research", "", "research"],
    ["default", "research", "research"],
  ])(
    "keeps resume with serving=%s, %s to %s",
    async (current, previous, selection) => {
      backend.current = current;
      await mount(previous, selection);
      const button = container.querySelector("button")!;
      await act(async () => button.click());
      const params = new URLSearchParams(button.dataset.search);
      expect(button.dataset.profile).toBe(selection);
      expect(params.get("profile")).toBe(selection || null);
      expect(params.get("resume")).toBe("old-session");
    },
  );

  it("clears the old resume when bootstrap adopts another active profile", async () => {
    backend.active = "research";
    await mount(null, "");
    const button = container.querySelector("button")!;
    expect(button.dataset.profile).toBe("research");
    const params = new URLSearchParams(button.dataset.search);
    expect(params.get("profile")).toBe("research");
    expect(params.has("resume")).toBe(false);
    expect(committedScopes).not.toContainEqual({
      profile: "research",
      resume: "old-session",
    });
  });

  it.each([
    {
      choice: "switcher",
      selection: "research",
      selector: "button",
      resume: null,
    },
    {
      choice: "link",
      selection: "research",
      selector: "a",
      resume: "research-session",
    },
    { choice: "own", selection: "", selector: "button", resume: "old-session" },
  ])(
    "does not let a late bootstrap override a newer $choice choice",
    async ({ selection, selector, resume }) => {
      let resolveInfo!: (value: { current: string; active: string }) => void;
      backend.info = new Promise((resolve) => {
        resolveInfo = resolve;
      });
      await mount(null, selection);
      const button = container.querySelector("button")!;
      await act(async () => {
        container.querySelector<HTMLElement>(selector)!.click();
      });
      await act(async () => {
        resolveInfo({ current: "default", active: "other" });
      });
      const params = new URLSearchParams(button.dataset.search);
      expect(button.dataset.profile).toBe(selection);
      expect(params.get("profile")).toBe(selection || null);
      expect(params.get("resume")).toBe(resume);
    },
  );

  it.each(["default", "research"])(
    "keeps the %s profile resume if bootstrap metadata fails",
    async (current) => {
      backend.current = current;
      let rejectInfo!: (error: Error) => void;
      backend.info = new Promise((_resolve, reject) => {
        rejectInfo = reject;
      });
      await mount(null, "");
      await act(async () => {
        rejectInfo(new Error("metadata unavailable"));
      });
      const button = container.querySelector("button")!;
      const params = new URLSearchParams(button.dataset.search);
      expect(button.dataset.profile).toBe(current);
      expect(params.get("profile")).toBe(current);
      expect(params.get("resume")).toBe("old-session");
      await act(async () => button.click());
      expect(new URLSearchParams(button.dataset.search).get("resume")).toBe(
        "old-session",
      );
    },
  );
});
