// @vitest-environment jsdom
// The management scope and its ?profile= URL param stay in sync. A link that
// changes the path AND the param together (Profiles > "Manage skills & tools"
// goes to /skills?profile=X) must switch the scope to X.

import { afterEach, describe, expect, it, vi } from "vitest";
import { act, useEffect } from "react";
import { createRoot, type Root } from "react-dom/client";
import {
  MemoryRouter,
  useLocation,
  useNavigate,
  type NavigateFunction,
} from "react-router";

vi.mock("@/lib/api", () => ({
  api: {
    getProfiles: () => new Promise(() => {}),
    getActiveProfile: () => new Promise(() => {}),
  },
  setManagementProfile: () => {},
}));

import { ProfileProvider } from "./ProfileProvider";
import { useProfileScope } from "./useProfileScope";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let container: HTMLDivElement;
let root: Root;
const router: { navigate?: NavigateFunction } = {};

function Probe() {
  const navigate = useNavigate();
  const { profile, setProfile } = useProfileScope();
  const { pathname, search } = useLocation();
  useEffect(() => {
    router.navigate = navigate;
  }, [navigate]);
  return (
    <button
      data-profile={profile}
      data-url={pathname + search}
      onClick={() => setProfile("gamma")}
    />
  );
}

function seen() {
  const probe = container.querySelector("button")!;
  return { profile: probe.dataset.profile, url: probe.dataset.url };
}

async function mount(initialEntry: string) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <MemoryRouter initialEntries={[initialEntry]}>
        <ProfileProvider>
          <Probe />
        </ProfileProvider>
      </MemoryRouter>,
    ),
  );
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("ProfileProvider URL sync", () => {
  it("adopts ?profile= from a link that also changes the path", async () => {
    await mount("/profiles?profile=alpha");
    expect(seen()).toEqual({ profile: "alpha", url: "/profiles?profile=alpha" });

    await act(async () => router.navigate!("/skills?profile=beta"));

    expect(seen()).toEqual({ profile: "beta", url: "/skills?profile=beta" });
  });

  it("keeps the scope on the URL after a bare nav link and after the switcher", async () => {
    await mount("/profiles?profile=alpha");

    await act(async () => router.navigate!("/config"));
    expect(seen()).toEqual({ profile: "alpha", url: "/config?profile=alpha" });

    await act(async () => container.querySelector("button")!.click());
    expect(seen()).toEqual({ profile: "gamma", url: "/config?profile=gamma" });
  });
});
