// @vitest-environment jsdom
// Tests for MachineLevelRouteGate: machine-level pages deep-linked under a
// management scope render an empty state instead of live controls; every
// other scope passes the page through untouched.

import { describe, it, expect, afterEach } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import type { ReactNode } from "react";

import { I18nProvider } from "@/i18n";
import { ProfileContext } from "@/contexts/profile-context";
import { MachineLevelRouteGate } from "./MachineLevelRouteGate";

let container: HTMLDivElement;
let root: Root;

function scope(
  profile: string,
  currentProfile: string,
  setProfile: (name: string) => void = () => {},
) {
  return { profile, currentProfile, profiles: [], setProfile };
}

async function renderScope(
  profile: string,
  currentProfile: string,
  ui: ReactNode,
  setProfile?: (name: string) => void,
) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <I18nProvider>
        <ProfileContext.Provider
          value={scope(profile, currentProfile, setProfile)}
        >
          {ui}
        </ProfileContext.Provider>
      </I18nProvider>,
    ),
  );
}

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("MachineLevelRouteGate", () => {
  it("renders the page when the dashboard manages its own profile", async () => {
    await renderScope("", "default", <MachineLevelRouteGate><p>live-page</p></MachineLevelRouteGate>);
    expect(container.textContent).toContain("live-page");
  });

  it("renders the page when the scope equals the dashboard's own profile", async () => {
    // Managing `default` FROM the default dashboard: the page shows exactly
    // what the banner claims, so no gate. Same rule as ProfileScopeBanner.
    await renderScope("default", "default", <MachineLevelRouteGate><p>live-page</p></MachineLevelRouteGate>);
    expect(container.textContent).toContain("live-page");
  });

  it("replaces live controls with the empty state under a foreign scope", async () => {
    await renderScope("architect", "default", <MachineLevelRouteGate><p>live-page</p></MachineLevelRouteGate>);
    expect(container.textContent).not.toContain("live-page");
    expect(container.textContent).toContain("architect");
    expect(container.textContent).toContain("not tied to the managed profile");
  });

  it("offers a switch-back button that clears the scope", async () => {
    let setTo: string | null = null;
    await renderScope(
      "architect",
      "default",
      <MachineLevelRouteGate><p>live-page</p></MachineLevelRouteGate>,
      (name) => {
        setTo = name;
      },
    );
    const btn = container.querySelector("button");
    expect(btn).not.toBeNull();
    expect(btn!.textContent).toContain("default");
    await act(async () => btn!.click());
    expect(setTo).toBe("");
  });
});
