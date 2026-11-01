// @vitest-environment jsdom
// A profile's theme override must change what is RENDERED without becoming the
// global theme. Reproduces the review scenario on PR #130196: switching to a
// profile with an override, then to an inheriting one, then to the dashboard's
// own profile must leave the global theme (localStorage + server) untouched.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";

const setThemeCalls: string[] = [];

/** In-memory Storage: the test must not depend on which Storage the host
 *  jsdom/Node combination exposes on `window`. */
function memoryStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() {
      return m.size;
    },
    clear: () => m.clear(),
    getItem: (k) => (m.has(k) ? m.get(k)! : null),
    key: (i) => Array.from(m.keys())[i] ?? null,
    removeItem: (k) => void m.delete(k),
    setItem: (k, v) => void m.set(k, String(v)),
  };
}
let storage: Storage;

// jsdom has no `CSS.escape`; the theme provider uses it when applying a font.
if (typeof (globalThis as { CSS?: unknown }).CSS === "undefined") {
  (globalThis as { CSS?: { escape: (s: string) => string } }).CSS = {
    escape: (s: string) => s.replace(/["\\]/g, "\\$&"),
  };
}
const profileThemes: Record<string, unknown> = {};

vi.mock("@/lib/api", () => ({
  api: {
    // The server's persisted global theme: the provider adopts it on mount.
    getThemes: () => Promise.resolve({ themes: [], active: "midnight" }),
    getFontPref: () => Promise.resolve({ font: "" }),
    setTheme: (name: string) => {
      setThemeCalls.push(name);
      return Promise.resolve({ ok: true });
    },
    setFontPref: () => Promise.resolve({ ok: true }),
    getProfiles: () => new Promise(() => {}),
    getActiveProfile: () => new Promise(() => {}),
  },
  fetchJSON: (url: string) => {
    const profile = new URL(url, "http://x").searchParams.get("profile") ?? "";
    return Promise.resolve(profileThemes[profile]);
  },
  setManagementProfile: () => {},
}));

import { ProfileProvider } from "./ProfileProvider";
import { useProfileTheme } from "./profile-theme";
import { useProfileScope } from "./useProfileScope";
import { ThemeProvider, useTheme } from "@/themes";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let container: HTMLDivElement;
let root: Root;
const ctl: { go?: (p: string) => void } = {};

function Probe() {
  const { setProfile } = useProfileScope();
  const theme = useTheme();
  const profileTheme = useProfileTheme();
  ctl.go = setProfile;
  return (
    <div
      data-global={theme.themeName}
      data-rendered={theme.activeThemeName}
      data-override={profileTheme.overrideThemeName ?? ""}
    />
  );
}

function seen() {
  const el = container.querySelector("div")!;
  return { global: el.dataset.global, rendered: el.dataset.rendered };
}

function overrideSeen() {
  return container.querySelector("div")!.dataset.override;
}

async function mount() {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () =>
    root.render(
      <MemoryRouter initialEntries={["/"]}>
        <ProfileProvider>
          <ThemeProvider>
            <Probe />
          </ThemeProvider>
        </ProfileProvider>
      </MemoryRouter>,
    ),
  );
}

async function switchTo(profile: string) {
  await act(async () => ctl.go!(profile));
  await act(async () => {}); // let the profile-theme fetch settle
}

beforeEach(() => {
  storage = memoryStorage();
  Object.defineProperty(window, "localStorage", { value: storage, configurable: true });
  storage.setItem("hermes-dashboard-theme", "midnight");
  setThemeCalls.length = 0;
  profileThemes["work"] = {
    profile: "work",
    theme: "ember",
    inherit_from_default: false,
    source: "override",
  };
  profileThemes["personal"] = {
    profile: "personal",
    theme: "midnight",
    inherit_from_default: true,
    source: "default",
  };
  profileThemes[""] = {
    profile: "",
    theme: "midnight",
    inherit_from_default: true,
    source: "global",
  };
});

afterEach(async () => {
  await act(async () => root.unmount());
  container.remove();
});

describe("profile theme override", () => {
  it("renders the override but never writes the global theme", async () => {
    await mount();
    await switchTo("work");
    expect(seen()).toEqual({ global: "midnight", rendered: "ember" });
    expect(storage.getItem("hermes-dashboard-theme")).toBe("midnight");
    expect(setThemeCalls).toEqual([]);
  });

  it("returns to the global theme on an inheriting profile and on the dashboard's own", async () => {
    await mount();
    await switchTo("work"); // 1+2: global midnight, work overrides with ember
    expect(seen().rendered).toBe("ember");

    await switchTo("personal"); // 3: inherits -> global theme again
    expect(seen()).toEqual({ global: "midnight", rendered: "midnight" });

    await switchTo(""); // 4: dashboard's own profile
    expect(seen()).toEqual({ global: "midnight", rendered: "midnight" });
    expect(storage.getItem("hermes-dashboard-theme")).toBe("midnight");
    expect(setThemeCalls).toEqual([]);
  });

  it("exposes the profile's own override so the theme list can highlight it", async () => {
    await mount();
    await switchTo("work");
    // ThemeSwitcher spreads this hook into its options list and marks the
    // matching theme active; without it a scoped profile shows none selected.
    expect(overrideSeen()).toBe("ember");
  });
});
