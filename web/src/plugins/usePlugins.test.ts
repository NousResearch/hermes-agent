// @vitest-environment jsdom
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import React from "react";
import { render, act, cleanup } from "@testing-library/react";
import {
  getCachedManifests,
  cacheManifests,
  canSeedLoadedFromCache,
  profileSwitchReloadTarget,
  usePlugins,
  MANIFEST_CACHE_KEY,
} from "./usePlugins";
import { api } from "@/lib/api";
import type { PluginManifest, RegisteredPlugin } from "./types";

function makeStorage(): Storage {
  const store = new Map<string, string>();
  return {
    getItem(key: string) {
      return store.get(key) ?? null;
    },
    setItem(key: string, value: string) {
      store.set(key, value);
    },
    removeItem(key: string) {
      store.delete(key);
    },
    clear() {
      store.clear();
    },
    get length() {
      return store.size;
    },
    key(index: number) {
      return Array.from(store.keys())[index] ?? null;
    },
  } as Storage;
}

const exampleManifest: PluginManifest = {
  name: "test",
  label: "Test",
  description: "A test plugin",
  icon: "Puzzle",
  version: "1.0.0",
  tab: { path: "/test" },
  entry: "index.js",
  has_api: false,
  source: "local",
};

describe("plugin manifest cache helpers", () => {
  let storage: Storage;

  beforeEach(() => {
    storage = makeStorage();
    vi.stubGlobal("sessionStorage", storage);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("getCachedManifests returns null when nothing is cached", () => {
    expect(getCachedManifests()).toBeNull();
  });

  it("getCachedManifests returns null for invalid JSON", () => {
    storage.setItem(MANIFEST_CACHE_KEY, "not-json");
    expect(getCachedManifests()).toBeNull();
  });

  it("getCachedManifests returns null for non-array JSON", () => {
    storage.setItem(MANIFEST_CACHE_KEY, JSON.stringify({ foo: "bar" }));
    expect(getCachedManifests()).toBeNull();
  });

  it("getCachedManifests returns null for scalar JSON", () => {
    storage.setItem(MANIFEST_CACHE_KEY, JSON.stringify(42));
    expect(getCachedManifests()).toBeNull();
  });

  it("getCachedManifests returns a valid manifest array", () => {
    const list: PluginManifest[] = [exampleManifest];
    cacheManifests(list);
    expect(getCachedManifests()).toEqual(list);
  });

  it("cacheManifests overwrites a previous cache on refresh", () => {
    const first: PluginManifest[] = [exampleManifest];
    cacheManifests(first);
    expect(getCachedManifests()).toEqual(first);

    const second: PluginManifest[] = [
      { ...exampleManifest, name: "updated", label: "Updated" },
    ];
    cacheManifests(second);
    expect(getCachedManifests()).toEqual(second);
  });

  it("cacheManifests swallows storage errors", () => {
    const badStorage = makeStorage();
    badStorage.setItem = () => {
      throw new Error("QuotaExceededError");
    };
    vi.stubGlobal("sessionStorage", badStorage);
    expect(() => cacheManifests([exampleManifest])).not.toThrow();
  });

  it("namespaces the manifest cache per management profile (#46408)", () => {
    const a: PluginManifest[] = [exampleManifest];
    const b: PluginManifest[] = [{ ...exampleManifest, name: "worker-plugin" }];
    cacheManifests(a, "alpha");
    cacheManifests(b, "beta");
    // Profiles never read each other's cached list.
    expect(getCachedManifests("alpha")).toEqual(a);
    expect(getCachedManifests("beta")).toEqual(b);
    // The unset profile keeps the legacy base key (the dashboard's own profile).
    expect(getCachedManifests()).toBeNull();
    cacheManifests(a);
    expect(getCachedManifests()).toEqual(a);
    expect(getCachedManifests("alpha")).toEqual(a);
  });
});

describe("canSeedLoadedFromCache (loading seed gate)", () => {
  it("returns false when there is no cache (first visit keeps loading=true)", () => {
    expect(canSeedLoadedFromCache(null)).toBe(false);
  });

  it("returns true for an empty cached list", () => {
    expect(canSeedLoadedFromCache([])).toBe(true);
  });

  it("returns true when no cached manifest overrides /chat", () => {
    const list: PluginManifest[] = [
      exampleManifest,
      {
        ...exampleManifest,
        name: "other",
        tab: { path: "/other", override: "/skills" },
      },
    ];
    expect(canSeedLoadedFromCache(list)).toBe(true);
  });

  it("returns false when a cached manifest overrides /chat — loading must stay true so App.tsx's pluginsLoading gate keeps the persistent chat host unmounted", () => {
    const list: PluginManifest[] = [
      exampleManifest,
      {
        ...exampleManifest,
        name: "chat-replacer",
        tab: { path: "/chat-alt", override: "/chat" },
      },
    ];
    expect(canSeedLoadedFromCache(list)).toBe(false);
  });

  it("tolerates malformed cached entries missing a tab object", () => {
    const malformed = [
      { ...exampleManifest, tab: undefined },
    ] as unknown as PluginManifest[];
    expect(canSeedLoadedFromCache(malformed)).toBe(true);
  });
});

describe("profileSwitchReloadTarget (review P1 on #107006)", () => {
  it("targets the same path with the new profile param, dropping the stale one", () => {
    const params = new URLSearchParams("profile=alpha");
    expect(profileSwitchReloadTarget("/x/y?profile=alpha", params, "beta")).toBe(
      "/x/y?profile=beta",
    );
  });

  it("drops ?profile= entirely when switching back to the dashboard's own profile", () => {
    const params = new URLSearchParams("profile=alpha&foo=1");
    expect(profileSwitchReloadTarget("/x/y?profile=alpha&foo=1", params, "")).toBe(
      "/x/y?foo=1",
    );
  });

  it("preserves unrelated query params and strips the hash", () => {
    const params = new URLSearchParams("foo=1");
    expect(profileSwitchReloadTarget("/x/y?foo=1#frag", params, "beta")).toBe(
      "/x/y?foo=1&profile=beta",
    );
  });
});

describe("usePlugins profile-switch execution retirement (#46408)", () => {
  // jsdom: window.location.assign needs a stub, and the hook reads
  // sessionStorage for the manifest cache.
  let storage: Storage;
  let assigned: string[] | null;

  beforeEach(() => {
    storage = makeStorage();
    vi.stubGlobal("sessionStorage", storage);
    assigned = [];
    Object.defineProperty(window, "location", {
      configurable: true,
      value: {
        ...window.location,
        href: "http://localhost:3000/",
        search: "",
        assign: (url: string) => {
          assigned!.push(url);
        },
      },
    });
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
    assigned = null;
    document.head.innerHTML = "";
    document.body.innerHTML = "";
  });

  function renderPlugins(initialProfile = "") {
    let latest!: {
      plugins: RegisteredPlugin[];
      manifests: PluginManifest[];
      loading: boolean;
    };
    function Probe({ profile }: { profile: string }) {
      latest = usePlugins(profile);
      return null;
    }
    const { rerender } = render(React.createElement(Probe, { profile: initialProfile }));
    return {
      setProfile: (profile: string) =>
        rerender(React.createElement(Probe, { profile })),
      state: () => latest,
      assigned: () => assigned!,
    };
  }

  function mockGetPlugins(pages: Record<string, PluginManifest[]>) {
    vi.spyOn(api, "getPlugins").mockImplementation(
      () =>
        new Promise((resolve) => {
          setTimeout(() => resolve(pages["default"] ?? []), 5);
        }),
    );
  }

  it("reloads the document on profile switch instead of resolving the new profile against the old JS realm", async () => {
    const onlyA: PluginManifest[] = [
      { ...exampleManifest, name: "alpha-only" },
    ];
    storage.setItem(
      `${MANIFEST_CACHE_KEY}:alpha`,
      JSON.stringify(onlyA),
    );
    mockGetPlugins({ default: [] });

    const view = renderPlugins("alpha");
    // Cache seed must not arm the loading gate while a /chat override could
    // exist — the switch path itself is what this test pins.
    await act(async () => {
      view.setProfile("beta");
    });

    // The switch defers to a full document reload — the ONLY closure that
    // retires profile A's executed registrations (_registered/_slotRegistry)
    // and side effects before profile B's manifests resolve.
    expect(view.assigned()).toEqual(["http://localhost:3000/?profile=beta"]);
  });

  it("keeps plugin fetching/assets frozen while the reload is pending", async () => {
    storage.setItem(
      `${MANIFEST_CACHE_KEY}:alpha`,
      JSON.stringify([{ ...exampleManifest, name: "alpha-only" }]),
    );
    // Mount fetches once for the initial profile; the switch must not fetch again.
    const getPlugins = vi.spyOn(api, "getPlugins").mockResolvedValue([]);

    const view = renderPlugins("alpha");
    await act(async () => {
      view.setProfile("beta");
    });

    expect(getPlugins).toHaveBeenCalledTimes(1);
    expect(view.assigned()).toEqual(["http://localhost:3000/?profile=beta"]);
    // No asset injection: the pending document is going away.
    expect(document.querySelectorAll("script[data-hermes-plugin]")).toHaveLength(0);
    expect(document.querySelectorAll("link[rel=stylesheet]")).toHaveLength(0);
  });
});
