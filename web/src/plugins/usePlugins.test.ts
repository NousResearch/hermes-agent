import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  getCachedManifests,
  cacheManifests,
  canSeedLoadedFromCache,
  MANIFEST_CACHE_KEY,
  pluginAssetUrl,
} from "./usePlugins";
import { HERMES_BASE_PATH } from "@/lib/api";
import type { PluginManifest } from "./types";

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
    expect(getCachedManifests("default")).toBeNull();
  });

  it("getCachedManifests returns null for invalid JSON", () => {
    storage.setItem(`${MANIFEST_CACHE_KEY}:default`, "not-json");
    expect(getCachedManifests("default")).toBeNull();
  });

  it("getCachedManifests returns null for non-array JSON", () => {
    storage.setItem(`${MANIFEST_CACHE_KEY}:default`, JSON.stringify({ foo: "bar" }));
    expect(getCachedManifests("default")).toBeNull();
  });

  it("getCachedManifests returns a valid manifest array", () => {
    const list: PluginManifest[] = [exampleManifest];
    cacheManifests("default", list);
    expect(getCachedManifests("default")).toEqual(list);
  });

  it("keeps each profile's list apart", () => {
    cacheManifests("work", [exampleManifest]);
    expect(getCachedManifests("default")).toBeNull();
    expect(getCachedManifests("work")).toEqual([exampleManifest]);
  });

  it("cacheManifests overwrites a previous cache on refresh", () => {
    const first: PluginManifest[] = [exampleManifest];
    cacheManifests("default", first);
    expect(getCachedManifests("default")).toEqual(first);

    const second: PluginManifest[] = [
      { ...exampleManifest, name: "updated", label: "Updated" },
    ];
    cacheManifests("default", second);
    expect(getCachedManifests("default")).toEqual(second);
  });

  it("cacheManifests swallows storage errors", () => {
    const badStorage = makeStorage();
    badStorage.setItem = () => {
      throw new Error("QuotaExceededError");
    };
    vi.stubGlobal("sessionStorage", badStorage);
    expect(() => cacheManifests("default", [exampleManifest])).not.toThrow();
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

describe("pluginAssetUrl", () => {
  it("names the profile whose plugin list described the plugin in its path", () => {
    expect(pluginAssetUrl("kanban", "dist/index.js", "work bench")).toBe(
      `${HERMES_BASE_PATH}/dashboard-plugins/@work%20bench/kanban/dist/index.js`,
    );
  });

  it("keeps the profile for URLs relative to a stylesheet or bundle", () => {
    // url("./icon.svg") in a stylesheet, or a chunk import, drops the sheet's query string.
    const sheet = new URL(pluginAssetUrl("kanban", "dist/style.css", "beta"), "http://127.0.0.1:9119");
    expect(new URL("./icon.svg", sheet).pathname).toBe(
      `${HERMES_BASE_PATH}/dashboard-plugins/@beta/kanban/dist/icon.svg`,
    );
    expect(new URL("../fonts/a.woff2", sheet).pathname).toBe(
      `${HERMES_BASE_PATH}/dashboard-plugins/@beta/kanban/fonts/a.woff2`,
    );
  });

  it("stays unscoped when no profile is known", () => {
    expect(pluginAssetUrl("kanban", "style.css", "")).toBe(
      `${HERMES_BASE_PATH}/dashboard-plugins/kanban/style.css`,
    );
  });
});
