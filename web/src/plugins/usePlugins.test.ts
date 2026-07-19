import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import {
  getCachedManifests,
  cacheManifests,
  MANIFEST_CACHE_KEY,
} from "./usePlugins";
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
});
