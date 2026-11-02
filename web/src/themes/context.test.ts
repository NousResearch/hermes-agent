// @vitest-environment jsdom
import { beforeEach, describe, expect, it } from "vitest";

import { readCachedThemeDefs, writeCachedThemeDefs } from "./context";
import { defaultTheme } from "./presets";
import type { DashboardTheme } from "./types";

const STORAGE_KEY = "hermes-dashboard-theme-def";

function makeTheme(name: string): DashboardTheme {
  return { ...defaultTheme, name, label: name };
}

describe("theme definition cache (flash-mitigation)", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("returns an empty map when nothing is cached", () => {
    expect(readCachedThemeDefs()).toEqual({});
  });

  it("round-trips a written definitions map", () => {
    const defs = { bigcaptain: makeTheme("bigcaptain") };
    writeCachedThemeDefs(defs);
    expect(readCachedThemeDefs()).toEqual(defs);
  });

  it("is resilient to corrupt JSON in the cache", () => {
    window.localStorage.setItem(STORAGE_KEY, "{not-json");
    expect(readCachedThemeDefs()).toEqual({});
  });

  it("is resilient to a non-object JSON value in the cache", () => {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify("a string"));
    expect(readCachedThemeDefs()).toEqual({});
  });

  it("overwrites a previous cache on subsequent writes", () => {
    writeCachedThemeDefs({ old: makeTheme("old") });
    writeCachedThemeDefs({ fresh: makeTheme("fresh") });
    expect(readCachedThemeDefs()).toEqual({ fresh: makeTheme("fresh") });
  });
});
