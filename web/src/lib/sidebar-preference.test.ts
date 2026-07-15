import { describe, expect, it } from "vitest";
import {
  LEGACY_SIDEBAR_COLLAPSED_KEY,
  readDesktopSidebarOpenFromStorage,
  SIDEBAR_OPEN_KEY,
  writeDesktopSidebarOpenToStorage,
} from "./sidebar-preference";

function mockStorage(initial: Record<string, string> = {}) {
  const data = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => (data.has(key) ? data.get(key)! : null),
    setItem: (key: string, value: string) => {
      data.set(key, value);
    },
    removeItem: (key: string) => {
      data.delete(key);
    },
    snapshot: () => Object.fromEntries(data),
  };
}

describe("readDesktopSidebarOpenFromStorage", () => {
  it("defaults to open when no preference is stored (fresh load)", () => {
    expect(readDesktopSidebarOpenFromStorage(mockStorage())).toBe(true);
  });

  it("returns false when desktop preference is explicitly closed", () => {
    const storage = mockStorage({ [SIDEBAR_OPEN_KEY]: "false" });
    expect(readDesktopSidebarOpenFromStorage(storage)).toBe(false);
  });

  it("returns true when desktop preference is explicitly open", () => {
    const storage = mockStorage({ [SIDEBAR_OPEN_KEY]: "true" });
    expect(readDesktopSidebarOpenFromStorage(storage)).toBe(true);
  });

  it("migrates legacy collapsed rail to fully closed and removes legacy key", () => {
    const storage = mockStorage({ [LEGACY_SIDEBAR_COLLAPSED_KEY]: "true" });
    expect(readDesktopSidebarOpenFromStorage(storage)).toBe(false);
    expect(storage.snapshot()).toEqual({ [SIDEBAR_OPEN_KEY]: "false" });
  });

  it("migrates legacy expanded rail to open sidebar", () => {
    const storage = mockStorage({ [LEGACY_SIDEBAR_COLLAPSED_KEY]: "false" });
    expect(readDesktopSidebarOpenFromStorage(storage)).toBe(true);
    expect(storage.snapshot()).toEqual({ [SIDEBAR_OPEN_KEY]: "true" });
  });
});

describe("writeDesktopSidebarOpenToStorage", () => {
  it("persists desktop open preference", () => {
    const storage = mockStorage();
    writeDesktopSidebarOpenToStorage(true, storage);
    expect(storage.snapshot()).toEqual({ [SIDEBAR_OPEN_KEY]: "true" });
  });

  it("persists desktop closed preference", () => {
    const storage = mockStorage();
    writeDesktopSidebarOpenToStorage(false, storage);
    expect(storage.snapshot()).toEqual({ [SIDEBAR_OPEN_KEY]: "false" });
  });
});
