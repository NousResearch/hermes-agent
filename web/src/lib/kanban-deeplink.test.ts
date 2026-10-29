/**
 * Unit tests for the Kanban plugin's task deep-link helpers.
 *
 * The IIFE in plugins/kanban/dashboard/dist/index.js exposes three pure
 * helpers on window.__HERMES_KANBAN_TEST__ *before* the SDK guard fires.
 * When window.__HERMES_PLUGIN_SDK__ is null the IIFE returns early, so only
 * those helpers are installed — making the test environment trivially cheap.
 *
 * We run the IIFE via Node's `vm.runInNewContext` with a minimal mock window
 * to exercise the real production code without jsdom.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { runInNewContext } from "node:vm";

// ---------------------------------------------------------------------------
// Load the plugin IIFE once
// ---------------------------------------------------------------------------

const DIST_PATH = resolve(
  __dirname,
  "../../../plugins/kanban/dashboard/dist/index.js",
);
const IIFE_CODE = readFileSync(DIST_PATH, "utf8");

// ---------------------------------------------------------------------------
// Minimal mock window — no jsdom required
// ---------------------------------------------------------------------------

interface TestHelpers {
  getInitialTaskId(search: string, lsValue: string | null): string | null;
  readSelectedTask(): string | null;
  clearSelectedTask(): void;
}

function makeContext(
  search = "",
  lsData: Record<string, string> = {},
): { win: Record<string, unknown>; store: Record<string, string> } {
  const store: Record<string, string> = { ...lsData };
  const win: Record<string, unknown> = {
    // No SDK → IIFE returns early after exposing __HERMES_KANBAN_TEST__
    __HERMES_PLUGIN_SDK__: null,
    __HERMES_KANBAN_TEST__: undefined,
    localStorage: {
      getItem: (k: string) => store[k] ?? null,
      setItem: (k: string, v: string) => {
        store[k] = v;
      },
      removeItem: (k: string) => {
        delete store[k];
      },
    },
    location: {
      search,
      href: `http://localhost/kanban${search}`,
    },
    // URLSearchParams is a Node.js global — pass it through so the IIFE can
    // access it as a free variable via the context.
    URLSearchParams,
    URL: globalThis.URL,
    history: { replaceState: () => {} },
  };
  return { win, store };
}

function loadHelpers(search = "", lsData: Record<string, string> = {}) {
  const { win, store } = makeContext(search, lsData);
  // Provide URLSearchParams + URL at the top-level context so the IIFE can
  // reference them as globals (they're used both in helpers and in the IIFE
  // body, which exits early when SDK is null).
  runInNewContext(IIFE_CODE, {
    window: win,
    URLSearchParams,
    URL: globalThis.URL,
  });
  const helpers = win.__HERMES_KANBAN_TEST__ as TestHelpers;
  return { helpers, store };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

const LS_KEY = "hermes.kanban.selectedTask";

describe("Kanban plugin — task deep-link helpers", () => {
  describe("getInitialTaskId", () => {
    it("returns task id from URL ?task= when present", () => {
      const { helpers } = loadHelpers();
      expect(helpers.getInitialTaskId("?task=t_abc123", null)).toBe("t_abc123");
    });

    it("returns task id from localStorage when URL param absent", () => {
      const { helpers } = loadHelpers();
      expect(helpers.getInitialTaskId("", "t_lsvalue")).toBe("t_lsvalue");
    });

    it("URL param takes priority over localStorage", () => {
      const { helpers } = loadHelpers();
      expect(helpers.getInitialTaskId("?task=t_url", "t_ls")).toBe("t_url");
    });

    it("returns null when neither URL nor LS has a task", () => {
      const { helpers } = loadHelpers();
      expect(helpers.getInitialTaskId("", null)).toBeNull();
    });

    it("returns null for empty string URL param", () => {
      const { helpers } = loadHelpers();
      expect(helpers.getInitialTaskId("?task=", null)).toBeNull();
    });

    it("trims whitespace from URL param", () => {
      const { helpers } = loadHelpers();
      expect(helpers.getInitialTaskId("?task=  t_abc  ", null)).toBe("t_abc");
    });

    it("returns null for whitespace-only LS value", () => {
      const { helpers } = loadHelpers();
      expect(helpers.getInitialTaskId("", "   ")).toBeNull();
    });
  });

  describe("readSelectedTask", () => {
    it("returns stored task id from localStorage", () => {
      const { helpers } = loadHelpers("", { [LS_KEY]: "t_stored" });
      expect(helpers.readSelectedTask()).toBe("t_stored");
    });

    it("returns null when LS key is absent", () => {
      const { helpers } = loadHelpers();
      expect(helpers.readSelectedTask()).toBeNull();
    });
  });

  describe("clearSelectedTask", () => {
    it("removes the LS key", () => {
      const { helpers, store } = loadHelpers("", { [LS_KEY]: "t_todelete" });
      expect(store[LS_KEY]).toBe("t_todelete");
      helpers.clearSelectedTask();
      expect(store[LS_KEY]).toBeUndefined();
    });

    it("does not throw when LS key is absent", () => {
      const { helpers } = loadHelpers();
      expect(() => helpers.clearSelectedTask()).not.toThrow();
    });
  });

  describe("one-shot contract: LS key is read once then cleared", () => {
    it("readSelectedTask + clearSelectedTask simulates mount behavior", () => {
      const lsData = { [LS_KEY]: "t_oneshot" };
      const { helpers, store } = loadHelpers("", lsData);

      // Simulates useState initialiser: read
      const initial = helpers.readSelectedTask();
      expect(initial).toBe("t_oneshot");

      // Simulates cleanup useEffect: clear
      helpers.clearSelectedTask();
      expect(store[LS_KEY]).toBeUndefined();

      // Second mount (new context, same LS now empty) returns null
      const { helpers: helpers2 } = loadHelpers("", store);
      expect(helpers2.readSelectedTask()).toBeNull();
    });
  });
});
