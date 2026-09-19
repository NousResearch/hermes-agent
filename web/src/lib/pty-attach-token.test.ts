// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * One "browser tab": its own sessionStorage, its own module state (the claimed
 * token) and the pages share the fake Web Locks manager, exactly like tabs on
 * one origin do.
 */
type Tab = {
  ptyAttachToken: (rotate?: boolean) => Promise<string>;
  storage: Storage;
};

const KEY = "hermes.pty.token.chat";

function fakeStorage(seed: Record<string, string> = {}): Storage {
  const store: Record<string, string> = { ...seed };
  return {
    clear: () => {
      for (const key of Object.keys(store)) delete store[key];
    },
    getItem: (key: string) => store[key] ?? null,
    key: (index: number) => Object.keys(store)[index] ?? null,
    get length() {
      return Object.keys(store).length;
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
  } as Storage;
}

/** Web Locks, as far as attach-token claiming uses them. */
function fakeLocks(held: Set<string>, requests: string[] = []) {
  return {
    request: (
      name: string,
      _options: { ifAvailable?: boolean },
      callback: (lock: { name: string } | null) => unknown,
    ) => {
      requests.push(name);
      if (held.has(name)) {
        callback(null);
        return Promise.resolve();
      }
      held.add(name);
      return Promise.resolve(callback({ name }));
    },
  };
}

/** The lock a tab holds disappears when the document unloads (reload/close). */
function simulateUnload(held: Set<string>) {
  held.clear();
}

async function openTab(
  held: Set<string>,
  seed: Record<string, string> = {},
  requests: string[] = [],
): Promise<Tab> {
  vi.resetModules(); // a new document has its own claimed-token state
  const storage = fakeStorage(seed);
  Object.defineProperty(window, "sessionStorage", { configurable: true, value: storage });
  Object.defineProperty(window.navigator, "locks", {
    configurable: true,
    value: fakeLocks(held, requests),
  });
  const { ptyAttachToken } = await import("./pty-attach-token");
  return { ptyAttachToken, storage };
}

beforeEach(() => {
  vi.stubGlobal("crypto", {
    getRandomValues: (values: Uint8Array) => {
      values.fill(Math.floor(Math.random() * 256));
      return values;
    },
  });
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  delete (window.navigator as { locks?: unknown }).locks;
});

describe("ptyAttachToken", () => {
  it("mints its own token for a duplicate tab instead of sharing the live one", async () => {
    const held = new Set<string>();
    const first = await openTab(held);
    const tokenA = await first.ptyAttachToken();

    // Chrome's "Duplicate tab" clones sessionStorage into the new tab.
    const duplicate = await openTab(held, { [KEY]: tokenA });
    const tokenB = await duplicate.ptyAttachToken();

    expect(tokenB).not.toBe(tokenA);
    expect(duplicate.storage.getItem(KEY)).toBe(tokenB);
  });

  it("reuses the stored token after a reload, once the old document released it", async () => {
    const held = new Set<string>();
    const before = await openTab(held, { [KEY]: "tab-token" });
    expect(await before.ptyAttachToken()).toBe("tab-token");

    simulateUnload(held); // reload: the previous document's lock is gone
    const reloaded = await openTab(held, { [KEY]: "tab-token" });

    expect(await reloaded.ptyAttachToken()).toBe("tab-token");
  });

  it("claims the token once per document so a reconnect never re-locks it", async () => {
    const held = new Set<string>();
    const requests: string[] = [];
    const tab = await openTab(held, { [KEY]: "tab-token" }, requests);

    expect(await tab.ptyAttachToken()).toBe("tab-token");
    expect(await tab.ptyAttachToken()).toBe("tab-token");

    expect(requests).toEqual([`hermes.pty.attach.tab-token`]);
  });

  it("rotates the token for an explicit fresh session", async () => {
    const held = new Set<string>();
    const tab = await openTab(held, { [KEY]: "tab-token" });

    const rotated = await tab.ptyAttachToken(true);

    expect(rotated).not.toBe("tab-token");
    expect(tab.storage.getItem(KEY)).toBe(rotated);
  });

  it("still isolates tabs on sessionStorage when the browser has no Web Locks", async () => {
    const tab = await openTab(new Set(), { [KEY]: "tab-token" });
    Object.defineProperty(window.navigator, "locks", { configurable: true, value: undefined });

    expect(await tab.ptyAttachToken()).toBe("tab-token");
  });
});
