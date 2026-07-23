import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let ptyAttachToken: (typeof import("./pty-attach-token"))["ptyAttachToken"];

async function reloadTokenModule() {
  vi.resetModules();
  ({ ptyAttachToken } = await import("./pty-attach-token"));
}

function storage(values = new Map<string, string>()) {
  return {
    values,
    getItem: vi.fn((key: string) => values.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => void values.set(key, value)),
  };
}

function browserPage(
  sessionStorage = storage(),
  navigationType: PerformanceNavigationTiming["type"] | undefined = "navigate",
) {
  const localStorage = storage();
  const listeners = new Map<string, Set<() => void>>();
  return {
    sessionStorage,
    localStorage,
    opener: null as object | null,
    performance: {
      getEntriesByType: vi.fn(() =>
        navigationType ? [{ type: navigationType }] : [],
      ),
    },
    addEventListener: vi.fn((type: string, listener: () => void) => {
      const handlers = listeners.get(type) ?? new Set();
      handlers.add(listener);
      listeners.set(type, handlers);
    }),
    dispatch(type: string) {
      for (const listener of listeners.get(type) ?? []) listener();
    },
  };
}

function broadcastChannel() {
  const listeners = new Map<string, Set<(event: MessageEvent) => void>>();

  class MockBroadcastChannel {
    private readonly handlers = new Set<(event: MessageEvent) => void>();
    private readonly name: string;

    constructor(name: string) {
      this.name = name;
      const channels = listeners.get(name) ?? new Set();
      channels.add(this.receive);
      listeners.set(name, channels);
    }

    addEventListener(_: "message", listener: (event: MessageEvent) => void) {
      this.handlers.add(listener);
    }

    removeEventListener(_: "message", listener: (event: MessageEvent) => void) {
      this.handlers.delete(listener);
    }

    postMessage(data: unknown) {
      for (const listener of listeners.get(this.name) ?? []) {
        if (listener !== this.receive) {
          queueMicrotask(() => listener({ data } as MessageEvent));
        }
      }
    }

    private receive = (event: MessageEvent) => {
      for (const handler of this.handlers) handler(event);
    };
  }

  return MockBroadcastChannel;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

beforeEach(async () => {
  vi.stubGlobal("BroadcastChannel", broadcastChannel());
  await reloadTokenModule();
});

describe("ptyAttachToken", () => {
  it("reuses the tab-local token across reloads without touching shared localStorage", async () => {
    const sessionStorage = storage();
    const firstPage = browserPage(sessionStorage);
    vi.stubGlobal("window", firstPage);
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => bytes.fill(7),
    });

    expect(await ptyAttachToken()).toBe("07".repeat(16));

    const reloadedPage = browserPage(sessionStorage, "reload");
    firstPage.dispatch("pagehide");
    vi.stubGlobal("window", reloadedPage);
    await reloadTokenModule();
    expect(await ptyAttachToken()).toBe("07".repeat(16));
    expect(sessionStorage.setItem).toHaveBeenCalledTimes(1);
    expect(firstPage.localStorage.getItem).not.toHaveBeenCalled();
    expect(firstPage.localStorage.setItem).not.toHaveBeenCalled();
  });

  it("keeps the token through same-tab history navigation", async () => {
    const sessionStorage = storage();
    const firstPage = browserPage(sessionStorage);
    vi.stubGlobal("window", firstPage);
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => bytes.fill(7),
    });

    expect(await ptyAttachToken()).toBe("07".repeat(16));

    const historyPage = browserPage(sessionStorage, "back_forward");
    firstPage.dispatch("pagehide");
    vi.stubGlobal("window", historyPage);
    await reloadTokenModule();

    expect(await ptyAttachToken()).toBe("07".repeat(16));
  });

  it("mints a browser-duplicated tab a fresh token after cloned back-forward navigation", async () => {
    let nextByte = 1;
    const parentPage = browserPage();
    vi.stubGlobal("window", parentPage);
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => bytes.fill(nextByte++),
    });

    expect(await ptyAttachToken()).toBe("01".repeat(16));

    const childPage = browserPage(
      storage(new Map(parentPage.sessionStorage.values)),
      "back_forward",
    );
    expect(childPage.opener).toBeNull();
    vi.stubGlobal("window", childPage);
    await reloadTokenModule();

    expect(await ptyAttachToken()).toBe("02".repeat(16));
    expect(await ptyAttachToken()).toBe("02".repeat(16));
    expect(parentPage.sessionStorage.values.get("hermes.pty.token.chat")).toBe(
      "01".repeat(16),
    );

    const reloadedChildPage = browserPage(childPage.sessionStorage, "reload");
    childPage.dispatch("pagehide");
    vi.stubGlobal("window", reloadedChildPage);
    await reloadTokenModule();
    expect(await ptyAttachToken()).toBe("02".repeat(16));
  });

  it("rotates the attachment after an explicitly fresh session", async () => {
    const page = browserPage();
    let nextByte = 1;
    vi.stubGlobal("window", page);
    vi.stubGlobal("crypto", {
      getRandomValues: (bytes: Uint8Array) => bytes.fill(nextByte++),
    });

    expect(await ptyAttachToken()).toBe("01".repeat(16));
    expect(await ptyAttachToken(true)).toBe("02".repeat(16));
  });
});
