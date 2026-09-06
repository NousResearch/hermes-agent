// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { PTY_TICKET_TIMEOUT_MS } from "@/lib/pty-reconnect";

class FakeFitAddon {
  fit() {}
}

class FakeWebglAddon {
  onContextLoss() {
    return { dispose() {} };
  }
}

class FakeTerminal {
  static instances: FakeTerminal[] = [];

  options: Record<string, unknown>;
  rows = 24;
  cols = 80;
  resetCount = 0;
  writes: string[] = [];
  parser = {
    registerOscHandler: vi.fn(),
  };
  unicode = { activeVersion: "" };

  constructor(options: Record<string, unknown>) {
    this.options = options;
    FakeTerminal.instances.push(this);
  }

  attachCustomKeyEventHandler() {
    return true;
  }

  attachCustomWheelEventHandler() {
    return true;
  }

  clearSelection() {}

  dispose() {}

  focus() {}

  getSelection() {
    return "";
  }

  loadAddon() {}

  onData() {
    return { dispose() {} };
  }

  onResize() {
    return { dispose() {} };
  }

  onScroll() {
    return { dispose() {} };
  }

  get buffer() {
    // Minimal active-buffer surface for the resume follow-scroll pin
    // (isViewportPinnedToBottom reads viewportY/baseY).
    return { active: { baseY: 0, viewportY: 0 } };
  }

  scrollToBottom() {}

  open() {}

  paste() {}

  refresh() {}

  reset() {
    this.resetCount += 1;
  }

  write(data: string) {
    this.writes.push(data);
  }
}

const maybeReloadForLoopbackWsAuthFailure = vi.fn(() => false);
const apiMocks = vi.hoisted(() => ({
  buildWsUrl: vi.fn(
    async (path: string, params?: Record<string, string>) => {
      const url = new URL(path, "ws://localhost");
      for (const [key, value] of Object.entries(params ?? {})) {
        url.searchParams.set(key, value);
      }
      return url.toString();
    },
  ),
}));

vi.mock("@xterm/addon-fit", () => ({ FitAddon: FakeFitAddon }));
vi.mock("@xterm/addon-unicode11", () => ({ Unicode11Addon: class {} }));
vi.mock("@xterm/addon-web-links", () => ({ WebLinksAddon: class {} }));
vi.mock("@xterm/addon-webgl", () => ({ WebglAddon: FakeWebglAddon }));
vi.mock("@xterm/xterm", () => ({ Terminal: FakeTerminal }));
vi.mock("@/components/ChatSidebar", () => ({
  ChatSidebar: () => null,
}));
vi.mock("@/components/ChatSessionList", () => ({
  ChatSessionList: () => null,
}));
vi.mock("@/components/Backdrop", () => ({ Backdrop: () => null }));
vi.mock("@/plugins", () => ({
  PluginSlot: () => null,
}));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn(), setTitle: vi.fn() }),
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: "" }),
}));
vi.mock("@/themes", () => ({
  useTheme: () => ({ theme: { terminalBackground: "#000000" } }),
}));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: {
      app: {
        closeModelTools: "Close model tools",
        modelToolsSheetSubtitle: "Tools",
        modelToolsSheetTitle: "Model",
      },
    },
  }),
}));
vi.mock("@/lib/dashboard-auth-reload", () => ({
  maybeReloadForLoopbackWsAuthFailure,
}));
vi.mock("@/lib/api", () => ({
  api: apiMocks,
  buildWsUrl: apiMocks.buildWsUrl,
}));

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  binaryType = "blob";
  closeCalls: Array<{ code?: number; reason?: string }> = [];
  onclose: ((event: CloseEventLike) => void) | null = null;
  onmessage: ((event: { data: ArrayBuffer | string }) => void) | null = null;
  onopen: (() => void) | null = null;
  readyState = FakeWebSocket.OPEN;
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close(code?: number, reason?: string) {
    this.closeCalls.push({ code, reason });
    this.readyState = FakeWebSocket.CLOSED;
  }

  send() {}
}

type CloseEventLike = {
  code: number;
  reason: string;
  wasClean: boolean;
};

let container: HTMLDivElement;
let root: Root;

// jsdom runs without an origin here (per-file @vitest-environment jsdom on a
// node-default config), so localStorage is undefined. Stub it so components
// that persist UI state (side panel collapse) can be exercised.
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => {
      store[key] = String(value);
    },
    removeItem: (key: string) => {
      delete store[key];
    },
    clear: () => {
      store = {};
    },
  };
})();

// React only routes updates through act() when this flag is set; without it
// the isActive re-renders in the keyboard-inset gate test warn.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  FakeTerminal.instances = [];
  maybeReloadForLoopbackWsAuthFailure.mockClear();
  apiMocks.buildWsUrl.mockReset();
  apiMocks.buildWsUrl.mockImplementation(
    async (path: string, params?: Record<string, string>) => {
      const url = new URL(path, "ws://localhost");
      for (const [key, value] of Object.entries(params ?? {})) {
        url.searchParams.set(key, value);
      }
      return url.toString();
    },
  );
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    value: "visible",
  });
  vi.stubGlobal("WebSocket", FakeWebSocket);
  vi.stubGlobal(
    "ResizeObserver",
    class {
      disconnect() {}
      observe() {}
      unobserve() {}
    },
  );
  vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => {
    cb(0);
    return 1;
  });
  vi.stubGlobal("cancelAnimationFrame", () => {});
  vi.stubGlobal("matchMedia", () => ({
    addEventListener() {},
    matches: false,
    media: "",
    removeEventListener() {},
  }));
  vi.stubGlobal("crypto", {
    getRandomValues: (values: Uint8Array) => {
      values.fill(7);
      return values;
    },
    randomUUID: () => "chat-test-id",
  });

  Object.defineProperty(window, "visualViewport", {
    configurable: true,
    value: { addEventListener() {}, removeEventListener() {}, width: 1280 },
  });
  Object.defineProperty(window, "__HERMES_SESSION_TOKEN__", {
    configurable: true,
    value: "stale-token",
    writable: true,
  });
  Object.defineProperty(window, "__HERMES_AUTH_REQUIRED__", {
    configurable: true,
    value: false,
    writable: true,
  });
  Object.defineProperty(window.navigator, "clipboard", {
    configurable: true,
    value: {
      readText: vi.fn(async () => ""),
      writeText: vi.fn(async () => {}),
    },
  });
  sessionStorage.clear();
  vi.stubGlobal("localStorage", localStorageMock);
  localStorageMock.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("ChatPage", () => {
  it("treats loopback 4401 closes as stale-token reload candidates", async () => {
    const { default: ChatPage } = await import("./ChatPage");

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    FakeWebSocket.instances[0].onclose?.({
      code: 4401,
      reason: "auth: token_mismatch",
      wasClean: true,
    });

    expect(maybeReloadForLoopbackWsAuthFailure).toHaveBeenCalledWith(4401);
  });

  it("attaches visualViewport keyboard-inset listeners only while the chat tab is active", async () => {
    // NS-434 follow-up: ChatPage stays mounted (hidden) on every dashboard
    // route. The keyboard-inset/scroll-pin listeners must only be live while
    // /chat is the active tab, or the scroll pin fires when a soft keyboard
    // opens on Settings etc.
    const addEventListener = vi.fn();
    const removeEventListener = vi.fn();
    Object.defineProperty(window, "visualViewport", {
      configurable: true,
      value: { addEventListener, removeEventListener, width: 1280 },
    });

    const { default: ChatPage } = await import("./ChatPage");

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive={false} />
      </MemoryRouter>,
    );
    expect(addEventListener).not.toHaveBeenCalled();

    await act(async () =>
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ChatPage isActive />
        </MemoryRouter>,
      ),
    );
    expect(addEventListener.mock.calls.map((c) => c[0]).sort()).toEqual([
      "resize",
      "scroll",
    ]);
    expect(removeEventListener).not.toHaveBeenCalled();

    await act(async () =>
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ChatPage isActive={false} />
        </MemoryRouter>,
      ),
    );
    expect(removeEventListener.mock.calls.map((c) => c[0]).sort()).toEqual([
      "resize",
      "scroll",
    ]);
  });

  it("resumes the same PTY epoch using raw binary byte offsets", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    let now = 5_000;
    vi.spyOn(Date, "now").mockImplementation(() => now);

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    const epoch = "a".repeat(32);
    const first = FakeWebSocket.instances[0];
    await act(async () => first.onopen?.());
    await act(async () =>
      first.onmessage?.({
        data: JSON.stringify({
          type: "pty.replay",
          epoch,
          start_offset: 0,
          replay_end_offset: 0,
          reset: true,
          reason: "initial",
        }),
      }),
    );
    // First half of "é" advances one raw byte even though TextDecoder has
    // not emitted a JavaScript character yet.
    await act(async () =>
      first.onmessage?.({ data: new Uint8Array([0xc3]).buffer }),
    );

    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    await act(async () => document.dispatchEvent(new Event("visibilitychange")));
    await act(async () =>
      first.onclose?.({ code: 1000, reason: "tab hidden", wasClean: true }),
    );

    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
    await act(async () => document.dispatchEvent(new Event("visibilitychange")));
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2));

    const second = FakeWebSocket.instances[1];
    const secondUrl = new URL(second.url);
    expect(secondUrl.searchParams.get("epoch")).toBe(epoch);
    expect(secondUrl.searchParams.get("offset")).toBe("1");
    expect(FakeTerminal.instances).toHaveLength(1);

    await act(async () => second.onopen?.());
    await act(async () =>
      second.onmessage?.({
        data: JSON.stringify({
          type: "pty.replay",
          epoch,
          start_offset: 1,
          replay_end_offset: 1,
          reset: false,
          reason: "resume",
        }),
      }),
    );
    await act(async () =>
      second.onmessage?.({ data: new Uint8Array([0xa9, 0xff]).buffer }),
    );
    expect(FakeTerminal.instances[0].writes).toContain("é�");

    now += 2_000;
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    await act(async () => document.dispatchEvent(new Event("visibilitychange")));
    await act(async () =>
      second.onclose?.({ code: 1000, reason: "tab hidden", wasClean: true }),
    );
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
    await act(async () => document.dispatchEvent(new Event("visibilitychange")));
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(3));
    expect(new URL(FakeWebSocket.instances[2].url).searchParams.get("offset")).toBe(
      "3",
    );
  });

  it("attaches visualViewport keyboard-inset listeners only while the chat tab is active", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    const root = createRoot(container!);
    const addEventListener = vi.fn();
    const removeEventListener = vi.fn();
    Object.defineProperty(window, "visualViewport", {
      configurable: true,
      value: { addEventListener, removeEventListener, width: 1280 },
    });

    await act(async () =>
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ChatPage isActive={false} />
        </MemoryRouter>,
      ),
    );
    expect(addEventListener).not.toHaveBeenCalled();

    await act(async () =>
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ChatPage isActive />
        </MemoryRouter>,
      ),
    );
    expect(addEventListener.mock.calls.map((c) => c[0]).sort()).toEqual([
      "resize",
      "scroll",
    ]);
    expect(removeEventListener).not.toHaveBeenCalled();

    await act(async () =>
      root.render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ChatPage isActive={false} />
        </MemoryRouter>,
      ),
    );
    expect(removeEventListener.mock.calls.map((c) => c[0]).sort()).toEqual([
      "resize",
      "scroll",
    ]);

  });

  it("binds lifecycle suspension to the socket that was closed", async () => {
    const { default: ChatPage } = await import("./ChatPage");

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    const first = FakeWebSocket.instances[0];
    await act(async () => first.onopen?.());
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "hidden",
    });
    await act(async () => document.dispatchEvent(new Event("visibilitychange")));

    // Resume before the old socket delivers its delayed close event.
    Object.defineProperty(document, "visibilityState", {
      configurable: true,
      value: "visible",
    });
    await act(async () => document.dispatchEvent(new Event("visibilitychange")));
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2));
    const second = FakeWebSocket.instances[1];
    await act(async () => second.onopen?.());

    const writesBeforeStaleFrame = FakeTerminal.instances[0].writes.length;
    await act(async () =>
      first.onmessage?.({ data: new Uint8Array([0x78]).buffer }),
    );
    expect(FakeTerminal.instances[0].writes).toHaveLength(
      writesBeforeStaleFrame,
    );

    // A clean close from the replacement is a real session end; it must not
    // consume the old socket's pending lifecycle-suspend marker.
    await act(async () =>
      second.onclose?.({ code: 1000, reason: "", wasClean: true }),
    );
    expect(container.textContent).toContain("Session ended.");

    await act(async () =>
      first.onclose?.({ code: 1000, reason: "tab hidden", wasClean: true }),
    );
    expect(container.textContent).toContain("Session ended.");
  });
});

describe("ChatPage side panel collapse", () => {
  async function renderChat() {
    const { default: ChatPage } = await import("./ChatPage");
    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );
  }

  it("collapses the desktop side panel and persists the choice", async () => {
    localStorage.clear();
    await renderChat();
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    const collapseButton = container.querySelector(
      '[aria-label="Collapse chat side panel"]',
    );
    expect(collapseButton).not.toBeNull();

    await act(async () => {
      collapseButton!.dispatchEvent(
        new MouseEvent("click", { bubbles: true }),
      );
    });

    expect(localStorage.getItem("hermes-chat-panel-collapsed")).toBe("1");
    expect(
      container.querySelector('[aria-label="Collapse chat side panel"]'),
    ).toBeNull();
    expect(
      container.querySelector('[aria-label="Show chat side panel"]'),
    ).not.toBeNull();

    // Reopening restores the panel and clears the persisted flag.
    await act(async () => {
      container
        .querySelector('[aria-label="Show chat side panel"]')!
        .dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });

    expect(localStorage.getItem("hermes-chat-panel-collapsed")).toBe("0");
    expect(
      container.querySelector('[aria-label="Collapse chat side panel"]'),
    ).not.toBeNull();
  });
});

// The gated-mode ticket request runs before any socket exists, so a rejection
// or a hang emits no `close` event and never arms PTY_CONNECTING_TIMEOUT_MS
// (that timer is set after `new WebSocket`). Without its own deadline the tab
// strands on "connecting" with no retry. Mirrors the ChatSidebar events-feed
// coverage in src/components/ChatSidebar.test.tsx.
describe("ChatPage PTY ticket connect deadline", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function renderChat() {
    const { default: ChatPage } = await import("./ChatPage");
    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );
  }

  /** Advance timers and flush the async connect that fires on the tick. */
  async function advance(ms: number) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  it("retries when the ticket request rejects", async () => {
    apiMocks.buildWsUrl.mockRejectedValueOnce(
      new Error("ticket endpoint unavailable"),
    );

    await renderChat();
    await advance(0);
    expect(FakeWebSocket.instances).toHaveLength(0);

    // First backoff step is 250ms; the retry must mint a fresh ticket.
    await advance(250);
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(2);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("times out a stalled ticket request and retries", async () => {
    let resolveStalledRequest!: (url: string) => void;
    apiMocks.buildWsUrl.mockImplementationOnce(
      () =>
        new Promise<string>((resolve) => {
          resolveStalledRequest = resolve;
        }),
    );

    await renderChat();
    await advance(0);
    expect(FakeWebSocket.instances).toHaveLength(0);

    await advance(PTY_TICKET_TIMEOUT_MS);
    expect(FakeWebSocket.instances).toHaveLength(0);

    // A late ticket from the timed-out attempt must not open a socket behind
    // the replacement the deadline scheduled.
    await act(async () => {
      resolveStalledRequest("ws://localhost/api/pty?channel=stale");
      await Promise.resolve();
    });
    expect(FakeWebSocket.instances).toHaveLength(0);

    await advance(250);
    expect(FakeWebSocket.instances).toHaveLength(1);
    expect(FakeWebSocket.instances[0].url).not.toContain("channel=stale");
  });

  it("leaves a settled ticket's socket to the CONNECTING timer", async () => {
    await renderChat();
    await advance(0);
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    // NS-591 regression: once the socket exists the ticket deadline is
    // disarmed, so PTY_CONNECTING_TIMEOUT_MS stays the only thing that may
    // force-close a wedged handshake — the two must not both fire.
    await advance(PTY_TICKET_TIMEOUT_MS);
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(1);
  });
});
