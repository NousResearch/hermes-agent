// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

class FakeFitAddon {
  static instances: FakeFitAddon[] = [];
  fit = vi.fn();

  constructor() {
    FakeFitAddon.instances.push(this);
  }
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
  parser = {
    registerOscHandler: vi.fn(),
  };
  unicode = { activeVersion: "" };
  refresh = vi.fn();

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

  open(host: HTMLElement) {
    // Mimic xterm.js: a scrollable .xterm-viewport child is created so
    // the touch-scroll handler can locate it (#81119).
    const viewport = document.createElement("div");
    viewport.className = "xterm-viewport";
    host.appendChild(viewport);
  }

  paste() {}

  write() {}
}

const maybeReloadForLoopbackWsAuthFailure = vi.fn(() => false);

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

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static OPEN = 1;

  binaryType = "blob";
  onclose: ((event: CloseEventLike) => void) | null = null;
  onmessage: ((event: { data: ArrayBuffer | string }) => void) | null = null;
  onopen: (() => void) | null = null;
  readyState = FakeWebSocket.OPEN;
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close() {
    this.readyState = 3;
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

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  FakeWebSocket.instances = [];
  FakeFitAddon.instances = [];
  FakeTerminal.instances = [];
  maybeReloadForLoopbackWsAuthFailure.mockClear();
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
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
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

  it("lets touch swipes scroll the xterm scrollback natively (#81119)", async () => {
    const { default: ChatPage } = await import("./ChatPage");

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );

    const host = document.querySelector(
      ".hermes-chat-xterm-host",
    ) as HTMLElement | null;
    expect(host).not.toBeNull();
    const viewport = host!.querySelector<HTMLElement>(".xterm-viewport");
    expect(viewport).not.toBeNull();
    // The terminal pane explicitly opts in to native vertical panning so the
    // browser owns the scroll gesture.
    expect(viewport!.style.touchAction).toBe("pan-y");

    // Simulate xterm.js's own target-phase handler, which would call
    // preventDefault() on touchmove and swallow the swipe. The capture-phase
    // listener must stop propagation before that handler fires.
    const xtermTouchMove = vi.fn((ev: Event) => ev.preventDefault());
    viewport!.addEventListener("touchmove", xtermTouchMove);

    const child = document.createElement("div");
    viewport!.appendChild(child);

    const dispatch = () => {
      const ev = new Event("touchmove", { bubbles: true, cancelable: true });
      Object.defineProperty(ev, "touches", {
        configurable: true,
        value: [{ identifier: 1 }],
      });
      child.dispatchEvent(ev);
      return ev;
    };
    const ev = dispatch();

    expect(xtermTouchMove).not.toHaveBeenCalled();
    expect(ev.defaultPrevented).toBe(false);

    // A direct touchmove on the viewport itself (no child) must also be
    // intercepted — xterm's preventDefault is bound on the viewport.
    viewport!.removeEventListener("touchmove", xtermTouchMove);
    viewport!.addEventListener("touchmove", xtermTouchMove);
    const ev2 = new Event("touchmove", { bubbles: true, cancelable: true });
    Object.defineProperty(ev2, "touches", {
      configurable: true,
      value: [{ identifier: 1 }],
    });
    viewport!.dispatchEvent(ev2);
    expect(xtermTouchMove).not.toHaveBeenCalled();
    expect(ev2.defaultPrevented).toBe(false);
  });

  it("refits and repaints the terminal after a viewport resize (#81119)", async () => {
    const { default: ChatPage } = await import("./ChatPage");

    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );

    const host = document.querySelector(
      ".hermes-chat-xterm-host",
    ) as HTMLElement | null;
    expect(host).not.toBeNull();

    // jsdom doesn't lay out elements, so clientWidth/Height are 0 by
    // default — the metrics sync early-returns while hidden.  Mock the
    // layout to simulate a mobile viewport and trigger a real refit.
    Object.defineProperty(host!, "clientWidth", {
      configurable: true,
      value: 800,
    });
    Object.defineProperty(host!, "clientHeight", {
      configurable: true,
      value: 600,
    });

    const terminal = FakeTerminal.instances[FakeTerminal.instances.length - 1];
    const fitAddon = FakeFitAddon.instances[FakeFitAddon.instances.length - 1];
    fitAddon.fit.mockClear();
    terminal.refresh.mockClear();

    act(() => {
      window.dispatchEvent(new Event("resize"));
    });

    // scheduleSyncTerminalMetrics debounces 60ms before calling fit.
    await vi.waitFor(() => expect(fitAddon.fit).toHaveBeenCalled());

    // The width changed enough to cross a font tier, so the explicit
    // refresh must fire as well — otherwise the canvas can stay stale on
    // touch devices until something else forces a repaint.
    expect(terminal.refresh).toHaveBeenCalledWith(0, terminal.rows - 1);
  });
});
