// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * Behaviour tests for the chat composer's screen-reader accessibility
 * (#36784). Unlike source-text assertions, these render ChatPage and observe
 * what assistive technology would observe:
 *
 *   - the xterm `Terminal` is CONSTRUCTED with `screenReaderMode: true`, the
 *     option that makes xterm create the offscreen textarea VoiceOver/NVDA
 *     need to discover and drive the composer;
 *   - the terminal host element is exposed in the accessibility tree as a
 *     labelled region (`role="region"` + non-empty `aria-label`);
 *   - the single registered custom key handler lets Tab/Shift-Tab escape the
 *     terminal (WCAG 2.1.2 No Keyboard Trap) while passing ordinary keys and
 *     non-keydown events through to the terminal.
 */

class FakeFitAddon {
  fit() {}
}

class FakeWebglAddon {
  onContextLoss() {
    return { dispose() {} };
  }
}

type KeyEventLike = {
  key: string;
  type: string;
};

class FakeTerminal {
  static instances: FakeTerminal[] = [];
  static keyHandlers: ((ev: KeyEventLike) => boolean)[] = [];

  options: Record<string, unknown>;
  rows = 24;
  cols = 80;
  parser = {
    registerOscHandler: vi.fn(),
  };
  unicode = { activeVersion: "" };

  constructor(options: Record<string, unknown>) {
    this.options = options;
    FakeTerminal.instances.push(this);
  }

  attachCustomKeyEventHandler(handler: (ev: KeyEventLike) => boolean) {
    FakeTerminal.keyHandlers.push(handler);
    return true;
  }

  attachCustomWheelEventHandler() {
    return true;
  }

  clearSelection() {}

  clearTextureAtlas() {}

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
    return { active: { baseY: 0, viewportY: 0 } };
  }

  scrollToBottom() {}

  open() {}

  paste() {}

  refresh() {}

  write() {}
}

const apiMocks = vi.hoisted(() => ({
  buildWsUrl: vi.fn(async () => "ws://localhost/api/pty?channel=chat-1"),
}));

vi.mock("@/lib/chatImagePaste", () => ({
  imageFilesFromTransfer: () => [],
  transferMayContainImage: () => false,
  uploadChatImage: vi.fn(),
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
  maybeReloadForLoopbackWsAuthFailure: vi.fn(() => false),
}));
vi.mock("@/lib/api", () => ({
  api: apiMocks,
  buildWsUrl: apiMocks.buildWsUrl,
}));

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static OPEN = 1;

  binaryType = "blob";
  onclose: ((event: { code: number; reason: string; wasClean: boolean }) => void) | null = null;
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

  send = vi.fn();
}

let container: HTMLDivElement;
let root: Root;

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

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

async function render(ui: ReactNode) {
  container = document.createElement("div");
  document.body.append(container);
  root = createRoot(container);
  await act(async () => root.render(ui));
}

beforeEach(() => {
  FakeTerminal.instances = [];
  FakeTerminal.keyHandlers = [];
  FakeWebSocket.instances = [];
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
    randomUUID: () => "chat-a11y-test-id",
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
  vi.stubGlobal("localStorage", localStorageMock);
  localStorageMock.clear();
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("ChatPage composer screen-reader accessibility (#36784)", () => {
  it("constructs the xterm Terminal with screenReaderMode so AT gets an editable textarea", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    expect(FakeTerminal.instances).toHaveLength(1);
    // The constructor option, not a post-hoc DOM patch: xterm only creates
    // the offscreen textarea screen readers need when this is true.
    expect(FakeTerminal.instances[0].options.screenReaderMode).toBe(true);
  });

  it("exposes the terminal host as a labelled region in the accessibility tree", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const host = container.querySelector(".hermes-chat-xterm-host");
    expect(host).not.toBeNull();
    expect(host!.getAttribute("role")).toBe("region");
    expect(host!.getAttribute("aria-label")).toBeTruthy();
  });

  it("lets Tab escape the terminal from the single registered key handler", async () => {
    const { default: ChatPage } = await import("./ChatPage");
    await render(
      <MemoryRouter initialEntries={["/chat"]}>
        <ChatPage isActive />
      </MemoryRouter>,
    );

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    // xterm stores ONE custom key handler per Terminal; a second registration
    // silently replaces the first (this exact regression shipped in the first
    // revision of #59091). Assert the count so the clipboard handler can
    // never be overwritten by a well-meaning addition.
    expect(FakeTerminal.keyHandlers).toHaveLength(1);

    const handler = FakeTerminal.keyHandlers[0];
    // Tab must be returned to the browser (handler returns false) so focus
    // can leave the terminal — WCAG 2.1.2 No Keyboard Trap.
    expect(handler({ key: "Tab", type: "keydown" })).toBe(false);
    expect(handler({ key: "Tab", type: "keyup" })).toBe(true);
    // Ordinary typing still flows into the terminal untouched.
    expect(handler({ key: "a", type: "keydown" })).toBe(true);
  });
});
