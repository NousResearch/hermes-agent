// @vitest-environment jsdom
import { act, useEffect, useState, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { MemoryRouter, useLocation, useNavigate } from "react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { FakeFitAddon, FakeTerminal, FakeWebglAddon } = vi.hoisted(() => {
  class HoistedFitAddon {
    fit() {}
  }

  class HoistedWebglAddon {
    onContextLoss() {
      return { dispose() {} };
    }
    dispose() {}
  }

  class HoistedTerminal {
    static latestDataHandler: ((data: string) => void) | null = null;
    options: Record<string, unknown>;
    rows = 24;
    cols = 80;
    parser = { registerOscHandler: vi.fn() };
    unicode = { activeVersion: "" };

    constructor(options: Record<string, unknown>) {
      this.options = options;
    }

    attachCustomKeyEventHandler() {
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
    onData(handler: (data: string) => void) {
      HoistedTerminal.latestDataHandler = handler;
      return {
        dispose() {
          if (HoistedTerminal.latestDataHandler === handler) {
            HoistedTerminal.latestDataHandler = null;
          }
        },
      };
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

  return {
    FakeFitAddon: HoistedFitAddon,
    FakeTerminal: HoistedTerminal,
    FakeWebglAddon: HoistedWebglAddon,
  };
});

const apiMocks = vi.hoisted(() => ({
  buildWsUrl: vi.fn(async () => "ws://localhost/api/pty?channel=chat-1"),
  getProfiles: vi.fn(() => new Promise(() => {})),
  getActiveProfile: vi.fn(() => new Promise(() => {})),
}));

vi.mock("@xterm/addon-fit", () => ({ FitAddon: FakeFitAddon }));
vi.mock("@xterm/addon-unicode11", () => ({ Unicode11Addon: class {} }));
vi.mock("@xterm/addon-web-links", () => ({ WebLinksAddon: class {} }));
vi.mock("@xterm/addon-webgl", () => ({ WebglAddon: FakeWebglAddon }));
vi.mock("@xterm/xterm", () => ({ Terminal: FakeTerminal }));
vi.mock("@/components/ChatSidebar", () => ({ ChatSidebar: () => null }));
vi.mock("@/components/ChatSessionList", () => ({ ChatSessionList: () => null }));
vi.mock("@/components/Backdrop", () => ({ Backdrop: () => null }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn(), setTitle: vi.fn() }),
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
  setManagementProfile: vi.fn(),
}));

import { ProfileProvider } from "@/contexts/ProfileProvider";
import ChatPage from "./ChatPage";

type CloseEventLike = { code: number; reason: string; wasClean: boolean };

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static OPEN = 1;

  binaryType = "blob";
  onclose: ((event: CloseEventLike) => void) | null = null;
  onmessage: ((event: { data: ArrayBuffer | string }) => void) | null = null;
  onopen: (() => void) | null = null;
  readyState = FakeWebSocket.OPEN;
  url: string;
  send = vi.fn();

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  close() {
    this.readyState = 3;
  }
}

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

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
  FakeTerminal.latestDataHandler = null;
  apiMocks.buildWsUrl.mockReset();
  apiMocks.buildWsUrl.mockResolvedValue("ws://localhost/api/pty?channel=chat-1");
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
    randomUUID: () => "chat-learn-test-id",
  });
  Object.defineProperty(window, "visualViewport", {
    configurable: true,
    value: {
      addEventListener() {},
      height: 800,
      offsetTop: 0,
      removeEventListener() {},
    },
  });
  Object.defineProperty(window, "__HERMES_SESSION_TOKEN__", {
    configurable: true,
    value: "test-token",
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
  const store: Record<string, string> = {};
  vi.stubGlobal("localStorage", {
    getItem: (k: string) => store[k] ?? null,
    setItem: (k: string, v: string) => {
      store[k] = String(v);
    },
    removeItem: (k: string) => {
      delete store[k];
    },
    clear: () => {
      for (const k of Object.keys(store)) delete store[k];
    },
  });
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("ChatPage /learn router-state handoff", () => {
  it("ignores ?learn= URL query parameters and never submits them to the PTY", async () => {
    vi.useFakeTimers();
    try {
      await render(
        <MemoryRouter initialEntries={["/chat?learn=%15!id%0d&other=1"]}>
          <ProfileProvider>
            <ChatPage />
          </ProfileProvider>
        </MemoryRouter>,
      );
      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const socket = FakeWebSocket.instances[0];
      await act(async () => socket.onopen?.());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500);
      });

      const sent = socket.send.mock.calls.map(([payload]) => String(payload));
      expect(sent.filter((msg) => msg.includes("/learn"))).toEqual([]);
      expect(sent.filter((msg) => msg.includes("!id"))).toEqual([]);
    } finally {
      vi.useRealTimers();
    }
  });

  it("delivers normalized router-state handoff once on cold start and strips learnSeed", async () => {
    vi.useFakeTimers();
    try {
      let latestLocation: { pathname: string; search: string; state: unknown } | null = null;
      function LocationSpy() {
        const loc = useLocation();
        latestLocation = { pathname: loc.pathname, search: loc.search, state: loc.state };
        return null;
      }

      await render(
        <MemoryRouter
          initialEntries={[
            {
              pathname: "/chat",
              state: {
                learnSeed: "docs \u001b]52;c;YWJj\u0007\u0015\u001b7guide\u001b8",
                from: "/skills",
              },
            },
          ]}
        >
          <ProfileProvider>
            <LocationSpy />
            <ChatPage />
          </ProfileProvider>
        </MemoryRouter>,
      );

      expect(latestLocation).toEqual({
        pathname: "/chat",
        search: "",
        state: { from: "/skills" },
      });

      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const socket = FakeWebSocket.instances[0];
      await act(async () => socket.onopen?.());

      // Terminal-generated startup replies must not cancel the pending handoff.
      await act(async () => {
        FakeTerminal.latestDataHandler?.("\u001b[1;1R");
        FakeTerminal.latestDataHandler?.("\u001b[?1;2c");
        FakeTerminal.latestDataHandler?.("\u001b[?2004;1$y");
        FakeTerminal.latestDataHandler?.("\u001b[I");
      });

      await act(async () => {
        await vi.advanceTimersByTimeAsync(799);
      });
      expect(socket.send).not.toHaveBeenCalledWith(expect.stringContaining("/learn"));

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1);
      });
      expect(socket.send).toHaveBeenCalledWith("/learn docs guide\r");
      expect(
        socket.send.mock.calls.filter(([payload]) => String(payload).startsWith("/learn")),
      ).toHaveLength(1);
    } finally {
      vi.useRealTimers();
    }
  });

  it("cancels a pending cold-start handoff when the user pastes a draft via bracketed paste", async () => {
    vi.useFakeTimers();
    try {
      await render(
        <MemoryRouter
          initialEntries={[
            {
              pathname: "/chat",
              state: { learnSeed: "startup skill" },
            },
          ]}
        >
          <ProfileProvider>
            <ChatPage />
          </ProfileProvider>
        </MemoryRouter>,
      );

      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const socket = FakeWebSocket.instances[0];
      await act(async () => socket.onopen?.());

      await act(async () => {
        await vi.advanceTimersByTimeAsync(300);
        FakeTerminal.latestDataHandler?.("\u001b[200~pasted draft\u001b[201~");
      });
      expect(socket.send).toHaveBeenCalledWith("\u001b[200~pasted draft\u001b[201~");
      socket.send.mockClear();

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1000);
      });
      expect(socket.send).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("retains an unsent handoff across a transient early close/reconnect and never replays after delivery", async () => {
    vi.useFakeTimers();
    try {
      await render(
        <MemoryRouter
          initialEntries={[
            {
              pathname: "/chat",
              state: { learnSeed: "retry workflow" },
            },
          ]}
        >
          <ProfileProvider>
            <ChatPage />
          </ProfileProvider>
        </MemoryRouter>,
      );

      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const first = FakeWebSocket.instances[0];
      await act(async () => first.onopen?.());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(300);
      });
      expect(first.send).not.toHaveBeenCalledWith(expect.stringContaining("/learn"));

      await act(async () => {
        first.readyState = 3;
        first.onclose?.({ code: 1006, reason: "", wasClean: false });
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(2));

      const second = FakeWebSocket.instances[1];
      await act(async () => second.onopen?.());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(800);
      });
      expect(first.send).not.toHaveBeenCalledWith(expect.stringContaining("/learn"));
      expect(second.send).toHaveBeenCalledWith("/learn retry workflow\r");

      await act(async () => {
        second.readyState = 3;
        second.onclose?.({ code: 1006, reason: "", wasClean: false });
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(3));
      const third = FakeWebSocket.instances[2];
      await act(async () => third.onopen?.());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(800);
      });
      expect(third.send).not.toHaveBeenCalledWith(expect.stringContaining("/learn"));
    } finally {
      vi.useRealTimers();
    }
  });

  it("leaves a pre-existing draft untouched and never submits on warm return to an open PTY", async () => {
    vi.useFakeTimers();
    try {
      let pushLearnReturn: (() => void) | null = null;
      function NavDriver() {
        const navigate = useNavigate();
        pushLearnReturn = () =>
          navigate("/chat", { state: { learnSeed: "warm skill handoff" } });
        return null;
      }

      await render(
        <MemoryRouter initialEntries={["/chat"]}>
          <ProfileProvider>
            <NavDriver />
            <ChatPage />
          </ProfileProvider>
        </MemoryRouter>,
      );

      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const socket = FakeWebSocket.instances[0];
      await act(async () => socket.onopen?.());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(800);
      });

      await act(async () => {
        FakeTerminal.latestDataHandler?.("unfinished draft");
      });
      expect(socket.send).toHaveBeenCalledWith("unfinished draft");
      socket.send.mockClear();

      await act(async () => {
        pushLearnReturn?.();
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(1500);
      });
      expect(socket.send).not.toHaveBeenCalled();
    } finally {
      vi.useRealTimers();
    }
  });

  it("preserves learnSeed across real ProfileProvider ?profile= sync and delayed ChatPage mount", async () => {
    vi.useFakeTimers();
    try {
      let navigateToChat: (() => void) | null = null;
      let releaseLazyMount: (() => void) | null = null;

      // Mirrors App.tsx: chat host latches on /chat activation and ChatPage
      // finishes mounting asynchronously (lazy chunk load) after ProfileProvider's
      // ?profile= URL sync effect has already run.
      function DelayedChatHost() {
        const location = useLocation();
        const navigate = useNavigate();
        const isChat = location.pathname === "/chat";
        const [latched, setLatched] = useState(isChat);
        const [lazyLoaded, setLazyLoaded] = useState(false);

        useEffect(() => {
          navigateToChat = () =>
            navigate("/chat", { state: { learnSeed: "profile preserved skill" } });
          releaseLazyMount = () => setLazyLoaded(true);
        }, [navigate]);

        useEffect(() => {
          if (isChat) setLatched(true);
        }, [isChat]);

        return latched && lazyLoaded ? <ChatPage isActive={isChat} /> : null;
      }

      await render(
        <MemoryRouter initialEntries={["/skills?profile=work"]}>
          <ProfileProvider>
            <DelayedChatHost />
          </ProfileProvider>
        </MemoryRouter>,
      );

      // Navigate bare to /chat with state; ProfileProvider syncs ?profile=work
      // before ChatPage mounts.
      await act(async () => {
        navigateToChat?.();
      });
      await act(async () => {
        releaseLazyMount?.();
      });

      await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
      const socket = FakeWebSocket.instances[0];
      await act(async () => socket.onopen?.());
      await act(async () => {
        await vi.advanceTimersByTimeAsync(800);
      });

      expect(socket.send).toHaveBeenCalledWith("/learn profile preserved skill\r");
    } finally {
      vi.useRealTimers();
    }
  });
});
