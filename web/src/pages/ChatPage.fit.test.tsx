import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { act } from "react";
import { render, cleanup } from "@testing-library/react";

// --- Mock the heavy xterm + UI surface so we can mount ChatPage in happy-dom ---

const fitMock = vi.fn();
const proposeDimensionsMock = vi.fn();
const refreshMock = vi.fn();
const writeMock = vi.fn();
const onDataMock = vi.fn(() => ({ dispose: vi.fn() }));

vi.mock("@xterm/xterm", () => ({
  Terminal: class {
    options: Record<string, unknown> = { fontSize: 14, lineHeight: 1 };
    cols = 80;
    rows = 24;
    css = "";
    js = "";
    unicode = { activeVersion: "11" };
    parser = { registerOscHandler: vi.fn() };
    open() {}
    write = writeMock;
    refresh = refreshMock;
    onData = onDataMock;
    onResize = () => ({ dispose: vi.fn() });
    attachCustomKeyEventHandler() {}
    attachCustomWheelEventHandler() {}
    clearSelection() {}
    focus() {}
    getSelection = () => "";
    paste() {}
    scrollLines() {}
    loadAddon() {}
    dispose() {}
  },
}));

vi.mock("@xterm/addon-fit", () => ({
  FitAddon: class {
    fit = fitMock;
    proposeDimensions = proposeDimensionsMock;
  },
}));

vi.mock("@xterm/addon-unicode11", () => ({
  Unicode11Addon: class {},
}));
vi.mock("@xterm/addon-web-links", () => ({
  WebLinksAddon: class {},
}));
vi.mock("@xterm/addon-webgl", () => ({
  WebglAddon: class {
    dispose() {}
  },
}));
vi.mock("lucide-react", () => ({
  Copy: () => null,
  PanelRight: () => null,
  X: () => null,
}));
vi.mock("@nous-research/ui", () => ({
  Button: () => null,
  Typography: () => null,
}));
vi.mock("@/components/ChatSidebar", () => ({ ChatSidebar: () => null }));
vi.mock("@/contexts/usePageHeader", () => ({
  usePageHeader: () => ({ setEnd: vi.fn() }),
}));
vi.mock("@/contexts/useProfileScope", () => ({
  useProfileScope: () => ({ profile: "default" }),
}));
vi.mock("@/i18n", () => ({
  useI18n: () => ({
    t: new Proxy(function () {}, {
      get: () => "",
      apply: () => "",
    }) as unknown as (...a: unknown[]) => string,
  }),
}));
vi.mock("@/themes", () => ({ useTheme: () => ({ theme: "dark" }) }));
vi.mock("@/plugins", () => ({ PluginSlot: () => null }));
vi.mock("@/lib/utils", () => ({ cn: (...c: unknown[]) => c.join(" ") }));
vi.mock("@/lib/api", () => ({
  api: { get: vi.fn() },
  HERMES_BASE_PATH: "",
  buildWsAuthParam: () => "",
  buildWsUrl: () => "ws://localhost/ws",
}));
vi.mock("react-router-dom", () => ({
  useSearchParams: () => [new URLSearchParams()],
}));

// WebSocket mock that captures the onmessage handler so we can simulate
// PTY chunks (the resumed-session path). onopen/onmessage are attached
// synchronously-ish via a macrotask so the component can register handlers
// first; under real timers act() flushes them.
class MockWebSocket {
  static OPEN = 1;
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: ((ev: unknown) => void) | null = null;
  constructor() {
    // Expose the instance so tests can drive onmessage directly, without
    // depending on the deferred onopen wiring or the component's ws-open
    // gating (auth/token resolution).
    (globalThis as unknown as { __mockWs?: MockWebSocket }).__mockWs = this;
    setTimeout(() => {
      this.onopen?.();
    }, 0);
  }
  send() {}
  close() {}
}
vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);

import ChatPage from "./ChatPage";

const tick = (ms: number) => new Promise((r) => setTimeout(r, ms));

describe("ChatPage xterm fit (PR #47390 / #47772)", () => {
  beforeEach(() => {
    fitMock.mockClear();
    proposeDimensionsMock.mockClear();
    writeMock.mockClear();
    // Host measures a DIFFERENT size than the terminal's current grid
    // (a real desync) — the observed-condition guard SHOULD refit.
    proposeDimensionsMock.mockReturnValue({ cols: 100, rows: 30 });
  });
  afterEach(() => {
    cleanup();
  });

  it("refits after the CSS transition window (post-mount fit, #47313)", async () => {
    await act(async () => {
      render(<ChatPage isActive />);
      await tick(700); // let rAF settle + 600ms post-mount timer fire
    });
    expect(fitMock).toHaveBeenCalled();
  });

  it("only refits when the grid actually desyncs (observed-condition guard)", async () => {
    // Host already matches: proposeDimensions == current grid -> the
    // post-mount timer must NOT trigger an additional fit (this is exactly
    // the blind-fit behavior #47772 showed produced the blank viewport).
    proposeDimensionsMock.mockReturnValue({ cols: 80, rows: 24 });
    await act(async () => {
      render(<ChatPage isActive />);
      await tick(700);
    });
    // On initial mount a fit may run once to establish the grid; the guard
    // must ensure the post-mount timer adds NO further fit when in sync.
    expect(fitMock.mock.calls.length).toBeLessThanOrEqual(1);
  });

  it("clears both timers on unmount (no leaked post-mount / data-fit timer)", async () => {
    await act(async () => {
      render(<ChatPage isActive />);
      await tick(50);
    });
    const beforeUnmount = fitMock.mock.calls.length;
    await act(async () => {
      cleanup();
      await tick(700); // would have fired the 600ms timer if not cleared
    });
    // No further fit after unmount (timers cleared).
    expect(fitMock.mock.calls.length).toBe(beforeUnmount);
  });
});
