// @vitest-environment jsdom
import { act, type ReactNode } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const apiMocks = vi.hoisted(() => ({
  buildWsUrl: vi.fn(async () => "ws://localhost/api/events?channel=chat-1"),
  getModelInfo: vi.fn(async () => ({
    capabilities: { supports_reasoning: false },
    model: "test/model",
  })),
}));

const gatewayMocks = vi.hoisted(() => {
  const handlers = new Map<string, (event: unknown) => void>();
  return {
    close: vi.fn(),
    connect: vi.fn(async () => undefined),
    handlers,
    on: vi.fn((event: string, handler: (event: unknown) => void) => {
      handlers.set(event, handler);
      return () => handlers.delete(event);
    }),
    onState: vi.fn((handler: (state: string) => void) => {
      handler("open");
      return () => undefined;
    }),
    request: vi.fn(async () => ({ session_id: "sidecar-1" })),
  };
});

const reloadMocks = vi.hoisted(() => ({
  maybeReloadForLoopbackWsAuthFailure: vi.fn(() => true),
}));

vi.mock("@/lib/api", () => ({
  api: { getModelInfo: apiMocks.getModelInfo },
  buildWsUrl: apiMocks.buildWsUrl,
}));
vi.mock("@/lib/dashboard-auth-reload", () => ({
  maybeReloadForLoopbackWsAuthFailure:
    reloadMocks.maybeReloadForLoopbackWsAuthFailure,
}));
vi.mock("@/lib/gatewayClient", () => ({
  GatewayClient: class {
    close = gatewayMocks.close;
    connect = gatewayMocks.connect;
    on = gatewayMocks.on;
    onState = gatewayMocks.onState;
    request = gatewayMocks.request;
  },
}));
vi.mock("@/components/ModelPickerDialog", () => ({
  ModelPickerDialog: () => null,
}));
vi.mock("@/components/ModelReloadConfirm", () => ({
  ModelReloadConfirm: () => null,
}));
vi.mock("@/components/ReasoningPicker", () => ({
  ReasoningPicker: () => null,
}));
vi.mock("@nous-research/ui/ui/components/button", () => ({
  Button: ({ children }: { children?: ReactNode }) => <button>{children}</button>,
}));
vi.mock("@nous-research/ui/ui/components/badge", () => ({
  Badge: ({ children }: { children?: ReactNode }) => <span>{children}</span>,
}));
vi.mock("@nous-research/ui/ui/components/card", () => ({
  Card: ({ children }: { children?: ReactNode }) => <div>{children}</div>,
}));

type EventLike = { code?: number; data?: string };

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];

  private listeners = new Map<string, Array<(event: EventLike) => void>>();
  readonly url: string;
  closed = false;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, listener: (event: EventLike) => void) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  close() {
    this.closed = true;
  }

  emit(type: string, event: EventLike) {
    for (const listener of this.listeners.get(type) ?? []) {
      listener(event);
    }
  }
}

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
  vi.clearAllMocks();
  reloadMocks.maybeReloadForLoopbackWsAuthFailure.mockReturnValue(true);
  vi.stubGlobal("WebSocket", FakeWebSocket);
});

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
  vi.unstubAllGlobals();
});

describe("ChatSidebar event socket", () => {
  it("routes loopback 4401 closes through stale-token recovery", async () => {
    const { ChatSidebar } = await import("./ChatSidebar");

    await render(<ChatSidebar channel="chat-1" />);

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    expect(apiMocks.buildWsUrl).toHaveBeenCalledWith("/api/events", {
      channel: "chat-1",
    });

    FakeWebSocket.instances[0].emit("close", { code: 4401 });

    expect(
      reloadMocks.maybeReloadForLoopbackWsAuthFailure,
    ).toHaveBeenCalledWith(4401);
  });
});

describe("ChatSidebar event socket reconnect", () => {
  beforeEach(() => {
    // Not loopback: exercise the gated-mode path so closes fall through to
    // the reconnect logic instead of triggering a page reload.
    reloadMocks.maybeReloadForLoopbackWsAuthFailure.mockReturnValue(false);
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  async function renderSidebar() {
    const { ChatSidebar } = await import("./ChatSidebar");
    await render(<ChatSidebar channel="chat-1" />);
    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
  }

  /** Advance timers and flush the async `connect()` that fires on the tick. */
  async function advance(ms: number) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(ms);
    });
  }

  it("reconnects after a transient close", async () => {
    await renderSidebar();

    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });
    expect(FakeWebSocket.instances).toHaveLength(1);

    await advance(1_000);
    expect(FakeWebSocket.instances).toHaveLength(2);
    expect(apiMocks.buildWsUrl).toHaveBeenCalledTimes(2);
  });

  it("backs off exponentially across repeated failures", async () => {
    await renderSidebar();

    // 1s, then 2s, then 4s — a socket that never opens keeps backing off.
    for (const [index, delay] of [1_000, 2_000, 4_000].entries()) {
      await act(async () => {
        FakeWebSocket.instances[index].emit("close", { code: 1006 });
      });

      // The previous (shorter) delay must not be enough to fire this one.
      if (index > 0) {
        await advance(delay - 1);
        expect(FakeWebSocket.instances).toHaveLength(index + 1);
      }

      await advance(delay);
      expect(FakeWebSocket.instances).toHaveLength(index + 2);
    }
  });

  it("schedules only one retry when error and close both fire", async () => {
    await renderSidebar();

    // A failed socket emits `error` then `close`. Scheduling from both
    // paths would queue two timers and leak the untracked one.
    await act(async () => {
      FakeWebSocket.instances[0].emit("error", {});
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });

    await advance(30_000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("resets the backoff after a successful reconnect", async () => {
    await renderSidebar();

    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });
    await advance(1_000);
    expect(FakeWebSocket.instances).toHaveLength(2);

    // Reconnected — the next drop should start from 1s again, not 2s.
    await act(async () => {
      FakeWebSocket.instances[1].emit("open", {});
      FakeWebSocket.instances[1].emit("close", { code: 1006 });
    });
    await advance(1_000);
    expect(FakeWebSocket.instances).toHaveLength(3);
  });

  it("does not retry auth rejections", async () => {
    await renderSidebar();

    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 4403 });
    });

    await advance(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("does not retry a normal closure", async () => {
    await renderSidebar();

    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 1000 });
    });

    await advance(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });

  it("gives up after the attempt cap instead of retrying forever", async () => {
    await renderSidebar();

    for (let i = 0; i < 40; i++) {
      const socket =
        FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
      await act(async () => {
        socket.emit("close", { code: 1006 });
      });
      await advance(30_000);
    }

    // 15 retries + the initial connection.
    expect(FakeWebSocket.instances.length).toBeLessThanOrEqual(16);
  });

  it("clears its own banner on a successful reconnect", async () => {
    await renderSidebar();

    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });
    expect(container.textContent).toContain("events feed disconnected");

    await advance(1_000);
    await act(async () => {
      FakeWebSocket.instances[1].emit("open", {});
    });

    // Banner gone entirely — including the "reconnect events feed" button,
    // which only renders while `error` is set.
    expect(container.textContent).not.toContain("events feed");
  });

  it("does not clear a credential warning when the feed recovers", async () => {
    await renderSidebar();

    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });
    await advance(1_000);

    // A sidecar error lands while the events socket is still reconnecting.
    // The banner is shared, so a blind `setError(null)` on reconnect would
    // hide a real problem the user needs to see.
    await act(async () => {
      gatewayMocks.handlers.get("error")?.({
        payload: { message: "ANTHROPIC_API_KEY is not set" },
      });
    });

    await act(async () => {
      FakeWebSocket.instances[1].emit("open", {});
    });

    expect(container.textContent).toContain("ANTHROPIC_API_KEY is not set");
  });

  it("does not overwrite a sidecar error when the feed drops", async () => {
    await renderSidebar();

    // A sidecar error is already on the banner...
    await act(async () => {
      gatewayMocks.handlers.get("error")?.({
        payload: { message: "ANTHROPIC_API_KEY is not set" },
      });
    });

    // ...when the events feed drops. `error` is that message's only home,
    // so overwriting it loses the warning permanently — the feed's own
    // banner would later clear itself to null and the sidecar never
    // re-emits.
    await act(async () => {
      FakeWebSocket.instances[0].emit("error", {});
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });

    expect(container.textContent).toContain("ANTHROPIC_API_KEY is not set");
    // The disconnect message must not have replaced it. (Matching the
    // banner text specifically — "reconnect events feed" is the button
    // label, which is expected to be present whenever a banner shows.)
    expect(container.textContent).not.toContain("events feed disconnected");
  });

  it("still reconnects while a foreign banner suppresses its message", async () => {
    await renderSidebar();

    await act(async () => {
      gatewayMocks.handlers.get("error")?.({
        payload: { message: "ANTHROPIC_API_KEY is not set" },
      });
    });
    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });

    // Declining to write the banner must not disable the retry itself.
    await advance(1_000);
    expect(FakeWebSocket.instances).toHaveLength(2);
  });

  it("clears the reconnect timer and closes the socket on unmount", async () => {
    await renderSidebar();

    await act(async () => {
      FakeWebSocket.instances[0].emit("close", { code: 1006 });
    });
    expect(vi.getTimerCount()).toBeGreaterThan(0);

    await act(async () => root.unmount());

    expect(FakeWebSocket.instances[0].closed).toBe(true);
    // The pending retry timer must be cleared, not merely neutered by the
    // `unmounting` flag — a live timer keeps the effect closure alive.
    expect(vi.getTimerCount()).toBe(0);

    await advance(60_000);
    expect(FakeWebSocket.instances).toHaveLength(1);
  });
});
