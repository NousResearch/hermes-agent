// @vitest-environment jsdom
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { GatewayClient } from "./gatewayClient";

const reloadMocks = vi.hoisted(() => ({
  maybeReloadForLoopbackWsAuthFailure: vi.fn(() => false),
}));

vi.mock("./dashboard-auth-reload", () => ({
  maybeReloadForLoopbackWsAuthFailure:
    reloadMocks.maybeReloadForLoopbackWsAuthFailure,
}));

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  static OPEN = 1;

  listeners = new Map<string, Array<(event: EventLike) => void>>();
  readyState = 0;
  url: string;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, cb: (event: EventLike) => void) {
    const list = this.listeners.get(type) ?? [];
    list.push(cb);
    this.listeners.set(type, list);
  }

  close() {}

  emit(type: string, event: EventLike) {
    for (const cb of this.listeners.get(type) ?? []) {
      cb(event);
    }
  }

  removeEventListener(type: string, cb: (event: EventLike) => void) {
    const list = this.listeners.get(type) ?? [];
    this.listeners.set(
      type,
      list.filter((item) => item !== cb),
    );
  }

  send() {}
}

type EventLike = {
  code?: number;
};

beforeEach(() => {
  FakeWebSocket.instances = [];
  reloadMocks.maybeReloadForLoopbackWsAuthFailure.mockClear();
  vi.stubGlobal("WebSocket", FakeWebSocket);
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
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("GatewayClient", () => {
  it("treats loopback 4401 closes as stale-token reload candidates", async () => {
    reloadMocks.maybeReloadForLoopbackWsAuthFailure.mockReturnValue(true);
    const gw = new GatewayClient();
    const connectPromise = gw.connect();

    await vi.waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    const socket = FakeWebSocket.instances[0];
    socket.readyState = 1;
    socket.emit("open", {});
    await connectPromise;

    socket.emit("close", { code: 4401 });

    expect(
      reloadMocks.maybeReloadForLoopbackWsAuthFailure,
    ).toHaveBeenCalledWith(4401);
    expect(gw.connectionState).toBe("open");
  });
});
