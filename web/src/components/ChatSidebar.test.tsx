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

const gatewayMocks = vi.hoisted(() => ({
  close: vi.fn(),
  connect: vi.fn(async () => undefined),
  on: vi.fn(() => () => undefined),
  onState: vi.fn((handler: (state: string) => void) => {
    handler("open");
    return () => undefined;
  }),
  request: vi.fn(async () => ({ session_id: "sidecar-1" })),
}));

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

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  addEventListener(type: string, listener: (event: EventLike) => void) {
    const listeners = this.listeners.get(type) ?? [];
    listeners.push(listener);
    this.listeners.set(type, listeners);
  }

  close() {}

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
