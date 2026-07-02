import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useRealtimeFetch } from "./useRealtimeFetch";

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  static instances: MockWebSocket[] = [];

  readonly url: string;
  readyState = MockWebSocket.CONNECTING;
  onopen: (() => void) | null = null;
  onmessage: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onclose: (() => void) | null = null;
  close = vi.fn(() => {
    this.readyState = MockWebSocket.CLOSED;
  });

  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }

  open(): void {
    this.readyState = MockWebSocket.OPEN;
    this.onopen?.();
  }

  message(): void {
    this.onmessage?.();
  }

  closeFromServer(): void {
    this.readyState = MockWebSocket.CLOSED;
    this.onclose?.();
  }
}

const fetchFn = vi.fn<() => Promise<string>>();
const onSuccess = vi.fn<(value: string) => void>();
const onError = vi.fn<(message: string, err: unknown) => void>();
const setLoading = vi.fn<(loading: boolean) => void>();
const onStatusChange = vi.fn();
const onNextRetryMsChange = vi.fn();

function renderRealtime(options: Partial<Parameters<typeof useRealtimeFetch<string>>[0]> = {}) {
  return renderHook(() =>
    useRealtimeFetch<string>({
      fetchFn,
      onSuccess,
      onError,
      setLoading,
      onStatusChange,
      onNextRetryMsChange,
      pollingIntervalMs: 1_000,
      retryIntervalMs: 500,
      maxRetryMs: 2_000,
      ...options,
    }),
  );
}

async function flushEffects(): Promise<void> {
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.clearAllMocks();
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket);
  window.__HERMES_SESSION_TOKEN__ = "test-token";
  window.__HERMES_AUTH_REQUIRED__ = false;
  fetchFn.mockResolvedValue("snapshot");
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useRealtimeFetch", () => {
  it("falls back to interval polling when no WebSocket path is supplied", async () => {
    renderRealtime({ websocketPath: undefined });

    await flushEffects();
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(onSuccess).toHaveBeenCalledWith("snapshot");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });

    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(MockWebSocket.instances).toHaveLength(0);
  });

  it("opens an authenticated WebSocket and refetches when a message arrives", async () => {
    const { unmount } = renderRealtime({ websocketPath: "/api/plugins/kanban/events?since=7" });

    await flushEffects();
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(MockWebSocket.instances).toHaveLength(1);

    const socket = MockWebSocket.instances[0];
    expect(socket.url).toContain("/api/plugins/kanban/events?since=7");
    expect(socket.url).toContain("token=test-token");

    act(() => socket.open());
    expect(onStatusChange).toHaveBeenLastCalledWith("connected");

    await act(async () => {
      socket.message();
      await Promise.resolve();
    });

    expect(fetchFn).toHaveBeenCalledTimes(2);

    unmount();
    expect(socket.close).toHaveBeenCalledTimes(1);
  });

  it("passes the original error object to onError for auth-aware consumers", async () => {
    const err = new Error("denied");
    fetchFn.mockRejectedValueOnce(err);

    renderRealtime({ websocketPath: undefined });

    await flushEffects();
    expect(onError).toHaveBeenCalledWith("denied", err);
    expect(setLoading).toHaveBeenLastCalledWith(false);
  });

  it("falls back to polling and schedules reconnect when the socket closes", async () => {
    renderRealtime({ websocketPath: "/api/plugins/kanban/events?since=0" });

    await flushEffects();
    expect(MockWebSocket.instances).toHaveLength(1);
    const socket = MockWebSocket.instances[0];
    act(() => socket.open());

    act(() => socket.closeFromServer());
    expect(onStatusChange).toHaveBeenLastCalledWith("reconnecting");
    expect(onNextRetryMsChange).toHaveBeenLastCalledWith(500);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_000);
    });
    expect(fetchFn.mock.calls.length).toBeGreaterThanOrEqual(2);

    await act(async () => {
      await vi.advanceTimersByTimeAsync(500);
    });
    expect(MockWebSocket.instances.length).toBeGreaterThanOrEqual(2);
  });
});
