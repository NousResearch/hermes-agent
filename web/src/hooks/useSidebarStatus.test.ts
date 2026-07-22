// @vitest-environment jsdom
import { createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { act } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "@/lib/api";
import type { StatusResponse } from "@/lib/api";
import { useSidebarStatus, type SidebarStatus } from "./useSidebarStatus";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

vi.mock("@/lib/api", () => ({
  api: { getStatus: vi.fn() },
}));

type Deferred<T> = {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
};

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function status(version: string): StatusResponse {
  return {
    active_sessions: 1,
    config_path: "/tmp/config.yaml",
    config_version: 1,
    env_path: "/tmp/.env",
    gateway_exit_reason: null,
    gateway_health_url: null,
    gateway_pid: 123,
    gateway_platforms: {},
    gateway_running: true,
    gateway_state: "running",
    gateway_updated_at: null,
    hermes_home: "/tmp/hermes",
    latest_config_version: 1,
    release_date: "2026-07-15",
    version,
  };
}

function Harness({ onRender }: { onRender: (value: SidebarStatus, retry: () => void) => void }) {
  const value = useSidebarStatus();
  onRender(value.status, value.retry);
  return null;
}

describe("useSidebarStatus", () => {
  let container: HTMLDivElement;
  let root: Root;
  let current: SidebarStatus;
  let retry: () => void;

  beforeEach(() => {
    vi.useFakeTimers();
    vi.mocked(api.getStatus).mockReset();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
    vi.useRealTimers();
  });

  function renderHook() {
    act(() => {
      root.render(
        createElement(Harness, {
          onRender(value: SidebarStatus, nextRetry: () => void) {
            current = value;
            retry = nextRetry;
          },
        }),
      );
    });
  }

  it("becomes unreachable only after two consecutive failures", async () => {
    vi.mocked(api.getStatus)
      .mockRejectedValueOnce(new Error("first"))
      .mockRejectedValueOnce(new Error("second"));

    renderHook();
    await act(async () => Promise.resolve());
    expect(current.kind).toBe("loading");

    act(() => retry());
    await act(async () => Promise.resolve());
    expect(current).toEqual({ kind: "unreachable", lastData: null });
  });

  it("recovers after a successful retry", async () => {
    vi.mocked(api.getStatus)
      .mockRejectedValueOnce(new Error("first"))
      .mockRejectedValueOnce(new Error("second"))
      .mockResolvedValueOnce(status("2.0.0"));

    renderHook();
    await act(async () => Promise.resolve());
    act(() => retry());
    await act(async () => Promise.resolve());
    expect(current.kind).toBe("unreachable");

    act(() => retry());
    await act(async () => Promise.resolve());
    expect(current).toEqual({ kind: "live", data: status("2.0.0") });
  });

  it("does not start a retry while a request is already in flight", async () => {
    const pending = deferred<StatusResponse>();
    vi.mocked(api.getStatus).mockReturnValue(pending.promise);

    renderHook();
    act(() => retry());
    expect(api.getStatus).toHaveBeenCalledTimes(1);

    await act(async () => pending.resolve(status("3.0.0")));
    expect(current).toEqual({ kind: "live", data: status("3.0.0") });
  });

  it("ignores every outstanding request after unmount", async () => {
    const pending = deferred<StatusResponse>();
    vi.mocked(api.getStatus).mockReturnValue(pending.promise);

    renderHook();
    act(() => retry());
    act(() => root.unmount());
    await act(async () => pending.resolve(status("4.0.0")));

    expect(current).toEqual({ kind: "loading" });
  });
});
