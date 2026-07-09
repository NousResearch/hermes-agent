import { describe, expect, it } from "vitest";

import {
  shouldBlockPtyInput,
  shouldReconnectPtyOnPageResume,
} from "./pty-reconnect";

describe("shouldReconnectPtyOnPageResume", () => {
  it("reconnects a missing socket when the active page becomes visible", () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: true,
        socketReadyState: null,
        ptyState: "reconnecting",
      }),
    ).toBe(true);
  });

  it("reconnects closed or closing sockets on visible resume", () => {
    for (const socketReadyState of [2, 3]) {
      expect(
        shouldReconnectPtyOnPageResume({
          isActive: true,
          visibilityState: "visible",
          online: true,
          socketReadyState,
          ptyState: "reconnecting",
        }),
      ).toBe(true);
    }
  });

  it("does not reconnect an open socket on visible resume", () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: true,
        socketReadyState: 1,
        ptyState: "open",
      }),
    ).toBe(false);
  });

  it("reconnects a still-connecting socket when the page is already in reconnecting state", () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: true,
        socketReadyState: 0,
        ptyState: "reconnecting",
      }),
    ).toBe(true);
  });

  it("does not reconnect while the page is hidden", () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "hidden",
        online: true,
        socketReadyState: 3,
        ptyState: "reconnecting",
      }),
    ).toBe(false);
  });

  it("defers reconnect while offline", () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: false,
        socketReadyState: 3,
        ptyState: "reconnecting",
      }),
    ).toBe(false);
  });

  it("does not fire a redundant reconnect while a connect is in flight (wsRef not yet assigned)", () => {
    // The async socket-open IIFE has begun but not yet assigned wsRef, so
    // socketReadyState reads null. Without the connectInFlight guard this
    // would return true and double-connect.
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: true,
        socketReadyState: null,
        ptyState: "connecting",
        connectInFlight: true,
      }),
    ).toBe(false);
  });

  it("still reconnects an in-flight connect when the page already believes it is closed", () => {
    // A stuck attempt the user is actively trying to recover (manual reconnect
    // or a closed state) must not be suppressed by the in-flight guard.
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: true,
        socketReadyState: null,
        ptyState: "closed",
        connectInFlight: true,
      }),
    ).toBe(true);
  });
});

describe("shouldBlockPtyInput", () => {
  it("allows input only while the PTY socket is open", () => {
    expect(shouldBlockPtyInput("open")).toBe(false);
    expect(shouldBlockPtyInput("connecting")).toBe(true);
    expect(shouldBlockPtyInput("reconnecting")).toBe(true);
    expect(shouldBlockPtyInput("closed")).toBe(true);
    expect(shouldBlockPtyInput("ended")).toBe(true);
  });
});
