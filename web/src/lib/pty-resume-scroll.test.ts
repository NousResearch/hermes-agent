import { describe, expect, it } from "vitest";

import { shouldReconnectPtyOnPageResume } from "./pty-reconnect";

// ---------------------------------------------------------------------------
// Resume-only scroll invariant (documented here, tested at the integration
// boundary we can reach without a DOM):
//
// scrollToBottom() must fire on tab-switch resume (isActive: false→true) and
// on visibilitychange→"visible", but NOT on ordinary resize events.
//
// The ChatPage component gates these via:
//   • isActive double-rAF effect: fires only when isActive flips to true
//   • visibilitychange handler: guards on document.visibilityState === "visible"
//
// These cannot be unit-tested without jsdom; the invariants below document
// the expected conditions and guard the reconnect helper that shares the
// same resume gate.
// ---------------------------------------------------------------------------

describe("shouldReconnectPtyOnPageResume — resume gate invariants", () => {
  it("fires when the tab becomes active and visible with a dead socket", () => {
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: true,
        socketReadyState: 3, // CLOSED
        ptyState: "reconnecting",
      }),
    ).toBe(true);
  });

  it("does NOT fire when the page is hidden — same gate that prevents spurious scrolls", () => {
    // The visibilitychange scroll handler checks visibilityState === "visible"
    // before scrolling; this test confirms the shared "hidden" guard works.
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

  it("does NOT fire when the tab is not active (isActive=false)", () => {
    // The isActive double-rAF effect returns early when isActive is false —
    // so neither reconnect nor scroll fires on inactive tabs.
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: false,
        visibilityState: "visible",
        online: true,
        socketReadyState: 3,
        ptyState: "reconnecting",
      }),
    ).toBe(false);
  });

  it("does NOT fire for an already-open socket (normal resize path)", () => {
    // ResizeObserver / window resize events go through syncTerminalMetrics()
    // which does NOT call scrollToBottom() — this test confirms that an open,
    // healthy socket is left alone on resume, mirroring the scroll invariant:
    // we only scroll when the layout has actually changed (resume), not on
    // every metric sync.
    expect(
      shouldReconnectPtyOnPageResume({
        isActive: true,
        visibilityState: "visible",
        online: true,
        socketReadyState: 1, // OPEN
        ptyState: "open",
      }),
    ).toBe(false);
  });
});
