import { describe, it, expect } from "vitest";

import {
  nextLiveSessionId,
  shouldDropSidebarEvent,
} from "./sidebar-event-routing";

describe("nextLiveSessionId", () => {
  it("learns the live id from a session.info frame", () => {
    expect(nextLiveSessionId("session.info", "sess-A", null)).toBe("sess-A");
  });

  it("updates the live id when session.info changes (e.g. /new)", () => {
    expect(nextLiveSessionId("session.info", "sess-B", "sess-A")).toBe("sess-B");
  });

  it("ignores session.info frames with no session_id", () => {
    expect(nextLiveSessionId("session.info", undefined, "sess-A")).toBe(
      "sess-A",
    );
  });

  it("leaves the live id unchanged for non-session.info frames", () => {
    expect(nextLiveSessionId("tool.start", "sess-B", "sess-A")).toBe("sess-A");
  });
});

describe("shouldDropSidebarEvent", () => {
  it("drops a scoped frame from a different session (the leak)", () => {
    // Background session sess-B emits tool.start while sess-A is live.
    expect(shouldDropSidebarEvent("tool.start", "sess-B", "sess-A")).toBe(true);
  });

  it("keeps a scoped frame from the live session", () => {
    expect(shouldDropSidebarEvent("tool.start", "sess-A", "sess-A")).toBe(
      false,
    );
  });

  it("keeps an unscoped frame (no session_id) — never swallow the live feed", () => {
    // #42359: dropping unscoped events swallowed the live answer.
    expect(shouldDropSidebarEvent("tool.start", undefined, "sess-A")).toBe(
      false,
    );
    expect(shouldDropSidebarEvent("tool.complete", "", "sess-A")).toBe(false);
  });

  it("keeps every frame before the live id is learned", () => {
    expect(shouldDropSidebarEvent("tool.start", "sess-B", null)).toBe(false);
  });

  it("never gates the unscoped control event", () => {
    expect(
      shouldDropSidebarEvent(
        "dashboard.new_session_requested",
        undefined,
        "sess-A",
      ),
    ).toBe(false);
    // even if some future broadcaster stamps a stray id on it, it stays through
    expect(
      shouldDropSidebarEvent(
        "dashboard.new_session_requested",
        "sess-B",
        "sess-A",
      ),
    ).toBe(false);
  });

  it("keeps the session.info frame that establishes a new live id", () => {
    // The component learns the id first (nextLiveSessionId), then this guard
    // runs against the updated id — so the establishing frame is never dropped.
    const live = nextLiveSessionId("session.info", "sess-A", null);
    expect(shouldDropSidebarEvent("session.info", "sess-A", live)).toBe(false);
  });
});
