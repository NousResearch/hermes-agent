import { describe, expect, it } from "vitest";
import {
  getSessionMessageRefreshMode,
  hasSessionActivityChanged,
  reconcileSessionActivity,
  shouldRefreshSessions,
  type SessionActivitySnapshot,
} from "./session-refresh";

const activity = (
  overrides: Partial<SessionActivitySnapshot> = {},
): SessionActivitySnapshot => ({
  id: "s1",
  is_active: true,
  ended_at: null,
  last_active: 10,
  message_count: 2,
  ...overrides,
});

describe("shouldRefreshSessions", () => {
  it("returns false on the first poll (no baseline yet)", () => {
    expect(shouldRefreshSessions(null, "s2")).toBe(false);
  });

  it("returns false when the current response has no sessions", () => {
    expect(shouldRefreshSessions("s1", null)).toBe(false);
    expect(shouldRefreshSessions(null, null)).toBe(false);
  });

  it("returns false when the newest session id is unchanged", () => {
    expect(shouldRefreshSessions("s1", "s1")).toBe(false);
  });

  it("returns true when a new session appears at the head of the list", () => {
    expect(shouldRefreshSessions("s1", "s2")).toBe(true);
  });
});

describe("getSessionMessageRefreshMode", () => {
  it("polls messages while an expanded session is active", () => {
    expect(
      getSessionMessageRefreshMode({
        isExpanded: true,
        hasMessages: true,
        wasActive: true,
        isActive: true,
      }),
    ).toBe("poll");
  });

  it("fetches once when an active session becomes inactive", () => {
    expect(
      getSessionMessageRefreshMode({
        isExpanded: true,
        hasMessages: true,
        wasActive: true,
        isActive: false,
      }),
    ).toBe("once");
  });

  it("fetches an initially expanded inactive session once", () => {
    expect(
      getSessionMessageRefreshMode({
        isExpanded: true,
        hasMessages: false,
        wasActive: false,
        isActive: false,
      }),
    ).toBe("once");
  });

  it("does nothing for a collapsed row", () => {
    expect(
      getSessionMessageRefreshMode({
        isExpanded: false,
        hasMessages: false,
        wasActive: true,
        isActive: true,
      }),
    ).toBe("none");
  });
});

describe("session activity reconciliation", () => {
  it("detects same-ID completion", () => {
    const previous = [activity()];
    const current = [
      activity({ is_active: false, ended_at: 20, last_active: 20 }),
    ];
    expect(hasSessionActivityChanged(previous, current)).toBe(true);
  });

  it("ignores unchanged same-ID activity", () => {
    const previous = [activity()];
    expect(hasSessionActivityChanged(previous, [activity()])).toBe(false);
  });

  it("updates a searched active row without replacing result membership", () => {
    const searched = [
      { ...activity(), snippet: ">>>match<<<", title: "Keep me" },
      {
        ...activity({ id: "s2", is_active: false }),
        snippet: "other",
        title: "Also keep me",
      },
    ];
    const current = [
      activity({
        is_active: false,
        ended_at: 20,
        last_active: 20,
        message_count: 3,
      }),
    ];

    const reconciled = reconcileSessionActivity(searched, current);

    expect(reconciled.map((session) => session.id)).toEqual(["s1", "s2"]);
    expect(reconciled[0]).toMatchObject({
      is_active: false,
      message_count: 3,
      snippet: ">>>match<<<",
      title: "Keep me",
    });
    expect(reconciled[1]).toBe(searched[1]);
  });
});
