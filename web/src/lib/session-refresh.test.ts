import { describe, it, expect } from "vitest";
import { sessionsSignature, shouldRefreshSessions } from "./session-refresh";

describe("shouldRefreshSessions", () => {
  it("returns false on the first poll (no baseline yet)", () => {
    expect(shouldRefreshSessions(null, "s2")).toBe(false);
  });

  it("returns false when the current response has no sessions", () => {
    expect(shouldRefreshSessions("s1", null)).toBe(false);
    expect(shouldRefreshSessions(null, null)).toBe(false);
  });

  it("returns false when the ordered ids are unchanged", () => {
    expect(shouldRefreshSessions("s1", "s1")).toBe(false);
  });

  it("returns true when a new session appears at the head of the list", () => {
    expect(shouldRefreshSessions("s1", "s2")).toBe(true);
  });
});

describe("sessionsSignature", () => {
  const ids = (...list: string[]) => list.map((id) => ({ id }));

  it("is null for an empty list", () => {
    expect(sessionsSignature([])).toBeNull();
  });

  it("is identical for identical ordered ids", () => {
    expect(sessionsSignature(ids("a", "b", "c"))).toBe(sessionsSignature(ids("a", "b", "c")));
  });

  it("changes when sessions behind an unchanged head swap places", () => {
    const before = sessionsSignature(ids("a", "b", "c"));
    const after = sessionsSignature(ids("a", "c", "b"));
    expect(after).not.toBe(before);
    expect(shouldRefreshSessions(before, after)).toBe(true);
  });

  it("changes when a new session becomes the head", () => {
    expect(shouldRefreshSessions(sessionsSignature(ids("a", "b")), sessionsSignature(ids("z", "a", "b")))).toBe(true);
  });
});
