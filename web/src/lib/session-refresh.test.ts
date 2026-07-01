import { describe, it, expect } from "vitest";
import { shouldRefreshSessions } from "./session-refresh";

describe("shouldRefreshSessions", () => {
  const session = {
    id: "s1",
    title: "Active chat",
    last_active: 1_782_902_400,
    is_active: true,
    message_count: 0,
    tool_call_count: 0,
    preview: "",
  };

  it("returns false on the first poll (no baseline yet)", () => {
    expect(shouldRefreshSessions(null, [{ ...session, id: "s2" }])).toBe(
      false,
    );
  });

  it("returns false when the current response has no sessions", () => {
    expect(shouldRefreshSessions([session], [])).toBe(false);
    expect(shouldRefreshSessions(null, [])).toBe(false);
  });

  it("returns false when the newest session id is unchanged", () => {
    expect(shouldRefreshSessions([session], [{ ...session }])).toBe(false);
  });

  it("returns true when a new session appears at the head of the list", () => {
    expect(
      shouldRefreshSessions([session], [{ ...session, id: "s2" }]),
    ).toBe(true);
  });

  it("returns true when sessions appear after an empty baseline", () => {
    expect(shouldRefreshSessions([], [session])).toBe(true);
  });

  it("returns true when an active session changes without a new newest id", () => {
    expect(
      shouldRefreshSessions(
        [session],
        [
          {
            ...session,
            message_count: 2,
            preview: "Hello there",
            last_active: 1_782_902_460,
          },
        ],
      ),
    ).toBe(true);
  });
});
