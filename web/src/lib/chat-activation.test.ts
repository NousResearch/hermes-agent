import { describe, expect, it } from "vitest";

import { latchChatActivation } from "./chat-activation";

describe("latchChatActivation", () => {
  it("stays inactive while the chat tab has never been active", () => {
    // A dashboard sitting on /sessions, /system, … must not flip the latch,
    // so the persistently-mounted ChatPage never opens /api/pty (which would
    // trigger the TUI/agent bootstrap on every page).
    expect(latchChatActivation(false, false)).toBe(false);
  });

  it("activates when the chat tab becomes active", () => {
    expect(latchChatActivation(false, true)).toBe(true);
  });

  it("stays activated after the chat tab is left (sticky / persistence)", () => {
    // Once the user has opened /chat, the PTY must survive navigating away so
    // a running agent turn is not torn down on every tab switch.
    expect(latchChatActivation(true, false)).toBe(true);
  });

  it("stays activated while the chat tab remains active", () => {
    expect(latchChatActivation(true, true)).toBe(true);
  });
});
