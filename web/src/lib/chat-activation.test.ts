import { describe, expect, it, vi } from "vitest";

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

describe("chatActivationStore", () => {
  it("activates on demand, notifies subscribers once, and is sticky", async () => {
    vi.resetModules();
    const { chatActivationStore } = await import("./chat-activation");
    expect(chatActivationStore.getSnapshot()).toBe(false);
    const seen: boolean[] = [];
    const unsub = chatActivationStore.subscribe(() =>
      seen.push(chatActivationStore.getSnapshot()),
    );
    chatActivationStore.activate();
    expect(chatActivationStore.getSnapshot()).toBe(true);
    // Sticky: a second activate() is a no-op and must not re-notify.
    chatActivationStore.activate();
    expect(seen).toEqual([true]);
    // Unsubscribe stops notifications.
    unsub();
    chatActivationStore.activate();
    expect(seen).toEqual([true]);
  });
});
