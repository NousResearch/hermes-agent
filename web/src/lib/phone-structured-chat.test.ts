import { describe, expect, it } from "vitest";

import {
  chatLocationFromStructured,
  ptyChatHref,
  shouldRedirectChatToStructured,
  shouldRedirectStructuredToChat,
  shouldUseStructuredChatOnPhone,
} from "./phone-structured-chat";

describe("phone chat surface", () => {
  it("keeps /chat on a coarse phone pointer instead of structured chat", () => {
    expect(shouldUseStructuredChatOnPhone({ maxTouchPoints: 5, pointerCoarse: true })).toBe(false);
    expect(shouldRedirectChatToStructured(
      { maxTouchPoints: 5, pointerCoarse: true },
      "?resume=abc-1",
    )).toBe(false);
  });

  it("keeps the PTY chat on a fine-pointer laptop that also has a touch digitizer", () => {
    expect(shouldUseStructuredChatOnPhone({ maxTouchPoints: 1, pointerCoarse: false })).toBe(false);
  });

  it("moves /chat/structured onto /chat without dropping resume or profile", () => {
    expect(shouldRedirectStructuredToChat("/chat/structured")).toBe(true);
    expect(chatLocationFromStructured("/chat/structured", "?resume=abc-1&profile=worker")).toBe(
      "/chat?resume=abc-1&profile=worker",
    );
  });

  it("leaves an explicit PTY diagnostic link on /chat", () => {
    expect(ptyChatHref("sess-9", "worker")).toBe("/chat?resume=sess-9&profile=worker&pty=1");
  });
});
