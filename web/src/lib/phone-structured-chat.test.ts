import { describe, expect, it } from "vitest";

import {
  ptyChatHref,
  shouldRedirectChatToStructured,
  shouldUseStructuredChatOnPhone,
  structuredChatLocationFromChat,
} from "./phone-structured-chat";

describe("phone structured chat surface", () => {
  it("keeps the PTY chat on a fine pointer without touch", () => {
    expect(shouldUseStructuredChatOnPhone({ maxTouchPoints: 0, pointerCoarse: false })).toBe(false);
  });

  it("keeps the PTY chat on a fine-pointer laptop that also has a touch digitizer", () => {
    expect(shouldUseStructuredChatOnPhone({ maxTouchPoints: 1, pointerCoarse: false })).toBe(false);
  });

  it("uses the structured composer on a coarse phone pointer", () => {
    expect(shouldUseStructuredChatOnPhone({ maxTouchPoints: 5, pointerCoarse: true })).toBe(true);
  });

  it("moves /chat resume query onto /chat/structured without dropping profile", () => {
    expect(structuredChatLocationFromChat("/chat", "?resume=abc-1&profile=worker")).toBe(
      "/chat/structured?resume=abc-1&profile=worker",
    );
  });

  it("does not bounce an explicit PTY diagnostic link back to structured chat", () => {
    expect(shouldRedirectChatToStructured(
      { maxTouchPoints: 5, pointerCoarse: true },
      "?resume=abc-1&pty=1",
    )).toBe(false);
    expect(ptyChatHref("sess-9", "worker")).toBe("/chat?resume=sess-9&profile=worker&pty=1");
  });
});
