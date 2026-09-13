import { describe, expect, it } from "vitest";

import {
  shouldUseStructuredChatOnPhone,
  structuredChatLocationFromChat,
} from "./phone-structured-chat";

describe("phone structured chat surface", () => {
  it("keeps the PTY chat on a fine pointer without touch", () => {
    expect(shouldUseStructuredChatOnPhone({ maxTouchPoints: 0, pointerCoarse: false })).toBe(false);
  });

  it("uses the structured composer on a coarse phone pointer", () => {
    expect(shouldUseStructuredChatOnPhone({ maxTouchPoints: 5, pointerCoarse: true })).toBe(true);
  });

  it("moves /chat resume query onto /chat/structured without dropping profile", () => {
    expect(structuredChatLocationFromChat("/chat", "?resume=abc-1&profile=worker")).toBe(
      "/chat/structured?resume=abc-1&profile=worker",
    );
  });
});
