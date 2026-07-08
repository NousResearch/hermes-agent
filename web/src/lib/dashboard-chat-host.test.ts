import { describe, expect, it } from "vitest";

import { shouldRenderPersistentChatHost } from "./dashboard-chat-host";

describe("shouldRenderPersistentChatHost", () => {
  it("renders immediately on the first chat route paint", () => {
    expect(shouldRenderPersistentChatHost(false, true)).toBe(true);
  });

  it("keeps the persistent host mounted after the first chat visit", () => {
    expect(shouldRenderPersistentChatHost(true, false)).toBe(true);
  });

  it("does not mount chat before it is visited", () => {
    expect(shouldRenderPersistentChatHost(false, false)).toBe(false);
  });
});
