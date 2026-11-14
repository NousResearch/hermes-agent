import { describe, expect, it } from "vitest";

import { generateChannelId } from "./chat-channel-id";

describe("generateChannelId", () => {
  it("prefixes chat when scoped and chat-fresh when not", () => {
    expect(generateChannelId("scope")).toMatch(/^chat-/);
    expect(generateChannelId()).toMatch(/^chat-fresh-/);
  });

  it("never returns the same id twice", () => {
    expect(generateChannelId("s")).not.toBe(generateChannelId("s"));
  });
});
