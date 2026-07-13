import { describe, expect, it } from "vitest";

import { normalizeLearnDeepLinkSeed } from "./learn-deeplink";

describe("normalizeLearnDeepLinkSeed", () => {
  it("keeps ordinary skill descriptions intact", () => {
    expect(normalizeLearnDeepLinkSeed("TypeScript refactor workflow")).toBe(
      "TypeScript refactor workflow",
    );
  });

  it("collapses control characters so a deep link cannot submit extra PTY input", () => {
    expect(normalizeLearnDeepLinkSeed("notes\r/exit\n\u001b[31mnext\u0007")).toBe(
      "notes /exit [31mnext",
    );
  });

  it("returns null for empty or control-only values", () => {
    expect(normalizeLearnDeepLinkSeed(null)).toBeNull();
    expect(normalizeLearnDeepLinkSeed(" \r\n\t\u001b ")).toBeNull();
  });
});
