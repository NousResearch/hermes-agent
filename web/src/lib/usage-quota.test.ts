import { describe, expect, it } from "vitest";

import { shortUnavailableReason } from "./usage-quota";

describe("shortUnavailableReason", () => {
  it("collapses HTTP errors to the status code", () => {
    expect(
      shortUnavailableReason("Usage endpoint returned HTTP 403. Check OAuth authentication and usage-read permissions."),
    ).toBe("HTTP 403");
  });

  it("tags token-less entries", () => {
    expect(shortUnavailableReason("Credential entry has no usable token.")).toBe("no token");
  });

  it("falls back to the first clause, truncated", () => {
    expect(shortUnavailableReason("Usage lookup unavailable.")).toBe("Usage lookup unavailable");
    expect(shortUnavailableReason("A very long explanation that keeps going without punctuation at all")).toHaveLength(28);
    expect(shortUnavailableReason("")).toBe("unavailable");
  });
});
