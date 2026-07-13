import { describe, expect, it } from "vitest";

import { formatTokens } from "./token-format";

describe("formatTokens", () => {
  it("promotes the toFixed rounding boundary to M (regression)", () => {
    expect(formatTokens(999_950)).toBe("1.0M"); // was "1000.0K"
    expect(formatTokens(999_999)).toBe("1.0M"); // was "1000.0K"
  });

  it("keeps values just below the boundary in K", () => {
    expect(formatTokens(999_949)).toBe("999.9K");
    expect(formatTokens(999_499)).toBe("999.5K");
  });

  it("formats the normal K/M/raw bands", () => {
    expect(formatTokens(999)).toBe("999");
    expect(formatTokens(1_000)).toBe("1.0K");
    expect(formatTokens(1_500_000)).toBe("1.5M");
    expect(formatTokens(1_000_000)).toBe("1.0M");
  });
});
