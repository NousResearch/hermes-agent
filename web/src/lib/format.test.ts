import { describe, expect, it } from "vitest";

import { formatTokenCount, formatTokens } from "./format";

describe("formatTokenCount", () => {
  it("strips trailing .0 for clean round numbers (default mode)", () => {
    expect(formatTokenCount(1_000)).toBe("1K");
    expect(formatTokenCount(128_000)).toBe("128K");
    expect(formatTokenCount(1_000_000)).toBe("1M");
    expect(formatTokenCount(2_000_000)).toBe("2M");
    expect(formatTokenCount(999)).toBe("999");
  });

  it("shows one decimal for non-round values", () => {
    expect(formatTokenCount(1_500)).toBe("1.5K");
    expect(formatTokenCount(1_500_000)).toBe("1.5M");
  });

  it("promotes at the toFixed rounding boundary to M (regression)", () => {
    // [999_950, 999_999] used to render as the impossible "1000.0K".
    expect(formatTokenCount(999_950)).toBe("1.0M");
    expect(formatTokenCount(999_999)).toBe("1.0M");
    expect(formatTokenCount(999_949)).toBe("999.9K");
  });
});

describe("formatTokens", () => {
  it("forces a single decimal for the K/M bands", () => {
    expect(formatTokens(1_000)).toBe("1.0K");
    expect(formatTokens(1_500_000)).toBe("1.5M");
    expect(formatTokens(1_000_000)).toBe("1.0M");
  });

  it("promotes the toFixed rounding boundary to M (regression)", () => {
    expect(formatTokens(999_950)).toBe("1.0M"); // was "1000.0K"
    expect(formatTokens(999_999)).toBe("1.0M"); // was "1000.0K"
  });

  it("keeps values just below the boundary in K", () => {
    expect(formatTokens(999_949)).toBe("999.9K");
    expect(formatTokens(999_499)).toBe("999.5K");
  });

  it("leaves sub-thousand values unformatted", () => {
    expect(formatTokens(999)).toBe("999");
  });
});
