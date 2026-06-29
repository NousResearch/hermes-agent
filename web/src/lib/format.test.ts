import { describe, it, expect } from "vitest";
import { formatTokenCount } from "./format";

describe("formatTokenCount", () => {
  it("promotes counts that round up to 1000K into the M unit", () => {
    // [999.95K, 1M): rounding the mantissa to one decimal yields 1000.0, which
    // must cross into the next unit rather than print "1000.0K".
    expect(formatTokenCount(999_999)).toBe("1M");
    expect(formatTokenCount(999_950)).toBe("1M");
  });

  it("does not emit a trailing .0 for whole M-band values", () => {
    expect(formatTokenCount(999_999_999)).toBe("1000M");
    expect(formatTokenCount(2_000_000)).toBe("2M");
  });

  it("keeps counts below the rounding band on their own unit", () => {
    expect(formatTokenCount(999_500)).toBe("999.5K");
    expect(formatTokenCount(999_949)).toBe("999.9K");
  });

  it("renders K-band values with one decimal of precision", () => {
    expect(formatTokenCount(1_500)).toBe("1.5K");
    expect(formatTokenCount(4_096)).toBe("4.1K");
  });

  it("strips trailing .0 for clean round numbers", () => {
    expect(formatTokenCount(1_000)).toBe("1K");
    expect(formatTokenCount(128_000)).toBe("128K");
    expect(formatTokenCount(1_000_000)).toBe("1M");
    expect(formatTokenCount(1_048_576)).toBe("1M");
    expect(formatTokenCount(1_500_000)).toBe("1.5M");
  });

  it("returns sub-thousand counts verbatim", () => {
    expect(formatTokenCount(0)).toBe("0");
    expect(formatTokenCount(999)).toBe("999");
  });
});
