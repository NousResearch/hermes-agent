import { describe, expect, it } from "vitest";

import {
  ANALYTICS_ALL,
  ANALYTICS_PERIODS,
  analyticsAvgDivisor,
  isAllRange,
} from "./analytics-range";

describe("analytics range", () => {
  it("keeps the three day presets in order", () => {
    expect(ANALYTICS_PERIODS.map((p) => [p.label, p.days])).toEqual([
      ["7d", 7],
      ["30d", 30],
      ["90d", 90],
    ]);
  });

  it("identifies the all-time range", () => {
    expect(isAllRange(ANALYTICS_ALL)).toBe(true);
    expect(isAllRange("all")).toBe(true);
    expect(isAllRange(30)).toBe(false);
    expect(isAllRange(7)).toBe(false);
  });

  it("divides a numeric range by its own day count", () => {
    expect(analyticsAvgDivisor(30, 12)).toBe(30);
    expect(analyticsAvgDivisor(7, 0)).toBe(7);
  });

  it("divides an all-time range by the active day count", () => {
    expect(analyticsAvgDivisor("all", 45)).toBe(45);
    // Fractional day counts are floored, not rounded.
    expect(analyticsAvgDivisor("all", 10.9)).toBe(10);
  });

  it("never divides by zero for an empty all-time dataset", () => {
    expect(analyticsAvgDivisor("all", 0)).toBe(1);
    expect(analyticsAvgDivisor("all", Number.NaN)).toBe(1);
    expect(analyticsAvgDivisor("all", Number.POSITIVE_INFINITY)).toBe(1);
  });
});
