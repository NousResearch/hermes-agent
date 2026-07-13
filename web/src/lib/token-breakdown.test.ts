import { describe, expect, it } from "vitest";
import { tokenBarBreakdown } from "./token-breakdown";

describe("tokenBarBreakdown", () => {
  it("uses the canonical total for bar proportions and reports reasoning separately", () => {
    const breakdown = tokenBarBreakdown({
      input: 100,
      output: 50,
      cacheRead: 25,
      cacheWrite: 25,
      reasoning: 40,
    });

    expect(breakdown.total).toBe(200);
    expect(breakdown.segments.map((segment) => segment.key)).toEqual([
      "cacheRead",
      "cacheWrite",
      "input",
      "output",
    ]);
    expect(
      breakdown.segments.reduce((sum, segment) => sum + segment.value, 0),
    ).toBe(breakdown.total);
    expect(
      breakdown.segments.reduce(
        (sum, segment) => sum + (segment.value / breakdown.total) * 100,
        0,
      ),
    ).toBe(100);
    expect(breakdown.metadata).toEqual([
      { key: "reasoning", label: "Reasoning", value: 40 },
    ]);
  });
});
