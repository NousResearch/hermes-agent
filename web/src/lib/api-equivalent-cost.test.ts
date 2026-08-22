import { describe, expect, it } from "vitest";

import { apiEquivalentCostStat } from "./api-equivalent-cost";

describe("apiEquivalentCostStat", () => {
  it("omits the shadow-cost stat when the backend keeps the feature disabled", () => {
    expect(apiEquivalentCostStat(null, "API-equivalent cost")).toBeNull();
  });

  it("labels and formats an enabled API-equivalent shadow cost", () => {
    expect(apiEquivalentCostStat(52.547839, "API-equivalent cost", 0)).toEqual({
      label: "API-equivalent cost",
      value: "$52.55",
    });
  });

  it("marks partial coverage instead of presenting a precise total", () => {
    expect(apiEquivalentCostStat(52.547839, "API-equivalent cost", 1_000)).toEqual({
      label: "API-equivalent cost",
      value: "$52.55+",
    });
  });

  it("does not present zero dollars when all subscription usage is unpriced", () => {
    expect(apiEquivalentCostStat(0, "API-equivalent cost", 1_000)).toEqual({
      label: "API-equivalent cost",
      value: "N/A",
    });
  });
});
