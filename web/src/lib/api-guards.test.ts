import { describe, expect, it } from "vitest";

import { asArray, asCount, asCountMap } from "./api-guards";

describe("asArray", () => {
  it("passes real arrays through", () => {
    const arr = [{ id: "1" }, { id: "2" }];
    expect(asArray(arr)).toBe(arr);
  });

  it("returns [] for null, undefined, and non-arrays", () => {
    expect(asArray(null)).toEqual([]);
    expect(asArray(undefined)).toEqual([]);
    expect(asArray("sessions")).toEqual([]);
    expect(asArray(42)).toEqual([]);
    expect(asArray({ sessions: [] })).toEqual([]);
  });
});

describe("asCount", () => {
  it("passes finite numbers through, including 0", () => {
    expect(asCount(7)).toBe(7);
    expect(asCount(0)).toBe(0);
  });

  it("falls back for non-numbers and non-finite values", () => {
    expect(asCount("12")).toBe(0);
    expect(asCount(null)).toBe(0);
    expect(asCount(undefined)).toBe(0);
    expect(asCount(Number.NaN)).toBe(0);
    expect(asCount(Number.POSITIVE_INFINITY)).toBe(0);
  });

  it("honours a custom fallback", () => {
    expect(asCount(null, -1)).toBe(-1);
  });
});

describe("asCountMap", () => {
  it("keeps numeric counts and coerces numeric strings", () => {
    expect(asCountMap({ api: 3, cron: "5" })).toEqual({ api: 3, cron: 5 });
  });

  it("drops non-numeric counts and keeps zero-valued keys", () => {
    expect(asCountMap({ api: 0, cron: null, webhook: "abc" })).toEqual({
      api: 0,
    });
  });

  it("returns {} for null, arrays, and primitives", () => {
    expect(asCountMap(null)).toEqual({});
    expect(asCountMap([1, 2])).toEqual({});
    expect(asCountMap("api")).toEqual({});
  });
});
