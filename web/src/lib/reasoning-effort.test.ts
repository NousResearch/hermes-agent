import { describe, it, expect } from "vitest";
import {
  EFFORT_OPTIONS,
  VALID_EFFORTS,
  normalizeEffort,
  filterEffortOptions,
} from "./reasoning-effort";

describe("normalizeEffort", () => {
  it("treats empty/unset as the Hermes default (medium)", () => {
    expect(normalizeEffort("")).toBe("medium");
    expect(normalizeEffort(null)).toBe("medium");
    expect(normalizeEffort(undefined)).toBe("medium");
    expect(normalizeEffort("   ")).toBe("medium");
  });

  it("passes through every valid effort level", () => {
    for (const level of ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]) {
      expect(normalizeEffort(level)).toBe(level);
    }
  });

  it("is case- and whitespace-insensitive", () => {
    expect(normalizeEffort("HIGH")).toBe("high");
    expect(normalizeEffort("  XHigh  ")).toBe("xhigh");
  });

  it("falls back to medium for unknown values", () => {
    expect(normalizeEffort("turbo")).toBe("medium");
    expect(normalizeEffort(42)).toBe("medium");
  });
});

describe("EFFORT_OPTIONS", () => {
  it("every option value is in VALID_EFFORTS (no orphan labels)", () => {
    for (const opt of EFFORT_OPTIONS) {
      expect(VALID_EFFORTS.has(opt.value)).toBe(true);
    }
  });

  it("covers the real reasoning levels plus thinking-off", () => {
    const values = new Set(EFFORT_OPTIONS.map((o) => o.value));
    for (const level of ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]) {
      expect(values.has(level)).toBe(true);
    }
  });
});

describe("filterEffortOptions", () => {
  it("keeps the full list when the provider has no level declaration", () => {
    expect(filterEffortOptions(undefined).map((option) => option.value)).toEqual(
      EFFORT_OPTIONS.map((option) => option.value),
    );
  });

  it("filters to provider-declared levels while preserving an existing saved value", () => {
    expect(filterEffortOptions(["none", "high"], "ultra").map((option) => option.value)).toEqual([
      "none",
      "high",
      "ultra",
    ]);
  });

  it("returns no options when the model declares no reasoning dial", () => {
    expect(filterEffortOptions([])).toEqual([]);
  });

  it("does not resurrect a saved value for a model with no reasoning dial", () => {
    expect(filterEffortOptions([], "high")).toEqual([]);
  });
});
