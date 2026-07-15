import { describe, it, expect } from "vitest";
import {
  EFFORT_OPTIONS,
  VALID_EFFORTS,
  normalizeEffort,
  optionsForSupportedEfforts,
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
    // Invariant against hermes_constants.VALID_REASONING_EFFORTS + 'none'.
    const values = new Set(EFFORT_OPTIONS.map((o) => o.value));
    for (const level of ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]) {
      expect(values.has(level)).toBe(true);
    }
  });
});

describe("optionsForSupportedEfforts", () => {
  it("keeps the generic menu when a model declares no restrictions", () => {
    expect(optionsForSupportedEfforts()).toEqual(EFFORT_OPTIONS);
  });

  it("shows only Inkling's declared levels and maps max to xhigh", () => {
    expect(
      optionsForSupportedEfforts(["none", "low", "medium", "high", "max"])
        .map((option) => option.value),
    ).toEqual(["none", "low", "medium", "high", "xhigh"]);
  });

  it("keeps the generic menu when every declared level is unknown", () => {
    expect(optionsForSupportedEfforts(["turbo"])).toEqual(EFFORT_OPTIONS);
  });
});
