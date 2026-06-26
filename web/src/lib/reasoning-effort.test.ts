import { describe, it, expect } from "vitest";
import {
  EFFORT_OPTIONS,
  VALID_EFFORTS,
  normalizeEffort,
} from "./reasoning-effort";

describe("normalizeEffort", () => {
  it("treats empty/unset as the Hermes default (medium)", () => {
    expect(normalizeEffort("")).toBe("medium");
    expect(normalizeEffort(null)).toBe("medium");
    expect(normalizeEffort(undefined)).toBe("medium");
    expect(normalizeEffort("   ")).toBe("medium");
  });

  it("passes through every official effort level plus thinking-off", () => {
    for (const level of ["none", "low", "medium", "high", "extra_high"]) {
      expect(normalizeEffort(level)).toBe(level);
    }
  });

  it("normalizes legacy aliases without surfacing them as selectable values", () => {
    expect(normalizeEffort("minimal")).toBe("low");
    expect(normalizeEffort("  XHigh  ")).toBe("extra_high");
    expect(normalizeEffort("extra high")).toBe("extra_high");
    expect(normalizeEffort("max")).toBe("extra_high");
  });

  it("is case- and whitespace-insensitive", () => {
    expect(normalizeEffort("HIGH")).toBe("high");
    expect(normalizeEffort("  Extra High  ")).toBe("extra_high");
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

  it("surfaces only thinking-off plus the official GPT-5.5 effort levels", () => {
    expect(EFFORT_OPTIONS).toEqual([
      { value: "none", label: "Off (no thinking)" },
      { value: "low", label: "Low" },
      { value: "medium", label: "Medium" },
      { value: "high", label: "High" },
      { value: "extra_high", label: "Extra High" },
    ]);
  });
});
