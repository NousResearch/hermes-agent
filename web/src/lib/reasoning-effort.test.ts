import { describe, it, expect } from "vitest";

import enabledReasoningEfforts from "./reasoning-effort-values.json";
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

  it("passes through every valid effort level", () => {
    for (const level of ["none", ...enabledReasoningEfforts]) {
      expect(normalizeEffort(level)).toBe(level);
    }
  });

  it("is case- and whitespace-insensitive", () => {
    expect(normalizeEffort("HIGH")).toBe("high");
    expect(normalizeEffort("  XHigh  ")).toBe("xhigh");
    expect(normalizeEffort("  MAX  ")).toBe("max");
  });

  it("falls back to medium for unknown values", () => {
    expect(normalizeEffort("turbo")).toBe("medium");
    expect(normalizeEffort("ultra")).toBe("medium");
    expect(normalizeEffort(42)).toBe("medium");
  });
});

describe("EFFORT_OPTIONS", () => {
  it("every option value is in VALID_EFFORTS (no orphan labels)", () => {
    for (const opt of EFFORT_OPTIONS) {
      expect(VALID_EFFORTS.has(opt.value)).toBe(true);
    }
  });

  it("matches the shared generic contract plus thinking-off exactly", () => {
    const values = EFFORT_OPTIONS.map((o) => o.value);
    expect(values).toEqual(["none", ...enabledReasoningEfforts]);
    expect(values).not.toContain("ultra");
  });

  it("keeps xhigh and max as distinct visible labels", () => {
    const labels = Object.fromEntries(EFFORT_OPTIONS.map((o) => [o.value, o.label]));
    expect(labels.xhigh).toBe("Extra High");
    expect(labels.max).toBe("Max");
  });
});
