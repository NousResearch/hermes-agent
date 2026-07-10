import { describe, it, expect } from "vitest";
import {
  EFFORT_OPTIONS,
  MODE_OPTIONS,
  VALID_EFFORTS,
  VALID_MODES,
  isCodexProvider,
  normalizeEffort,
  normalizeMode,
} from "./reasoning-effort";

describe("normalizeEffort", () => {
  it("treats empty/unset as the Hermes default (medium)", () => {
    expect(normalizeEffort("")).toBe("medium");
    expect(normalizeEffort(null)).toBe("medium");
    expect(normalizeEffort(undefined)).toBe("medium");
    expect(normalizeEffort("   ")).toBe("medium");
  });

  it("passes through every valid effort level", () => {
    for (const level of ["none", "minimal", "low", "medium", "high", "xhigh"]) {
      expect(normalizeEffort(level)).toBe(level);
    }
  });

  it("is case- and whitespace-insensitive", () => {
    expect(normalizeEffort("HIGH")).toBe("high");
    expect(normalizeEffort("  XHigh  ")).toBe("xhigh");
  });

  it("falls back to medium for unknown values", () => {
    expect(normalizeEffort("turbo")).toBe("medium");
    expect(normalizeEffort("max")).toBe("medium"); // 'max' is a label, not a value
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
    for (const level of ["none", "minimal", "low", "medium", "high", "xhigh"]) {
      expect(values.has(level)).toBe(true);
    }
  });
});

describe("normalizeMode", () => {
  it("accepts standard and pro case-insensitively", () => {
    expect(normalizeMode("standard")).toBe("standard");
    expect(normalizeMode(" PRO ")).toBe("pro");
  });

  it("falls back to standard for empty or unknown values", () => {
    expect(normalizeMode("")).toBe("standard");
    expect(normalizeMode(null)).toBe("standard");
    expect(normalizeMode("turbo")).toBe("standard");
  });
});

describe("MODE_OPTIONS", () => {
  it("contains exactly the supported Codex modes", () => {
    expect(MODE_OPTIONS.map((option) => option.value)).toEqual([
      "standard",
      "pro",
    ]);
    for (const option of MODE_OPTIONS) {
      expect(VALID_MODES.has(option.value)).toBe(true);
    }
  });
});

describe("isCodexProvider", () => {
  it("matches only the openai-codex provider", () => {
    expect(isCodexProvider("openai-codex")).toBe(true);
    expect(isCodexProvider(" OpenAI-Codex ")).toBe(true);
    expect(isCodexProvider("openai")).toBe(false);
    expect(isCodexProvider("")).toBe(false);
  });
});
