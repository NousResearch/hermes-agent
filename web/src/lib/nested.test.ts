import { describe, it, expect } from "vitest";
import { getNestedValue, setNestedValue } from "./nested";

describe("setNestedValue prototype-pollution hardening", () => {
  it("does not pollute Object.prototype via a __proto__ path", () => {
    expect(() => setNestedValue({}, "__proto__.polluted", true)).toThrow();
    // The global prototype must stay clean regardless of how the call is handled.
    expect(({} as Record<string, unknown>).polluted).toBeUndefined();
  });

  it("rejects a constructor.prototype path too", () => {
    expect(() => setNestedValue({}, "constructor.prototype.x", 1)).toThrow();
    expect(({} as Record<string, unknown>).x).toBeUndefined();
  });

  it("rejects a dangerous leaf segment", () => {
    expect(() => setNestedValue({}, "__proto__", 1)).toThrow();
    expect(() => setNestedValue({}, "a.constructor", 1)).toThrow();
  });
});

describe("setNestedValue / getNestedValue happy path", () => {
  it("sets a nested value without mutating the input", () => {
    const input = { a: {} as Record<string, unknown> };
    const result = setNestedValue(input, "a.b", 1);
    expect(result).toEqual({ a: { b: 1 } });
    // structuredClone semantics: original is untouched.
    expect(input).toEqual({ a: {} });
  });

  it("creates intermediate objects for a fresh path", () => {
    expect(setNestedValue({}, "a.b.c", "x")).toEqual({ a: { b: { c: "x" } } });
  });

  it("reads a nested value", () => {
    expect(getNestedValue({ a: { b: 2 } }, "a.b")).toBe(2);
  });

  it("returns undefined for a missing path", () => {
    expect(getNestedValue({ a: {} }, "a.b")).toBeUndefined();
    expect(getNestedValue({}, "x.y")).toBeUndefined();
  });
});
