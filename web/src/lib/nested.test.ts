import { describe, expect, it } from "vitest";
import { getNestedValue, setNestedValue } from "./nested";

describe("nested configuration paths", () => {
  it.each([
    "terminal.backends.coder.__proto__.polluted",
    "terminal.backends.coder.constructor.prototype.polluted",
    "terminal.backends.coder..polluted",
  ])("rejects unsafe writes through %s", (path) => {
    expect(() => setNestedValue({}, path, true)).toThrow(
      "Unsafe nested configuration path",
    );
  });

  it("does not traverse inherited properties when reading", () => {
    const inherited = { inherited: "secret" };
    const value = Object.create(inherited) as Record<string, unknown>;

    expect(getNestedValue(value, "inherited")).toBeUndefined();
  });
});
