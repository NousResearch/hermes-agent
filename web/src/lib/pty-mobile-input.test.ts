import { describe, expect, it } from "vitest";

import {
  normalizePtyMobileInput,
  shouldTreatInputAsMobileReplacement,
  updatePtyInputLine,
} from "./pty-mobile-input";

describe("shouldTreatInputAsMobileReplacement", () => {
  it("recognizes explicit browser replacement input", () => {
    expect(shouldTreatInputAsMobileReplacement("insertReplacementText", "Kain", false)).toBe(true);
    expect(shouldTreatInputAsMobileReplacement("insertFromComposition", "Kain", false)).toBe(true);
    expect(shouldTreatInputAsMobileReplacement("insertCompositionText", "Kain", false)).toBe(true);
  });

  it("treats multi-character mobile insertText as replacement-like", () => {
    expect(shouldTreatInputAsMobileReplacement("insertText", "Kain", true)).toBe(true);
    expect(shouldTreatInputAsMobileReplacement("insertText", "K", true)).toBe(false);
    expect(shouldTreatInputAsMobileReplacement("insertText", "Kain", false)).toBe(false);
  });
});

describe("normalizePtyMobileInput", () => {
  it("turns a Gboard full-line suggestion into a line replacement", () => {
    const result = normalizePtyMobileInput(
      "hello my name is Kain Kain",
      "hello my name is kain",
      true,
    );

    expect(result.normalized).toBe(true);
    expect(result.nextLine).toBe("hello my name is Kain");
    expect(result.data).toBe("\x7f".repeat("hello my name is kain".length) + "hello my name is Kain");
  });

  it("turns a Gboard last-word suggestion into a last-word replacement", () => {
    const result = normalizePtyMobileInput("Kain", "hello my name is kain", true);

    expect(result.normalized).toBe(true);
    expect(result.nextLine).toBe("hello my name is Kain");
    expect(result.data).toBe("\x7f".repeat("hello my name is kain".length) + "hello my name is Kain");
  });

  it("does not normalize ordinary appends when replacement is not active", () => {
    const result = normalizePtyMobileInput(
      "hello my name is Kain Kain",
      "hello my name is kain",
      false,
    );

    expect(result.normalized).toBe(false);
    expect(result.nextLine).toBe("hello my name is kainhello my name is Kain Kain");
  });

  it("does not normalize control input", () => {
    const result = normalizePtyMobileInput("\r", "hello", true);

    expect(result.normalized).toBe(false);
    expect(result.nextLine).toBe("");
    expect(result.data).toBe("\r");
  });
});

describe("updatePtyInputLine", () => {
  it("tracks printable text, delete, and submit", () => {
    expect(updatePtyInputLine("", "abc")).toBe("abc");
    expect(updatePtyInputLine("abc", "\x7f")).toBe("ab");
    expect(updatePtyInputLine("abc", "\r")).toBe("");
  });
});
