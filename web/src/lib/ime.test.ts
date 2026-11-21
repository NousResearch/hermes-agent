import { describe, expect, it } from "vitest";

import { isImeComposing, isSubmitEnter } from "./ime";

describe("isImeComposing", () => {
  it("reads isComposing from a plain DOM event and from a React event's nativeEvent", () => {
    expect(isImeComposing({ key: "Enter", isComposing: true })).toBe(true);
    expect(isImeComposing({ key: "Escape", nativeEvent: { isComposing: true } })).toBe(true);
  });

  it("treats keyCode 229 as composing, for the commit Enter that follows compositionend", () => {
    expect(isImeComposing({ key: "Enter", keyCode: 229 })).toBe(true);
    expect(isImeComposing({ key: "Enter", nativeEvent: { keyCode: 229 } })).toBe(true);
  });

  it("is false for ordinary keys", () => {
    expect(isImeComposing({ key: "Enter", isComposing: false, keyCode: 13 })).toBe(false);
    expect(isImeComposing({ key: "Escape", nativeEvent: { isComposing: false, keyCode: 27 } })).toBe(false);
  });
});

describe("isSubmitEnter", () => {
  it("accepts a plain Enter and rejects the Enter that commits a composition", () => {
    expect(isSubmitEnter({ key: "Enter", keyCode: 13 })).toBe(true);
    expect(isSubmitEnter({ key: "Enter", nativeEvent: { isComposing: true } })).toBe(false);
    expect(isSubmitEnter({ key: "Enter", keyCode: 229 })).toBe(false);
  });

  it("ignores other keys", () => {
    expect(isSubmitEnter({ key: "Escape" })).toBe(false);
  });
});
