import { describe, expect, it } from "vitest";

import {
  normalizePtyMobileInput,
  resolveMobileSoftDelete,
  shouldTreatInputAsMobileReplacement,
  updatePtyInputLine,
} from "./pty-mobile-input";

describe("resolveMobileSoftDelete", () => {
  it("maps soft-keyboard backspace to DEL", () => {
    // Gboard fires deleteContentBackward on backspace instead of a keydown.
    expect(resolveMobileSoftDelete("deleteContentBackward")).toBe("\x7f");
    expect(resolveMobileSoftDelete("deleteHardTextBackward")).toBe("\x7f");
    expect(resolveMobileSoftDelete("deleteSoftTextBackward")).toBe("\x7f");
  });

  it("maps word deletes to the readline/prompt_toolkit bindings", () => {
    // Mirrors the desktop Ctrl+Backspace / Ctrl+Delete shortcuts in ChatPage.
    expect(resolveMobileSoftDelete("deleteWordBackward")).toBe("\x17");
    expect(resolveMobileSoftDelete("deleteWordForward")).toBe("\x1bd");
    expect(resolveMobileSoftDelete("deleteContentForward")).toBe("\x1b[3~");
  });

  it("ignores insert and unknown input types", () => {
    expect(resolveMobileSoftDelete("insertText")).toBe(null);
    expect(resolveMobileSoftDelete("insertCompositionText")).toBe(null);
    expect(resolveMobileSoftDelete(undefined)).toBe(null);
  });

  it("the tracked line shrinks when a soft-delete byte passes through", () => {
    // Contract between the two layers: whatever resolveMobileSoftDelete
    // emits must be modelled by updatePtyInputLine so a later mobile
    // replacement is rewritten against the post-delete line.
    const bytes = resolveMobileSoftDelete("deleteContentBackward");
    expect(bytes).not.toBe(null);
    expect(updatePtyInputLine("abc", bytes!)).toBe("ab");
  });
});

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

  it("does not collapse legitimate single-letter reduplication", () => {
    // "a a" is a plausible thing to type; the >=2-char guard keeps the
    // duplicate-final-word collapse from eating it inside the window.
    const result = normalizePtyMobileInput("a a", "a", true);

    expect(result.normalized).toBe(false);
    expect(result.data).toBe("a a");
  });
});

describe("normalizePtyMobileInput with a stale tracked line", () => {
  it("emits deletes sized to the stale tracker unless the tracker was reset", () => {
    // #106403: the tracked line survives a dashboard tab switch while the
    // TUI composer moves on. A re-emitted word inside the replacement
    // window is then rewritten against the stale snapshot — the DELETE
    // count is the tracker's length, not the composer's, so the composer
    // ends up with repeated characters ("hel" + 3×DEL + "hello" over a
    // composer holding "hello" leaves "hehello"). Clearing the tracker on
    // deactivation makes the same input pass through untouched.
    const stale = normalizePtyMobileInput("hello", "hel", true);
    expect(stale.normalized).toBe(true);
    expect(stale.data).toBe("\x7f".repeat("hel".length) + "hello");

    const reset = normalizePtyMobileInput("hello", "", true);
    expect(reset.normalized).toBe(false);
    expect(reset.data).toBe("hello");
    expect(reset.nextLine).toBe("hello");
  });
});

describe("updatePtyInputLine", () => {
  it("tracks printable text, delete, and submit", () => {
    expect(updatePtyInputLine("", "abc")).toBe("abc");
    expect(updatePtyInputLine("abc", "\x7f")).toBe("ab");
    expect(updatePtyInputLine("abc", "\r")).toBe("");
  });

  it("resets tracking on escape sequences instead of appending their payload", () => {
    // Left-arrow arrives as one CSI chunk; the tracker cannot model cursor
    // moves, so it must disarm rather than record "hello[D".
    expect(updatePtyInputLine("hello", "\x1b[D")).toBe("");
    expect(updatePtyInputLine("hello", "\x1b[H")).toBe("");
    expect(updatePtyInputLine("hello", "\x1bOP")).toBe("");
  });
});

describe("normalizePtyMobileInput after cursor movement", () => {
  it("does not emit a replacement against a tracker reset by arrow keys", () => {
    // Simulate: type "hello my name is kain", press left-arrow, then a
    // Gboard suggestion arrives. The tracker reset means no replacement
    // heuristic can fire against a stale line snapshot.
    const afterArrow = updatePtyInputLine("hello my name is kain", "\x1b[D");
    const result = normalizePtyMobileInput("Kain", afterArrow, true);

    expect(result.normalized).toBe(false);
    expect(result.data).toBe("Kain");
  });
});
