import { describe, expect, it } from "vitest";

import { applySlashCompletion, parseSlash } from "./slashExec";

// ---------------------------------------------------------------------------
// parseSlash
// ---------------------------------------------------------------------------

describe("parseSlash", () => {
  it("parses a bare slash command", () => {
    expect(parseSlash("/help")).toEqual({ name: "help", arg: "" });
  });

  it("strips the leading slash", () => {
    expect(parseSlash("/model opus-4.6")).toEqual({
      name: "model",
      arg: "opus-4.6",
    });
  });

  it("handles multiple leading slashes (defensive)", () => {
    expect(parseSlash("//compact")).toEqual({ name: "compact", arg: "" });
  });

  it("returns empty name for an empty command", () => {
    expect(parseSlash("/")).toEqual({ name: "", arg: "" });
    expect(parseSlash("")).toEqual({ name: "", arg: "" });
  });

  it("trims trailing whitespace from arg", () => {
    expect(parseSlash("/resume  abc  ")).toEqual({ name: "resume", arg: "abc" });
  });
});

// ---------------------------------------------------------------------------
// applySlashCompletion — the double-slash prevention logic
// ---------------------------------------------------------------------------

describe("applySlashCompletion", () => {
  it("does NOT produce a double-slash when the server returns a slash-prefixed item", () => {
    // Server extra commands like /compact return text="/compact" with replaceFrom=1.
    // The user has typed "/" so input="/", replaceFrom=1.
    // Without stripping: "/" + "/compact" = "//compact"  ← bug
    // With stripping:    "/" + "compact"  = "/compact"   ← correct
    const result = applySlashCompletion({
      input: "/",
      itemText: "/compact",
      replaceFrom: 1,
    });
    expect(result).toBe("/compact");
    expect(result.startsWith("//")).toBe(false);
  });

  it("does NOT produce a double-slash when the user has typed a partial command", () => {
    const result = applySlashCompletion({
      input: "/com",
      itemText: "/compact",
      replaceFrom: 1,
    });
    expect(result).toBe("/compact");
    expect(result.startsWith("//")).toBe(false);
  });

  it("keeps the item text as-is for bare (non-slash-prefixed) completion items", () => {
    // Plain commands ("help", "exit") return bare names without a leading slash.
    const result = applySlashCompletion({
      input: "/hel",
      itemText: "help",
      replaceFrom: 1,
    });
    expect(result).toBe("/help");
  });

  it("does not strip when replaceFrom is 0 (edge case)", () => {
    // When replaceFrom=0 the stripping guard is inactive.
    const result = applySlashCompletion({
      input: "/compact",
      itemText: "/compact",
      replaceFrom: 0,
    });
    // input.slice(0,0)="" + "/compact" = "/compact"
    expect(result).toBe("/compact");
  });

  it("handles input that does not start with slash (fallback)", () => {
    // Defensive: if somehow input doesn't start with "/" no stripping occurs.
    const result = applySlashCompletion({
      input: "hel",
      itemText: "help",
      replaceFrom: 0,
    });
    expect(result).toBe("help");
  });
});
