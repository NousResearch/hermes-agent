import { describe, expect, it } from "vitest";
import { applyPtyFilters, PtyResumeSanitizer } from "./pty-resume-sanitizer";

describe("applyPtyFilters", () => {
  it("passes through normal text unchanged", () => {
    const input = "hello world\nfoo bar\n";
    expect(applyPtyFilters(input)).toBe(input);
  });

  it("collapses pathological blank-line bursts (≥50 newlines)", () => {
    const burst = "a\n" + "\n".repeat(100) + "b\n";
    const result = applyPtyFilters(burst);
    // Burst collapsed to "\n\n", surrounding text preserved
    expect(result).toBe("a\n\nb\n");
    // Verify no CSI masking
    expect(result).not.toContain("\x1b");
  });

  it("leaves short blank-line runs untouched", () => {
    const short = "a\n\n\nb\n"; // 3 newlines
    expect(applyPtyFilters(short)).toBe(short);
  });

  it("strips erase-line CSI sequences", () => {
    expect(applyPtyFilters("hello\x1b[K world")).toBe("hello world");
    expect(applyPtyFilters("foo\x1b[2Kbar")).toBe("foobar");
    expect(applyPtyFilters("\x1b[K")).toBe("");
  });

  it("strips erase-char CSI sequences", () => {
    expect(applyPtyFilters("abc\x1b[3Xdef")).toBe("abcdef");
    expect(applyPtyFilters("\x1b[X")).toBe("");
  });

  it("leaves normal SGR sequences untouched", () => {
    // \x1b[31m (red), \x1b[0m (reset) — must not be removed
    const input = "\x1b[31mred text\x1b[0m\n";
    expect(applyPtyFilters(input)).toBe(input);
  });

  it("handles empty string", () => {
    expect(applyPtyFilters("")).toBe("");
  });

  it("handles mixed CSI and newlines correctly", () => {
    const input = "line1\x1b[K\n" + "\n".repeat(60) + "line2\x1b[2X\n";
    const result = applyPtyFilters(input);
    expect(result).toBe("line1\n\nline2\n");
    expect(result).not.toContain("\x1b");
  });
});

describe("PtyResumeSanitizer — stateful frame handling", () => {
  it("processes a complete chunk in one frame", () => {
    const s = new PtyResumeSanitizer();
    expect(s.next("hello\n")).toBe("hello\n");
    expect(s.flush()).toBe("");
  });

  it("buffers a trailing partial escape and resolves in next frame", () => {
    const s = new PtyResumeSanitizer();
    // Frame 1 ends mid-CSI
    const out1 = s.next("hello\x1b[");
    expect(out1).toBe("hello"); // partial buffered, not emitted

    // Frame 2 completes the CSI
    const out2 = s.next("2K world\n");
    expect(out2).toBe(" world\n"); // erase-line stripped
    expect(s.flush()).toBe("");
  });

  it("buffers bare \\x1b and resolves on completion", () => {
    const s = new PtyResumeSanitizer();
    expect(s.next("before\x1b")).toBe("before");
    expect(s.next("[Kafter")).toBe("after");
  });

  it("buffers \\x1b[\\d+ prefix and resolves", () => {
    const s = new PtyResumeSanitizer();
    expect(s.next("x\x1b[4")).toBe("x");
    expect(s.next("2Ky")).toBe("y");
  });

  it("passes through when no partial escape is buffered", () => {
    const s = new PtyResumeSanitizer();
    expect(s.next("chunk1\n")).toBe("chunk1\n");
    expect(s.next("chunk2\n")).toBe("chunk2\n");
  });

  it("collapses a pathological newline burst within a single frame", () => {
    const s = new PtyResumeSanitizer();
    // 60 newlines in one frame triggers the 50+ threshold
    const out = s.next("start\n" + "\n".repeat(60) + "end\n");
    expect(out).toBe("start\n\nend\n");
  });

  it("drains buffered partial on flush", () => {
    const s = new PtyResumeSanitizer();
    s.next("trailing\x1b[");
    // Connection closes — flush drains the incomplete partial.
    // Bare \x1b[ is not a complete CSI (missing terminal letter), so it passes through.
    expect(s.flush()).toBe("\x1b[");
  });

  it("resets state across instances", () => {
    const a = new PtyResumeSanitizer();
    a.next("\x1b[");
    const b = new PtyResumeSanitizer();
    // [K without \x1b prefix is literal text, not a CSI code
    expect(b.next("[Khello")).toBe("[Khello");
    expect(a.flush()).toBe("\x1b[");
  });

  it("handles empty chunk without disturbing pending buffer", () => {
    const s = new PtyResumeSanitizer();
    s.next("before\x1b[");
    expect(s.next("")).toBe("");  // empty → no output, pending preserved
    expect(s.next("2Kafter")).toBe("after");
  });

  it("handles consecutive split escapes across three frames", () => {
    const s = new PtyResumeSanitizer();
    expect(s.next("a\x1b")).toBe("a");
    expect(s.next("[")).toBe("");
    expect(s.next("2Kb")).toBe("b");
  });
});
