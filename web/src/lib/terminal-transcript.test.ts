import { describe, it, expect } from "vitest";
import { reassembleWrappedLines, type TranscriptRow } from "./terminal-transcript";

const rows = (...specs: Array<[string, boolean]>): TranscriptRow[] =>
  specs.map(([text, isWrapped]) => ({ text, isWrapped }));

describe("reassembleWrappedLines", () => {
  it("returns empty string for no rows", () => {
    expect(reassembleWrappedLines([])).toBe("");
  });

  it("keeps unwrapped rows on separate lines", () => {
    expect(
      reassembleWrappedLines(rows(["first", false], ["second", false])),
    ).toBe("first\nsecond");
  });

  it("joins a wrapped continuation onto the previous logical line", () => {
    // A long path soft-wrapped across two physical rows must copy as one line.
    expect(
      reassembleWrappedLines(
        rows(["/very/long/path/that/", false], ["wraps/here", true]),
      ),
    ).toBe("/very/long/path/that/wraps/here");
  });

  it("joins a run of several wrapped rows with no separators", () => {
    expect(
      reassembleWrappedLines(
        rows(["aaa", false], ["bbb", true], ["ccc", true], ["ddd", true]),
      ),
    ).toBe("aaabbbcccddd");
  });

  it("mixes wrapped and unwrapped rows correctly", () => {
    expect(
      reassembleWrappedLines(
        rows(
          ["cmd --flag ", false],
          ["value", true],
          ["$ next", false],
          ["output", false],
        ),
      ),
    ).toBe("cmd --flag value\n$ next\noutput");
  });

  it("treats a leading wrapped row as the start of the transcript", () => {
    // Defensive: nothing precedes row 0, so it can only start a logical line.
    expect(reassembleWrappedLines(rows(["orphan", true]))).toBe("orphan");
  });

  it("trims trailing whitespace and blank rows", () => {
    expect(
      reassembleWrappedLines(
        rows(["line", false], ["   ", false], ["", false]),
      ),
    ).toBe("line");
  });

  it("preserves interior blank lines", () => {
    expect(
      reassembleWrappedLines(
        rows(["a", false], ["", false], ["b", false]),
      ),
    ).toBe("a\n\nb");
  });
});
