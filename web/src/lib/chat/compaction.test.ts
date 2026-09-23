// @vitest-environment node
import { describe, expect, it } from "vitest";

import {
  COMPACTION_END_MARKER,
  COMPACTION_PREFIXES,
  isCompactionContent,
  splitCompactionContent,
} from "@/lib/chat/compaction";

describe("splitCompactionContent", () => {
  it("returns null for non-compaction content", () => {
    expect(splitCompactionContent("Hello there.")).toBeNull();
    expect(splitCompactionContent("")).toBeNull();
    // Random text that happens to mention the word "context" must NOT match.
    expect(splitCompactionContent("Context is everything.")).toBeNull();
  });

  it("matches every known prefix", () => {
    for (const prefix of COMPACTION_PREFIXES) {
      const body = `${prefix}\nbody\n${COMPACTION_END_MARKER}\nhello`;
      const out = splitCompactionContent(body);
      expect(out).not.toBeNull();
      expect(out?.summary.startsWith(prefix)).toBe(true);
      expect(out?.remainder).toBe("hello");
    }
  });

  it("trims leading whitespace before matching the prefix (#29824)", () => {
    const body = `   \n${COMPACTION_PREFIXES[0]}\nbody\n${COMPACTION_END_MARKER}\nthe reply`;
    const out = splitCompactionContent(body);
    expect(out).not.toBeNull();
    expect(out?.remainder).toBe("the reply");
  });

  it("returns { summary, '' } when there is no end marker", () => {
    const body = `${COMPACTION_PREFIXES[0]}\njust a summary, nothing after`;
    const out = splitCompactionContent(body);
    expect(out).toEqual({ summary: body, remainder: "" });
  });

  it("strips leading whitespace from the remainder", () => {
    const body = `${COMPACTION_PREFIXES[0]}\nsummary\n${COMPACTION_END_MARKER}\n\n   \n   the reply`;
    const out = splitCompactionContent(body);
    expect(out?.remainder).toBe("the reply");
  });

  it("keeps the END marker in sync with the documented constant", () => {
    // The marker is exported so the renderer can reference the exact bytes
    // the compressor writes; guard against silent drift.
    expect(COMPACTION_END_MARKER).toContain("END OF CONTEXT SUMMARY");
  });
});

describe("isCompactionContent", () => {
  it("matches a content that has no end marker", () => {
    expect(isCompactionContent(`${COMPACTION_PREFIXES[2]} body`)).toBe(true);
  });

  it("does not match ordinary assistant content", () => {
    expect(isCompactionContent("Sure, here you go.")).toBe(false);
  });
});