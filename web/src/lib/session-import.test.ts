import { describe, expect, it } from "vitest";

import { importSummary, parseImportSessions } from "./session-import";

describe("parseImportSessions", () => {
  it("accepts a single exported session", () => {
    expect(parseImportSessions('{"id":"session-1","messages":[]}')).toEqual([
      { id: "session-1", messages: [] },
    ]);
  });

  it("accepts arrays and wrapped session exports", () => {
    const sessions = [{ id: "one" }, { id: "two" }];
    expect(parseImportSessions(JSON.stringify(sessions))).toEqual(sessions);
    expect(parseImportSessions(JSON.stringify({ sessions }))).toEqual(sessions);
  });

  it("accepts JSONL session exports", () => {
    expect(parseImportSessions('{"id":"one"}\n\n{"id":"two"}\n')).toEqual([
      { id: "one" },
      { id: "two" },
    ]);
  });

  it("rejects empty files and non-object entries", () => {
    expect(() => parseImportSessions("  \n")).toThrow("File is empty");
    expect(() => parseImportSessions('[{"id":"one"},42]')).toThrow(
      "Expected exported session JSON or JSONL",
    );
  });
});

describe("importSummary", () => {
  it("includes skipped and detached counts only when present", () => {
    expect(
      importSummary({
        ok: true,
        imported: 2,
        skipped: 1,
        detached: 1,
        imported_ids: ["one", "two"],
        skipped_ids: ["existing"],
        errors: [],
      }),
    ).toBe("2 imported; 1 skipped; 1 detached from missing parents");
  });
});
