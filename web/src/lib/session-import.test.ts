import { describe, expect, it } from "vitest";

import {
  importSummary,
  parseImportSessions,
  SessionImportParseError,
  type SessionImportParseErrorCode,
} from "./session-import";

function expectParseError(text: string, code: SessionImportParseErrorCode) {
  try {
    parseImportSessions(text);
    throw new Error("Expected session import parsing to fail");
  } catch (error) {
    expect(error).toBeInstanceOf(SessionImportParseError);
    expect((error as SessionImportParseError).code).toBe(code);
  }
}

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
    expectParseError("  \n", "empty");
    expectParseError('[{"id":"one"},42]', "invalid-format");
    expectParseError('{"id":', "invalid-format");
    expectParseError('{"id":"one"}\nnot-json', "invalid-format");
  });
});

describe("importSummary", () => {
  it("includes skipped and detached counts only when present", () => {
    expect(
      importSummary(
        {
          ok: true,
          imported: 2,
          skipped: 1,
          detached: 1,
          imported_ids: ["one", "two"],
          skipped_ids: ["existing"],
          errors: [],
        },
        {
          imported: "{count} imported",
          skipped: "{count} skipped",
          detached: "{count} detached from missing parents",
        },
      ),
    ).toBe("2 imported; 1 skipped; 1 detached from missing parents");
  });
});
