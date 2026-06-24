import { describe, it, expect } from "vitest";

import { parseSlash } from "./slashExec";

describe("parseSlash", () => {
  it("parses a bare command with no argument", () => {
    expect(parseSlash("/help")).toEqual({ name: "help", arg: "" });
  });

  it("parses a single-line argument", () => {
    expect(parseSlash("/goal ship the feature")).toEqual({
      name: "goal",
      arg: "ship the feature",
    });
  });

  it("keeps newlines in a multi-line argument (regression for #41323)", () => {
    // Without the dotAll flag the `.*` capture stopped at the first newline,
    // so a pasted multi-line `/goal` produced an empty slash command.
    expect(parseSlash("/goal line one\nline two")).toEqual({
      name: "goal",
      arg: "line one\nline two",
    });
  });

  it("strips leading slashes before the command name", () => {
    // Guards the `replace(/^\/+/, "")` step: a broken pattern here silently
    // fails to strip the leading slash, breaking every web slash command.
    expect(parseSlash("//goal do it")).toEqual({ name: "goal", arg: "do it" });
  });

  it("returns an empty command when there is no command token", () => {
    expect(parseSlash("   ")).toEqual({ name: "", arg: "" });
  });
});
