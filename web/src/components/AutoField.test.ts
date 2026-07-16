import { describe, expect, it } from "vitest";

import { formatListValue, parseListValue } from "./AutoField";

describe("context.external_files list editor", () => {
  it("round-trips comma-containing paths one per line", () => {
    const paths = ["/tmp/rules,team.md", "/tmp/other.md"];

    expect(formatListValue(paths, "lines")).toBe(
      "/tmp/rules,team.md\n/tmp/other.md",
    );
    expect(parseListValue(formatListValue(paths, "lines"), "lines")).toEqual(paths);
  });

  it("preserves a trailing line while editing and removes it on blur", () => {
    expect(parseListValue("/tmp/first.md\n", "lines", true)).toEqual([
      "/tmp/first.md",
      "",
    ]);
    expect(parseListValue("/tmp/first.md\n", "lines")).toEqual([
      "/tmp/first.md",
    ]);
  });

  it("keeps the generic list editor comma-separated", () => {
    expect(formatListValue(["web", "terminal"])).toBe("web, terminal");
    expect(parseListValue("web, terminal")).toEqual(["web", "terminal"]);
  });
});
