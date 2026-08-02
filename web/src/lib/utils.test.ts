import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { themedBody, themedChrome, themedFont } from "./utils";

describe("dashboard typography helper classes", () => {
  it("keeps all reusable themed text helpers on the display-font token path", () => {
    for (const className of [themedFont, themedBody, themedChrome]) {
      expect(className).toContain("font-mondwest");
    }
    expect(themedBody).toContain("normal-case");
    expect(themedChrome).toContain("text-display");
  });
});

describe("dashboard display tracking", () => {
  it("uses the theme letter-spacing token emitted by ThemeProvider", () => {
    const stylesheet = readFileSync(new URL("../index.css", import.meta.url), "utf8");

    expect(stylesheet).toContain(
      ".text-display {\n  letter-spacing: var(--theme-letter-spacing);\n}",
    );
    expect(stylesheet).not.toContain("--theme-display-letter-spacing");
  });
});
