import { describe, expect, it } from "vitest";
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
