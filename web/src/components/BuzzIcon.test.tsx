import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { BuzzIcon } from "./BuzzIcon";

describe("BuzzIcon", () => {
  it("preserves the exact approved Buzz asset bytes", () => {
    const asset = readFileSync(new URL("../assets/BuzzLogo24px.svg", import.meta.url));
    expect(createHash("sha256").update(asset).digest("hex")).toBe(
      "6efb8bf616e0febd3940f411927d42cccddfd798112c9fc53e3f7b9ae46f4ce0",
    );
  });

  it("renders the approved 24px Buzz mark as a current-color mask", () => {
    const markup = renderToStaticMarkup(<BuzzIcon className="test-icon" />);

    expect(markup).toContain("<span");
    expect(markup).toContain("BuzzLogo24px.svg");
    expect(markup).toContain("mask-image:url(");
    expect(markup).toContain("background-color:currentColor");
    expect(markup).toContain("display:inline-block");
    expect(markup).toContain('class="test-icon"');
  });
});
