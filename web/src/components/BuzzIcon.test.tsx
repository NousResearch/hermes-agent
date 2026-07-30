import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { BuzzIcon } from "./BuzzIcon";

describe("BuzzIcon", () => {
  it("renders the 24px Buzz line mark as a current-color mask", () => {
    const markup = renderToStaticMarkup(<BuzzIcon className="test-icon" />);

    expect(markup).toContain("<span");
    expect(markup).toContain("BuzzLogo24px.svg");
    expect(markup).toContain("mask-image:url(");
    expect(markup).toContain("background-color:currentColor");
    expect(markup).toContain("display:inline-block");
    expect(markup).toContain('class="test-icon"');
  });
});
