import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { Stats } from "./Stats";

describe("Stats", () => {
  it("hides decorative dot leaders from accessibility APIs", () => {
    const markup = renderToStaticMarkup(
      <Stats items={[{ label: "Sessions", value: "3" }]} />,
    );

    expect(markup).toContain('aria-hidden="true"');
    expect(markup).toContain("border-dotted");
    expect(markup).not.toContain("·");
    expect(markup).toContain("Sessions");
    expect(markup).toContain(">3<");
  });
});
