import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { AutoField } from "./AutoField";

describe("AutoField", () => {
  it("gives generated boolean switches an accessible name", () => {
    const markup = renderToStaticMarkup(
      <AutoField
        schemaKey="buzz.extra.require_mention"
        schema={{ type: "boolean" }}
        value={true}
        onChange={() => undefined}
      />,
    );

    expect(markup).toContain('role="switch"');
    expect(markup).toContain('aria-label="Require Mention"');
  });
});