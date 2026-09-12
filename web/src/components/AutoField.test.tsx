// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AutoField } from "./AutoField";

describe("AutoField", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  it("masks secret schema fields", async () => {
    await act(async () => {
      root.render(
        <AutoField
          onChange={vi.fn()}
          schema={{ type: "secret" }}
          schemaKey="terminal.backends.acme.token"
          value="sensitive-token"
        />,
      );
    });

    expect(container.querySelector('input[type="password"]')?.getAttribute("value")).toBe(
      "sensitive-token",
    );
  });
});
