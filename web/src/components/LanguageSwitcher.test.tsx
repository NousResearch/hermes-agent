// @vitest-environment jsdom

import { afterEach, describe, expect, it } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

import { I18nProvider } from "@/i18n";
import { LanguageSwitcher } from "./LanguageSwitcher";

let container: HTMLDivElement;
let root: Root;

Object.defineProperty(window, "matchMedia", {
  configurable: true,
  value: () => ({ matches: false, addEventListener: () => {}, removeEventListener: () => {} }),
});
globalThis.IS_REACT_ACT_ENVIRONMENT = true;

afterEach(async () => {
  await act(async () => root?.unmount());
  container?.remove();
});

describe("LanguageSwitcher", () => {
  it("renders a graphical mobile trigger alongside its breakpoint-hidden label", async () => {
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);

    await act(async () => {
      root.render(
        <I18nProvider>
          <LanguageSwitcher />
        </I18nProvider>,
      );
    });

    const trigger = container.querySelector("button[aria-label]");
    expect(trigger?.querySelector("svg")).not.toBeNull();
    expect(trigger?.textContent).toContain("EN");
  });
});
