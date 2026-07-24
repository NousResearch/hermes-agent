/** @vitest-environment jsdom */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ModelPickerDialog } from "./ModelPickerDialog";

async function flushEffects() {
  await act(async () => {
    await Promise.resolve();
    await new Promise((resolve) => setTimeout(resolve, 0));
  });
}

describe("ModelPickerDialog mobile structure", () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    vi.restoreAllMocks();
  });

  it("stacks provider and model navigation on phones and preserves selection", async () => {
    const onApply = vi.fn().mockResolvedValue(undefined);
    const onClose = vi.fn();
    const loader = vi.fn().mockResolvedValue({
      model: "model/current",
      provider: "provider-one",
      providers: [
        {
          is_current: true,
          models: ["model/current", "model/a-very-long-mobile-model-identifier"],
          name: "Provider One With A Long Mobile Name",
          slug: "provider-one",
        },
      ],
    });

    act(() => {
      root.render(
        <ModelPickerDialog
          alwaysGlobal
          loader={loader}
          onApply={onApply}
          onClose={onClose}
        />,
      );
    });
    await flushEffects();

    const dialog = document.querySelector<HTMLElement>("[role='dialog']");
    const grid = document.querySelector<HTMLElement>(
      "[data-testid='model-picker-responsive-grid']",
    );
    const search = document.querySelector<HTMLInputElement>(
      "input[placeholder='Filter providers and models…']",
    );

    expect(dialog?.className).toContain("hermes-modal-backdrop");
    expect(dialog?.firstElementChild?.className).toContain("hermes-modal-panel-viewport");
    expect(dialog?.firstElementChild?.className).toContain("lg:h-auto");
    expect(grid?.className).toContain("grid-rows-[minmax(7rem,0.75fr)_minmax(8rem,1.25fr)]");
    expect(grid?.className).toContain("sm:grid-cols-[200px_minmax(0,1fr)]");
    expect(document.querySelector("[aria-label='Model providers']")).not.toBeNull();
    expect(document.querySelector("[aria-label='Models']")).not.toBeNull();
    expect(search?.getAttribute("autocapitalize")).toBe("none");
    expect(search?.getAttribute("spellcheck")).toBe("false");
    expect(search?.className).toContain("lg:min-h-8");

    const modelControl = Array.from(document.querySelectorAll<HTMLElement>("button, [role='button']")).find(
      (element) => element.textContent?.includes("model/a-very-long-mobile-model-identifier"),
    );
    act(() => modelControl?.click());
    const switchButton = Array.from(document.querySelectorAll<HTMLButtonElement>("button")).find(
      (button) => button.textContent?.trim() === "Switch",
    );
    await act(async () => {
      switchButton?.click();
      await Promise.resolve();
    });

    expect(onApply).toHaveBeenCalledWith({
      confirmExpensiveModel: false,
      model: "model/a-very-long-mobile-model-identifier",
      persistGlobal: true,
      provider: "provider-one",
    });
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
