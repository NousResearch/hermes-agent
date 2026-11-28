// @vitest-environment jsdom
// Regression for #152302: an unset numeric option must stay resettable to null.
import { act, type ComponentProps } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { AutoField } from "./AutoField";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Native input and label keep the test on AutoField's actual event handler.
vi.mock("@nous-research/ui/ui/components/input", () => ({
  Input: (props: ComponentProps<"input">) => <input {...props} />,
}));
vi.mock("@nous-research/ui/ui/components/label", () => ({
  Label: (props: ComponentProps<"label">) => <label {...props} />,
}));
vi.mock("@nous-research/ui/ui/components/select", () => ({
  Select: () => null,
  SelectOption: () => null,
}));
vi.mock("@nous-research/ui/ui/components/switch", () => ({
  Switch: () => null,
}));

describe("AutoField numeric config clearing", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    container = document.createElement("div");
    document.body.append(container);
    root = createRoot(container);
  });

  afterEach(() => {
    act(() => root.unmount());
    container.remove();
  });

  function renderField(nullable: boolean) {
    const onChange = vi.fn();
    act(() => {
      root.render(
        <AutoField
          schemaKey="auxiliary.compression.no_progress_timeout"
          schema={{ type: "number", nullable }}
          value={120}
          onChange={onChange}
        />,
      );
    });
    const input = container.querySelector('input[type="number"]') as HTMLInputElement;
    expect(input).not.toBeNull();
    return { input, onChange };
  }

  function type(input: HTMLInputElement, value: string) {
    // Use the native setter so React detects the DOM edit rather than skipping
    // an event on a JS-mutated tracked input.
    const setValue = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
    act(() => {
      setValue.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  it("sends null when clearing a nullable timeout and numbers when editing it", () => {
    const { input, onChange } = renderField(true);
    type(input, "");
    expect(onChange).toHaveBeenLastCalledWith(null);
    type(input, "150");
    expect(onChange).toHaveBeenLastCalledWith(150);
  });

  it("keeps the existing zero-on-clear behaviour for non-nullable numeric fields", () => {
    const { input, onChange } = renderField(false);
    type(input, "");
    expect(onChange).toHaveBeenLastCalledWith(0);
  });
});
