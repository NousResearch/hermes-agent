// @vitest-environment jsdom
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

import { EditMessageDialog } from "./EditMessageDialog";

let root: Root | null = null;
let host: HTMLDivElement;

afterEach(() => {
  act(() => root?.unmount());
  root = null;
  host?.remove();
});

function renderDialog(props: Partial<React.ComponentProps<typeof EditMessageDialog>> = {}) {
  host = document.createElement("div");
  document.body.append(host);
  root = createRoot(host);
  act(() => root?.render(createElement(EditMessageDialog, {
    initialText: "original prompt",
    loading: false,
    onCancel: vi.fn(),
    onConfirm: vi.fn(),
    open: true,
    ...props,
  })));
}

describe("EditMessageDialog", () => {
  it("requires confirmation and returns the edited text", () => {
    const onConfirm = vi.fn();
    renderDialog({ onConfirm });
    const textarea = document.body.querySelector<HTMLTextAreaElement>("textarea[aria-label='Edited user message']")!;
    act(() => {
      const valueSetter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")?.set;
      valueSetter?.call(textarea, "revised prompt");
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
      textarea.dispatchEvent(new Event("change", { bubbles: true }));
      document.body.querySelector<HTMLButtonElement>("button[data-confirm]")?.click();
    });
    expect(onConfirm).toHaveBeenCalledWith("revised prompt");
  });

  it("does not render or submit when closed", () => {
    const onConfirm = vi.fn();
    renderDialog({ open: false, onConfirm });
    expect(host.querySelector("[role='dialog']")).toBeNull();
    expect(onConfirm).not.toHaveBeenCalled();
  });
});
