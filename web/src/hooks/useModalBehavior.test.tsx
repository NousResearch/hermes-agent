// @vitest-environment jsdom
// The modal's document keydown listener closes on a real Escape and leaves the Escape that
// cancels an IME composition to the field, so a half-typed CJK word doesn't close the dialog.

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";

import { useModalBehavior } from "./useModalBehavior";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT =
  true;

let container: HTMLDivElement;
let root: Root;

function Modal({ onClose }: { onClose: () => void }) {
  const ref = useModalBehavior({ open: true, onClose });
  return <div ref={ref} />;
}

function pressEscape(init: KeyboardEventInit = {}): KeyboardEvent {
  const event = new KeyboardEvent("keydown", {
    key: "Escape",
    bubbles: true,
    cancelable: true,
    ...init,
  });
  act(() => {
    document.dispatchEvent(event);
  });
  return event;
}

beforeEach(() => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
});

afterEach(() => {
  act(() => root.unmount());
  container.remove();
});

describe("useModalBehavior Escape", () => {
  it("closes on an ordinary Escape", () => {
    const onClose = vi.fn();
    act(() => root.render(<Modal onClose={onClose} />));

    const event = pressEscape({ keyCode: 27 });

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(event.defaultPrevented).toBe(true);
  });

  it("stays open when Escape cancels an IME composition", () => {
    const onClose = vi.fn();
    act(() => root.render(<Modal onClose={onClose} />));

    const composing = pressEscape({ isComposing: true });
    const processKey = pressEscape({ keyCode: 229 });

    expect(onClose).not.toHaveBeenCalled();
    expect(composing.defaultPrevented).toBe(false);
    expect(processKey.defaultPrevented).toBe(false);
  });
});
