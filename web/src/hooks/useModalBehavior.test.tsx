/** @vitest-environment jsdom */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useModalBehavior } from "./useModalBehavior";

function OverlayHarness({
  modalLayout = false,
  onClose,
  open,
}: {
  modalLayout?: boolean;
  onClose: () => void;
  open: boolean;
}) {
  const overlayRef = useModalBehavior({ open, onClose, modalLayout });

  return (
    <>
      <button id="opener" type="button">
        Open
      </button>
      <section id="background">
        <button id="background-control" type="button">
          Background
        </button>
      </section>
      {open ? (
        <div
          ref={overlayRef}
          aria-labelledby="overlay-title"
          aria-modal="true"
          role="dialog"
        >
          <div>
            <h2 id="overlay-title">Overlay</h2>
            <button id="first" data-modal-initial-focus type="button">
              First
            </button>
            <button id="last" type="button">
              Last
            </button>
          </div>
        </div>
      ) : null}
    </>
  );
}

describe("useModalBehavior", () => {
  let host: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
  });

  afterEach(() => {
    act(() => root.unmount());
    host.remove();
    document.body.style.overflow = "";
  });

  function render(open: boolean, onClose = vi.fn(), modalLayout = false) {
    act(() => {
      root.render(
        <OverlayHarness
          modalLayout={modalLayout}
          onClose={onClose}
          open={open}
        />,
      );
    });
    return onClose;
  }

  it("moves focus inside, traps Tab, and redirects escaped focus", () => {
    render(false);
    document.querySelector<HTMLButtonElement>("#opener")?.focus();

    render(true);
    const first = document.querySelector<HTMLButtonElement>("#first");
    const last = document.querySelector<HTMLButtonElement>("#last");
    const background =
      document.querySelector<HTMLButtonElement>("#background-control");

    expect(document.activeElement).toBe(first);

    last?.focus();
    document.dispatchEvent(
      new KeyboardEvent("keydown", { bubbles: true, key: "Tab" }),
    );
    expect(document.activeElement).toBe(first);

    first?.focus();
    document.dispatchEvent(
      new KeyboardEvent("keydown", {
        bubbles: true,
        key: "Tab",
        shiftKey: true,
      }),
    );
    expect(document.activeElement).toBe(last);

    background?.focus();
    expect(document.activeElement).toBe(first);
  });

  it("dismisses with Escape and restores focus and background state", () => {
    const close = vi.fn();
    render(false, close);
    const opener = document.querySelector<HTMLButtonElement>("#opener");
    opener?.focus();

    render(true, close);
    const background = document.querySelector<HTMLElement>("#background");
    expect(background?.hasAttribute("inert")).toBe(true);
    expect(background?.getAttribute("aria-hidden")).toBe("true");
    expect(document.body.style.overflow).toBe("hidden");

    document.dispatchEvent(
      new KeyboardEvent("keydown", { bubbles: true, key: "Escape" }),
    );
    expect(close).toHaveBeenCalledTimes(1);

    render(false, close);
    expect(document.activeElement).toBe(opener);
    expect(background?.hasAttribute("inert")).toBe(false);
    expect(background?.hasAttribute("aria-hidden")).toBe(false);
    expect(document.body.style.overflow).toBe("");
  });

  it("adds the mobile-safe modal shell hooks only for modal layouts", () => {
    render(true, vi.fn(), true);
    const overlay = document.querySelector<HTMLElement>("[role='dialog']");
    const panel = overlay?.firstElementChild;

    expect(overlay?.classList.contains("hermes-modal-backdrop")).toBe(true);
    expect(panel?.classList.contains("hermes-modal-panel-viewport")).toBe(
      true,
    );
  });
});
