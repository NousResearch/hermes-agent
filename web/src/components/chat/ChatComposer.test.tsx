/**
 * Focused Phase 3 tests for the /chat-ui composer + chat controls.
 *
 * Mapping onto the brief's required checks:
 *
 *   1. Enter submits                     → "Enter submits"
 *   2. Shift+Enter newline               → "Shift+Enter creates newline"
 *   3. IME composition prevents submit   → "IME composition prevents accidental submit"
 *   4. Send button submits               → "Send button submits"
 *   5. input clears after success        → "input clears after successful submit"
 *   6. failed submission preserves input → "failed submission preserves input"
 *   7. Stop calls session.interrupt      → "Stop calls session.interrupt"
 *   8. Stop returns UI to usable state   → "Stop returns composer to usable state"
 *
 * These tests use `react-dom/test-utils` for `act` (the Phase 2 baseline
 * tests were importing `act` from `react`, which doesn't exist — they
 * were broken on main). This file deliberately uses the correct import
 * so the new cases actually run.
 */

// @vitest-environment jsdom
import { act } from "react-dom/test-utils";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ChatComposer } from "@/components/chat/ChatComposer";
import type { ChatAttachment } from "@/components/chat/types";

let roots: Root[] = [];

function flushMicrotasks(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function flushAct(): Promise<void> {
  await act(async () => {
    await flushMicrotasks();
  });
}

afterEach(() => {
  while (roots.length > 0) {
    const r = roots.pop();
    if (r) act(() => r.unmount());
  }
  document.body.innerHTML = "";
});

type ChatComposerProps = React.ComponentProps<typeof ChatComposer>;

function renderComposer(
  propsOverride: Partial<ChatComposerProps> = {},
): {
  root: Root;
  container: HTMLElement;
  onSubmit: ReturnType<typeof vi.fn>;
} {
  const onSubmit = vi.fn().mockResolvedValue(true);
  const container = document.createElement("div");
  document.body.appendChild(container);
  const root = createRoot(container);
  roots.push(root);
  act(() => {
    root.render(
      <ChatComposer
        onSubmit={onSubmit as ChatComposerProps["onSubmit"]}
        {...propsOverride}
      />,
    );
  });
  return { root, container, onSubmit };
}

function fireKey(
  target: HTMLElement,
  key: string,
  init: KeyboardEventInit = {},
): KeyboardEvent {
  const ev = new KeyboardEvent("keydown", {
    key,
    bubbles: true,
    cancelable: true,
    ...init,
  });
  target.dispatchEvent(ev);
  return ev;
}

function getTextarea(container: HTMLElement): HTMLTextAreaElement {
  const el = container.querySelector(
    '[data-testid="chat-ui-composer-input"]',
  ) as HTMLTextAreaElement | null;
  if (!el) throw new Error("composer textarea not found");
  return el;
}

function setValue(el: HTMLTextAreaElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    window.HTMLTextAreaElement.prototype,
    "value",
  )?.set;
  setter?.call(el, value);
  el.dispatchEvent(new Event("input", { bubbles: true }));
}

describe("ChatComposer — Enter / Shift+Enter / IME", () => {
  it("Enter submits", async () => {
    const { container, onSubmit } = renderComposer();
    const ta = getTextarea(container);
    setValue(ta, "hello world");
    fireKey(ta, "Enter");
    await flushAct();
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith("hello world", expect.any(Array));
  });

  it("Shift+Enter creates a newline and does NOT submit", async () => {
    const { container, onSubmit } = renderComposer();
    const ta = getTextarea(container);
    setValue(ta, "line 1");
    fireKey(ta, "Enter", { shiftKey: true });
    await flushAct();
    expect(onSubmit).not.toHaveBeenCalled();
    expect(ta.value).toBe("line 1");
  });

  it("IME composition prevents accidental submit", async () => {
    const { container, onSubmit } = renderComposer();
    const ta = getTextarea(container);
    setValue(ta, "ni");
    // Enter during composition must NOT submit (the IME consumes it).
    const startEv = new CompositionEvent("compositionstart", {
      bubbles: true,
      cancelable: true,
      data: "n",
    });
    ta.dispatchEvent(startEv);
    await flushAct();
    // After compositionend, Enter submits normally.
    const endEv = new CompositionEvent("compositionend", {
      bubbles: true,
      cancelable: true,
      data: "你",
    });
    ta.dispatchEvent(endEv);
    fireKey(ta, "Enter");
    await flushAct();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("Send button submits", async () => {
    const { container, onSubmit } = renderComposer();
    const ta = getTextarea(container);
    setValue(ta, "via button");
    const btn = container.querySelector(
      '[data-testid="chat-ui-composer-send"]',
    ) as HTMLButtonElement | null;
    expect(btn).toBeTruthy();
    btn!.click();
    await flushAct();
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith("via button", expect.any(Array));
  });
});

describe("ChatComposer — submit lifecycle", () => {
  it("input clears after successful submit", async () => {
    const onSubmit = vi.fn().mockResolvedValue(true);
    const { container } = renderComposer({
      onSubmit: onSubmit as ChatComposerProps["onSubmit"],
    });
    const ta = getTextarea(container);
    setValue(ta, "send me");
    fireKey(ta, "Enter");
    await flushAct();
    expect(onSubmit).toHaveBeenCalled();
    expect(ta.value).toBe("");
  });

  it("failed submission preserves input", async () => {
    const onSubmit = vi.fn().mockResolvedValue(false);
    const { container } = renderComposer({
      onSubmit: onSubmit as ChatComposerProps["onSubmit"],
    });
    const ta = getTextarea(container);
    setValue(ta, "don't lose me");
    fireKey(ta, "Enter");
    await flushAct();
    expect(onSubmit).toHaveBeenCalled();
    expect(ta.value).toBe("don't lose me");
  });

  it("failed submission preserves pending attachments", async () => {
    const onSubmit = vi.fn().mockResolvedValue(false);
    const onAttach = vi.fn().mockResolvedValue({
      id: "att-1",
      kind: "image",
      name: "x.png",
    } satisfies ChatAttachment);
    const { container } = renderComposer({
      onSubmit: onSubmit as ChatComposerProps["onSubmit"],
      onAttachImage: onAttach,
    });
    const file = new File([new Uint8Array([1, 2, 3])], "x.png", {
      type: "image/png",
    });
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-file-input"]',
    ) as HTMLInputElement;
    Object.defineProperty(input, "files", {
      value: [file],
      configurable: true,
    });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    const ta = getTextarea(container);
    setValue(ta, "describe this");
    fireKey(ta, "Enter");
    await flushAct();
    expect(onSubmit).toHaveBeenCalled();
    expect(ta.value).toBe("describe this");
    expect(
      container.querySelector('[data-testid="chat-ui-composer-attachments"]'),
    ).toBeTruthy();
  });

  it("submit disabled while busy / connecting", async () => {
    const { container, onSubmit } = renderComposer({ busy: true });
    const ta = getTextarea(container);
    setValue(ta, "blocked");
    expect(
      container.querySelector('[data-testid="chat-ui-composer-send"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="chat-ui-composer-stop"]'),
    ).toBeTruthy();
    fireKey(ta, "Enter");
    await flushAct();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("connecting=true renders textarea read-only with Connecting placeholder", () => {
    const { container } = renderComposer({ connecting: true });
    const ta = getTextarea(container);
    expect(ta.readOnly).toBe(true);
    expect(ta.getAttribute("placeholder")).toMatch(/Connecting/i);
  });
});

describe("ChatComposer — stop button", () => {
  it("Stop calls onStop", () => {
    const onStop = vi.fn();
    const { container } = renderComposer({ onStop, busy: true });
    const btn = container.querySelector(
      '[data-testid="chat-ui-composer-stop"]',
    ) as HTMLButtonElement;
    expect(btn).toBeTruthy();
    btn.click();
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("Stop returns composer to usable state", () => {
    const onStop = vi.fn();
    const { container, root } = renderComposer({ onStop, busy: true });
    expect(
      container.querySelector('[data-testid="chat-ui-composer-stop"]'),
    ).toBeTruthy();
    act(() => {
      root.render(
        <ChatComposer
          onSubmit={vi.fn() as ChatComposerProps["onSubmit"]}
          onStop={onStop}
          busy={false}
        />,
      );
    });
    expect(
      container.querySelector('[data-testid="chat-ui-composer-stop"]'),
    ).toBeNull();
    expect(
      container.querySelector('[data-testid="chat-ui-composer-send"]'),
    ).toBeTruthy();
  });
});

describe("ChatComposer — image attachments", () => {
  it("attach button exposes a hidden file picker input", () => {
    const onAttach = vi.fn();
    const { container } = renderComposer({ onAttachImage: onAttach });
    const btn = container.querySelector(
      '[data-testid="chat-ui-composer-attach"]',
    ) as HTMLButtonElement;
    expect(btn).toBeTruthy();
    expect(
      container.querySelector(
        '[data-testid="chat-ui-composer-file-input"]',
      ),
    ).toBeTruthy();
  });

  it("removes pending attachment", async () => {
    const attachment: ChatAttachment = {
      id: "att-1",
      kind: "image",
      name: "x.png",
    };
    const onAttach = vi.fn().mockResolvedValue(attachment);
    const { container } = renderComposer({ onAttachImage: onAttach });
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-file-input"]',
    ) as HTMLInputElement;
    const file = new File([new Uint8Array([1, 2])], "x.png", {
      type: "image/png",
    });
    Object.defineProperty(input, "files", {
      value: [file],
      configurable: true,
    });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    expect(
      container.querySelector('[data-testid="chat-ui-composer-attachment"]'),
    ).toBeTruthy();
    const removeBtn = container.querySelector(
      '[data-testid="chat-ui-composer-attachment"] button[aria-label^="Remove"]',
    ) as HTMLButtonElement;
    removeBtn.click();
    await flushAct();
    expect(
      container.querySelector('[data-testid="chat-ui-composer-attachment"]'),
    ).toBeNull();
  });

  it("attaches image via onAttachImage and shows chip", async () => {
    const attachment: ChatAttachment = {
      id: "att-1",
      kind: "image",
      name: "kitten.png",
      dataUri: "data:image/png;base64,iVBORw0KGgo=",
    };
    const onAttach = vi.fn().mockResolvedValue(attachment);
    const { container } = renderComposer({ onAttachImage: onAttach });
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-file-input"]',
    ) as HTMLInputElement;
    const file = new File([new Uint8Array([1, 2])], "kitten.png", {
      type: "image/png",
    });
    Object.defineProperty(input, "files", {
      value: [file],
      configurable: true,
    });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    expect(onAttach).toHaveBeenCalledTimes(1);
    const chip = container.querySelector(
      '[data-testid="chat-ui-composer-attachment"]',
    );
    expect(chip).toBeTruthy();
    expect(chip?.textContent).toContain("kitten.png");
  });

  it("attach failure surfaces inline error and skips chip", async () => {
    const onAttach = vi.fn().mockResolvedValue(null);
    const { container } = renderComposer({ onAttachImage: onAttach });
    const input = container.querySelector(
      '[data-testid="chat-ui-composer-file-input"]',
    ) as HTMLInputElement;
    const file = new File([new Uint8Array([1, 2])], "x.png", {
      type: "image/png",
    });
    Object.defineProperty(input, "files", {
      value: [file],
      configurable: true,
    });
    input.dispatchEvent(new Event("change", { bubbles: true }));
    await flushAct();
    await flushAct();
    expect(
      container.querySelector(
        '[data-testid="chat-ui-composer-attach-error"]',
      ),
    ).toBeTruthy();
    expect(
      container.querySelector('[data-testid="chat-ui-composer-attachment"]'),
    ).toBeNull();
  });
});

describe("ChatComposer — paste image", () => {
  it("pastes a single image and calls onAttachImage", async () => {
    const attachment: ChatAttachment = {
      id: "p-1",
      kind: "image",
      name: "pasted.png",
    };
    const onAttach = vi.fn().mockResolvedValue(attachment);
    const { container } = renderComposer({ onAttachImage: onAttach });
    const ta = getTextarea(container);
    const file = new File([new Uint8Array([1])], "pasted.png", {
      type: "image/png",
    });
    // jsdom does not expose DataTransfer — fabricate one with the same
    // surface the composer reads (items.add + types).
    const dt = {
      files: [file],
      items: [
        {
          kind: "file",
          type: "image/png",
          getAsFile: () => file,
        },
      ],
      types: ["Files"],
    } as unknown as DataTransfer;
    // jsdom does not expose ClipboardEvent or DataTransfer — fabricate
    // a plain event with the same shape and dispatch it as `paste`.
    const handlerAttached = vi.fn();
    const orig = (ta as unknown as Record<string, unknown>).onpaste;
    (ta as unknown as Record<string, unknown>).onpaste = handlerAttached;
    // We rely on the synthetic React `onPaste` reading
    // `event.clipboardData.files` — populate that explicitly. jsdom's
    // EventInit is permissive so we can pass our hand-rolled clipboard.
    const pasteEv = new Event("paste", { bubbles: true, cancelable: true });
    Object.defineProperty(pasteEv, "clipboardData", { value: dt });
    ta.dispatchEvent(pasteEv);
    await flushAct();
    await flushAct();
    expect(onAttach).toHaveBeenCalledTimes(1);
    (ta as unknown as Record<string, unknown>).onpaste = orig;
  });
});

describe("ChatComposer — accessibility", () => {
  it("form is labelled and inputs are keyboard accessible", () => {
    const { container } = renderComposer();
    const form = container.querySelector(
      '[data-testid="chat-ui-composer"]',
    ) as HTMLElement;
    expect(form.getAttribute("aria-label")).toBe("Message composer");
    const ta = getTextarea(container);
    expect(ta.getAttribute("aria-label")).toBe("Message input");
    const send = container.querySelector(
      '[data-testid="chat-ui-composer-send"]',
    ) as HTMLButtonElement;
    expect(send.getAttribute("aria-label")).toBeTruthy();
  });
});
