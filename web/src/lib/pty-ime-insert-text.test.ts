// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import {
  attachPtyTextareaInputGuards,
  createDanglingImeInsertTextTracker,
  isImePlaceholderKey,
} from "./pty-ime-insert-text";
import { createPtyCompositionForwarder } from "./pty-composition";

const keydown = (overrides: Partial<KeyboardEvent> = {}) =>
  new KeyboardEvent("keydown", { key: "Unidentified", ...overrides });
const insertText = (data: string) =>
  new InputEvent("beforeinput", { inputType: "insertText", data, bubbles: true, composed: false });

describe("isImePlaceholderKey", () => {
  it("recognises both Gboard shapes: keyCode 229 and key Unidentified", () => {
    expect(isImePlaceholderKey({ keyCode: 229, key: "" })).toBe(true);
    expect(isImePlaceholderKey({ keyCode: 0, key: "Unidentified" })).toBe(true);
    expect(isImePlaceholderKey({ keyCode: 76, key: "l" })).toBe(false);
  });
});

describe("createDanglingImeInsertTextTracker", () => {
  it("arms the fallback for a 229 keydown with no keyup before the input", () => {
    const tracker = createDanglingImeInsertTextTracker();
    tracker.onKeydown({ keyCode: 229, key: "Unidentified" });
    expect(tracker.shouldFallback({ inputType: "insertText", data: "l" })).toBe(true);
  });

  it("leaves plain desktop keydowns to xterm", () => {
    const tracker = createDanglingImeInsertTextTracker();
    tracker.onKeydown({ keyCode: 76, key: "l" });
    expect(tracker.shouldFallback({ inputType: "insertText", data: "l" })).toBe(false);
  });

  it("stands down once the placeholder keyup arrived before the input", () => {
    const tracker = createDanglingImeInsertTextTracker();
    tracker.onKeydown({ keyCode: 229, key: "Unidentified" });
    tracker.onKeyup({ keyCode: 229, key: "Unidentified" });
    expect(tracker.shouldFallback({ inputType: "insertText", data: "l" })).toBe(false);
  });

  it("ignores non-insertText input types and empty data", () => {
    const tracker = createDanglingImeInsertTextTracker();
    tracker.onKeydown({ keyCode: 229, key: "Unidentified" });
    expect(tracker.shouldFallback({ inputType: "insertCompositionText", data: "l" })).toBe(false);
    expect(tracker.shouldFallback({ inputType: "insertText", data: "" })).toBe(false);
    expect(tracker.shouldFallback({ inputType: "insertText", data: null })).toBe(false);
  });

  it("does not arm for a composition commit's trailing insertText", () => {
    const tracker = createDanglingImeInsertTextTracker();
    tracker.onKeydown({ keyCode: 229, key: "Unidentified" });
    tracker.noteCompositionEnd();
    // Trailing insertText of a real composition commit lands within ms.
    expect(
      tracker.shouldFallback({ inputType: "insertText", data: "你好" }, Date.now() + 20),
    ).toBe(false);
    // Much later (a fresh dangling keydown chain, Gboard English) it arms again.
    expect(
      tracker.shouldFallback({ inputType: "insertText", data: "l" }, Date.now() + 200),
    ).toBe(true);
  });
});

describe("attachPtyTextareaInputGuards", () => {
  const makeHooks = () => ({
    isMobileLike: true,
    markReplacementWindow: vi.fn(),
    onCompositionCommit: vi.fn(),
    onDanglingInsertText: vi.fn(),
  });

  it("routes a Gboard dangling-229 insertText to the fallback hook", () => {
    const textarea = document.createElement("textarea");
    const hooks = makeHooks();
    const cleanup = attachPtyTextareaInputGuards(textarea, hooks);

    textarea.dispatchEvent(keydown());
    textarea.dispatchEvent(insertText("l"));

    expect(hooks.onDanglingInsertText).toHaveBeenCalledExactlyOnceWith("l");
    cleanup();
  });

  it("leaves desktop insertText (real keydown) to xterm", () => {
    const textarea = document.createElement("textarea");
    const hooks = makeHooks();
    const cleanup = attachPtyTextareaInputGuards(textarea, hooks);

    textarea.dispatchEvent(keydown({ keyCode: 76, key: "l" }));
    textarea.dispatchEvent(insertText("l"));

    expect(hooks.onDanglingInsertText).not.toHaveBeenCalled();
    cleanup();
  });

  it("marks the replacement window and forwards composition commits", () => {
    const textarea = document.createElement("textarea");
    const hooks = makeHooks();
    const cleanup = attachPtyTextareaInputGuards(textarea, hooks);

    textarea.dispatchEvent(new CompositionEvent("compositionend", { data: "你好" }));

    expect(hooks.markReplacementWindow).toHaveBeenCalledExactlyOnceWith();
    expect(hooks.onCompositionCommit).toHaveBeenCalledExactlyOnceWith("你好");
    // The commit also blocks a trailing insertText from re-entering the fallback.
    textarea.dispatchEvent(insertText("你好"));
    expect(hooks.onDanglingInsertText).not.toHaveBeenCalled();
    cleanup();
  });

  it("stops listening after cleanup", () => {
    const textarea = document.createElement("textarea");
    const hooks = makeHooks();
    const cleanup = attachPtyTextareaInputGuards(textarea, hooks);
    cleanup();

    textarea.dispatchEvent(keydown());
    textarea.dispatchEvent(insertText("l"));
    textarea.dispatchEvent(new CompositionEvent("compositionend", { data: "x" }));

    expect(hooks.onDanglingInsertText).not.toHaveBeenCalled();
    expect(hooks.onCompositionCommit).not.toHaveBeenCalled();
  });

  it("end-to-end: a Gboard letter reaches the PTY exactly once via the forwarder", () => {
    vi.useFakeTimers();
    const textarea = document.createElement("textarea");
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);
    const cleanup = attachPtyTextareaInputGuards(textarea, {
      isMobileLike: true,
      markReplacementWindow: () => undefined,
      onCompositionCommit: (data) => forwarder.onCompositionEnd(data),
      onDanglingInsertText: (data) => forwarder.onInsertTextFallback(data),
    });

    // Gboard sequence per letter: keydown 229 (no keyup) -> insertText.
    textarea.dispatchEvent(keydown());
    textarea.dispatchEvent(insertText("l"));
    // xterm swallows it (no onData) — the fallback timer delivers it.
    vi.advanceTimersByTime(16);
    expect(send).toHaveBeenCalledExactlyOnceWith("l");

    // xterm's late echo must be filtered by the onData path.
    expect(forwarder.filterTerminalData("l")).toBe("");
    cleanup();
    vi.useRealTimers();
  });
});
