// @vitest-environment jsdom
import { afterEach, describe, expect, it, vi } from "vitest";

import { bindPtyCompositionBridge, createPtyCompositionForwarder } from "./pty-composition";

describe("createPtyCompositionForwarder", () => {
  afterEach(() => vi.useRealTimers());

  it("forwards committed dead-key text when xterm emits no onData", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ä");
    vi.runAllTimers();

    expect(send).toHaveBeenCalledExactlyOnceWith("ä");
  });

  it("leaves xterm's committed input alone when it arrives before the fallback", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ä");
    forwarder.noteTerminalData("äx");
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });

  it("forwards a pending composition after unrelated terminal data", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ä");
    forwarder.noteTerminalData("x");
    vi.advanceTimersByTime(15);
    expect(send).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);

    expect(send).toHaveBeenCalledExactlyOnceWith("ä");
  });

  it("forwards a pending composition when unrelated data precedes matching chunks", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ab");
    forwarder.noteTerminalData("x");
    forwarder.noteTerminalData("a");
    forwarder.noteTerminalData("b");
    vi.runAllTimers();

    expect(send).toHaveBeenCalledExactlyOnceWith("ab");
  });

  it("cancels a pending composition when matching text arrives in clean chunks", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ab");
    forwarder.noteTerminalData("a");
    forwarder.noteTerminalData("b");
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });

  it("ignores ESC/SGR data while matching composition chunks", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ab");
    forwarder.noteTerminalData("a");
    forwarder.noteTerminalData("\x1b[<0;10;10M");
    forwarder.noteTerminalData("b");
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });

  it("forwards a second composition after the first fallback completes", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ä");
    vi.runAllTimers();
    forwarder.onCompositionEnd("ö");
    vi.runAllTimers();

    expect(send).toHaveBeenNthCalledWith(1, "ä");
    expect(send).toHaveBeenNthCalledWith(2, "ö");
  });

  it("preserves an earlier rapid composition before scheduling the next", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("a");
    forwarder.onCompositionEnd("ä");
    vi.runAllTimers();

    expect(send).toHaveBeenNthCalledWith(1, "a");
    expect(send).toHaveBeenNthCalledWith(2, "ä");
  });

  it("cancels a pending composition on disposal", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ä");
    forwarder.dispose();
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });

  it("does not send an empty cancelled composition", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("");
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });
});

describe("mobile IME double-send dedup (#115505)", () => {
  afterEach(() => vi.useRealTimers());

  it("delivers a word exactly once across the three Android/Gboard double-send paths", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    // 1. compositionend fires twice with identical data.
    forwarder.onCompositionEnd("hello");
    vi.advanceTimersByTime(16);
    forwarder.onCompositionEnd("hello");
    vi.advanceTimersByTime(16);
    expect(send).toHaveBeenCalledExactlyOnceWith("hello");

    // 2. onData already carried the commit; a trailing compositionend must not re-arm the fallback.
    vi.advanceTimersByTime(100);
    forwarder.noteTerminalData("wor");
    forwarder.noteTerminalData("ld");
    forwarder.onCompositionEnd("world");
    vi.runAllTimers();
    expect(send).toHaveBeenCalledTimes(1);

    // 3. onData echoes (or strictly extends) text the composition path just committed.
    vi.advanceTimersByTime(100);
    forwarder.onCompositionEnd("again");
    vi.advanceTimersByTime(16);
    expect(send).toHaveBeenLastCalledWith("again");
    expect(forwarder.filterTerminalData("again")).toBe("");
    expect(forwarder.filterTerminalData("again ")).toBe(" ");
  });

  it("keeps real repeats and unrelated input: outside the echo window and for different text", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("hello");
    vi.advanceTimersByTime(16);
    vi.advanceTimersByTime(100);
    forwarder.onCompositionEnd("hello");
    vi.advanceTimersByTime(16);
    expect(send).toHaveBeenCalledTimes(2);

    forwarder.noteTerminalData("abc");
    forwarder.onCompositionEnd("xyz");
    vi.runAllTimers();
    expect(send).toHaveBeenLastCalledWith("xyz");

    expect(forwarder.filterTerminalData("other")).toBe("other");
    vi.advanceTimersByTime(100);
    expect(forwarder.filterTerminalData("xyz")).toBe("xyz");
  });
});

// xterm 6 reads a commit as `textarea.value.substring(<length at
// compositionstart>)`; after compositionend that slice must be the commit.
describe("bindPtyCompositionBridge", () => {
  function bridge() {
    const textarea = document.createElement("textarea");
    const armReplacementWindow = vi.fn();
    const cleanup = bindPtyCompositionBridge(
      textarea,
      createPtyCompositionForwarder(() => undefined),
      { armReplacementWindow, isReplacementInput: () => false },
    );
    return { armReplacementWindow, cleanup, textarea };
  }

  function compose(textarea: HTMLTextAreaElement, mirrorAfter: string, data: string) {
    textarea.dispatchEvent(new CompositionEvent("compositionstart"));
    const start = textarea.value.length;
    textarea.value = mirrorAfter;
    textarea.dispatchEvent(new CompositionEvent("compositionend", { data }));
    return textarea.value.substring(start);
  }

  it.each([
    ["a mid-line commit", "B1（）", "B1（语调奇怪）", "语调奇怪", "语调奇怪"],
    ["an appended commit", "B1", "B1语调奇怪", "语调奇怪", "语调奇怪"],
    ["a commit whose key left a space", "", "녕 ", "녕", "녕 "],
    ["a commit whose compositionend data is empty", "", "é", "", "é"],
  ])("leaves xterm the right slice for %s", (_, before, mirrorAfter, data, slice) => {
    const { textarea } = bridge();
    textarea.value = before;

    expect(compose(textarea, mirrorAfter, data)).toBe(slice);
  });

  it("stops rewriting the mirror after cleanup", () => {
    const { armReplacementWindow, cleanup, textarea } = bridge();
    textarea.value = "B1（）";
    cleanup();

    // Pre-condition: a mid-line commit, whose raw slice is the shifted tail.
    expect(compose(textarea, "B1（语调奇怪）", "语调奇怪")).toBe("调奇怪）");
    expect(textarea.value).toBe("B1（语调奇怪）");
    expect(armReplacementWindow).not.toHaveBeenCalled();
  });

  it("stands down when the mirror shrank below the commit", () => {
    const { textarea } = bridge();
    textarea.value = "B1（）";

    textarea.dispatchEvent(new CompositionEvent("compositionstart"));
    // Blur (and Enter/Ctrl-C) clears the mirror while a composition is open.
    textarea.value = "";
    textarea.dispatchEvent(new CompositionEvent("compositionend", { data: "语调奇怪" }));

    // xterm slices "" and the forwarder's fallback delivers the word once;
    // rewriting here would make xterm send a fragment as well.
    expect(textarea.value).toBe("");
  });

  it("stands down when the composition replaced text already in the mirror", () => {
    const { textarea } = bridge();
    textarea.value = "hello";

    textarea.dispatchEvent(new CompositionEvent("compositionstart"));
    // A recomposition (Gboard backspacing into a word) shrinks the mirror.
    textarea.value = "help";
    textarea.dispatchEvent(new CompositionEvent("compositionend", { data: "help" }));

    expect(textarea.value).toBe("help");
  });

  it("does not rewrite a compositionend that had no compositionstart", () => {
    const { textarea } = bridge();
    textarea.value = "B1（）";

    textarea.dispatchEvent(new CompositionEvent("compositionend", { data: "语调奇怪" }));

    expect(textarea.value).toBe("B1（）");
  });

  it("leaves a mobile-like surface to the replacement path", () => {
    const textarea = document.createElement("textarea");
    const armReplacementWindow = vi.fn();
    bindPtyCompositionBridge(
      textarea,
      createPtyCompositionForwarder(() => undefined),
      {
        armReplacementWindow,
        isReplacementInput: () => false,
        isMobileInput: () => true,
      },
    );
    textarea.value = "B1（）";

    expect(compose(textarea, "B1（语调奇怪）", "语调奇怪")).toBe("调奇怪）");
    expect(textarea.value).toBe("B1（语调奇怪）");
    // Standing down must not take the replacement window with it.
    expect(armReplacementWindow).toHaveBeenCalledTimes(1);
  });

  it("arms the replacement window and hands commits to the forwarder", () => {
    vi.useFakeTimers();
    try {
      const send = vi.fn();
      const armReplacementWindow = vi.fn();
      const textarea = document.createElement("textarea");
      bindPtyCompositionBridge(textarea, createPtyCompositionForwarder(send), {
        armReplacementWindow,
        isReplacementInput: (input) => input.inputType === "insertReplacementText",
      });
      const beforeInput = (inputType: string) =>
        textarea.dispatchEvent(new InputEvent("beforeinput", { data: "x", inputType }));

      beforeInput("insertText");
      expect(armReplacementWindow).not.toHaveBeenCalled();
      beforeInput("insertReplacementText");
      expect(armReplacementWindow).toHaveBeenCalledTimes(1);

      // No onData follows, so the forwarder's 16 ms fallback delivers the commit.
      textarea.dispatchEvent(new CompositionEvent("compositionend", { data: "语调" }));
      expect(armReplacementWindow).toHaveBeenCalledTimes(2);
      vi.advanceTimersByTime(16);
      expect(send).toHaveBeenCalledExactlyOnceWith("语调");
    } finally {
      vi.useRealTimers();
    }
  });
});
