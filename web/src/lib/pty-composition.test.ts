import { afterEach, describe, expect, it, vi } from "vitest";

import { createPtyCompositionForwarder } from "./pty-composition";

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
