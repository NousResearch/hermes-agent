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

describe("mobile IME double-send dedup", () => {
  afterEach(() => vi.useRealTimers());

  it("ignores a duplicate compositionend carrying identical data", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();
    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();

    expect(send).toHaveBeenCalledTimes(1);
    expect(send).toHaveBeenCalledWith("hello");
  });

  it("still forwards a re-typed word once the echo window has passed", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();
    vi.advanceTimersByTime(81);
    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();

    expect(send).toHaveBeenCalledTimes(2);
  });

  it("ignores a compositionend trailing terminal data that already carried the commit", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    // Android: xterm onData fires first with the committed word…
    forwarder.noteTerminalData("hello");
    // …and the late compositionend would arm a fresh fallback duplicate.
    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });

  it("recognizes the commit when onData carries it split across chunks", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.noteTerminalData("he");
    forwarder.noteTerminalData("llo");
    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });

  it("still sends when terminal data differs from the trailing compositionend", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.noteTerminalData("hello");
    forwarder.onCompositionEnd("world");
    vi.runAllTimers();

    expect(send).toHaveBeenCalledExactlyOnceWith("world");
  });

  it("drops an onData payload that echoes just-committed composition text", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();
    expect(send).toHaveBeenCalledExactlyOnceWith("hello");

    expect(forwarder.filterTerminalData("hello")).toBe("");
  });

  it("forwards only the new suffix of a strict extension", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();

    expect(forwarder.filterTerminalData("hello world")).toBe(" world");
  });

  it("passes unrelated onData through unchanged", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();

    expect(forwarder.filterTerminalData("world")).toBe("world");
  });

  it("passes onData through once the echo window has passed", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("hello");
    vi.runAllTimers();
    vi.advanceTimersByTime(81);

    expect(forwarder.filterTerminalData("hello")).toBe("hello");
  });
});
