import { afterEach, describe, expect, it, vi } from "vitest";

import {
  createPtyCompositionForwarder,
  shouldForwardPtyBeforeInputCommit,
} from "./pty-composition";

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

  it("forwards iOS dictation beforeinput text when xterm emits no onData", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onBeforeInput("insertText", "hello from dictation", true);
    vi.runAllTimers();

    expect(send).toHaveBeenCalledExactlyOnceWith("hello from dictation");
  });

  it("deduplicates native beforeinput text when xterm emits the same commit", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onBeforeInput("insertText", "hello from dictation", true);
    forwarder.noteTerminalData("hello from ");
    forwarder.noteTerminalData("dictation");
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });

  it("ignores interim and replacement beforeinput events as fallback commits", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onBeforeInput("insertCompositionText", "partial", true);
    forwarder.onBeforeInput("insertReplacementText", "replacement", true);
    vi.runAllTimers();

    expect(send).not.toHaveBeenCalled();
  });
});

describe("shouldForwardPtyBeforeInputCommit", () => {
  it("recognizes final composition and mobile dictation-like insertions", () => {
    expect(
      shouldForwardPtyBeforeInputCommit(
        "insertFromComposition",
        "こんにちは",
        false,
      ),
    ).toBe(true);
    expect(shouldForwardPtyBeforeInputCommit("insertText", "hello", true)).toBe(
      true,
    );
  });

  it("rejects empty, interim, replacement, and desktop plain insertions", () => {
    expect(
      shouldForwardPtyBeforeInputCommit("insertFromComposition", "", true),
    ).toBe(false);
    expect(
      shouldForwardPtyBeforeInputCommit(
        "insertCompositionText",
        "hello",
        true,
      ),
    ).toBe(false);
    expect(
      shouldForwardPtyBeforeInputCommit("insertReplacementText", "hello", true),
    ).toBe(false);
    expect(shouldForwardPtyBeforeInputCommit("insertText", "h", true)).toBe(
      false,
    );
    expect(shouldForwardPtyBeforeInputCommit("insertText", "hello", false)).toBe(
      false,
    );
    expect(shouldForwardPtyBeforeInputCommit("insertFromPaste", "hello", true)).toBe(
      false,
    );
  });
});
