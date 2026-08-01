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

  it("keeps the fallback pending when xterm emits unrelated input", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("ä");
    forwarder.noteTerminalData("x");
    vi.runAllTimers();

    expect(send).toHaveBeenCalledExactlyOnceWith("ä");
  });

  it("supersedes an earlier composition and cancels it on disposal", () => {
    vi.useFakeTimers();
    const send = vi.fn();
    const forwarder = createPtyCompositionForwarder(send);

    forwarder.onCompositionEnd("a");
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
