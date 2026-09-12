import { describe, expect, it, vi } from "vitest";

import { submitVoiceTranscriptToPty } from "./pty-voice-submit";

describe("submitVoiceTranscriptToPty", () => {
  it("sends Return only after xterm reports the transcript render", () => {
    const socket = { readyState: WebSocket.OPEN, send: vi.fn() };
    const current = () => socket;
    let terminalReady: (() => void) | undefined;

    expect(
      submitVoiceTranscriptToPty(current, "Call the crew", (ready) => {
        terminalReady = ready;
      }, vi.fn()),
    ).toBe(true);
    expect(socket.send).toHaveBeenCalledTimes(1);
    expect(socket.send).toHaveBeenNthCalledWith(1, "Call the crew");
    terminalReady?.();
    expect(socket.send).toHaveBeenCalledTimes(2);
    expect(socket.send).toHaveBeenNthCalledWith(2, "\r");
  });

  it("does not send Return through a replacement or closed socket", () => {
    const first = { readyState: WebSocket.OPEN, send: vi.fn<(data: string) => void>() };
    const second = { readyState: WebSocket.OPEN, send: vi.fn<(data: string) => void>() };
    let active: { readyState: number; send(data: string): void } = first;
    let terminalReady: (() => void) | undefined;
    const onReturnFailed = vi.fn();
    expect(
      submitVoiceTranscriptToPty(() => active, "Hello", (ready) => {
        terminalReady = ready;
      }, onReturnFailed),
    ).toBe(true);
    active = second;
    terminalReady?.();
    expect(first.send).toHaveBeenCalledOnce();
    expect(second.send).not.toHaveBeenCalled();
    expect(onReturnFailed).toHaveBeenCalledOnce();

    active = { readyState: WebSocket.CLOSED, send: vi.fn<(data: string) => void>() };
    expect(
      submitVoiceTranscriptToPty(
        () => active,
        "Nope",
        () => undefined,
        onReturnFailed,
      ),
    ).toBe(false);
  });
});
