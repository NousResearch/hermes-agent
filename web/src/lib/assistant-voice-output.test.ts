// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";

import { assistantFinalText, speakAssistantFinal } from "./assistant-voice-output";

describe("assistant voice output", () => {
  it("accepts only successful structured message.complete payloads", () => {
    expect(assistantFinalText("message.complete", { status: "complete", text: "Done" })).toBe("Done");
    expect(assistantFinalText("message.complete", { status: "error", text: "secret failure" })).toBeNull();
    expect(assistantFinalText("message.delta", { text: "partial" })).toBeNull();
    expect(assistantFinalText("message.complete", { status: "complete", text: "  " })).toBeNull();
  });

  it("speaks through browser synthesis and settles once on end or cancellation", () => {
    class FakeUtterance {
      onend: (() => void) | null = null;
      onerror: (() => void) | null = null;
      readonly text: string;
      constructor(text: string) { this.text = text; }
    }
    vi.stubGlobal("SpeechSynthesisUtterance", FakeUtterance);
    const synth = { speak: vi.fn(), cancel: vi.fn() };
    const settled = vi.fn();
    const playback = speakAssistantFinal(synth, "Done", settled);
    expect(playback).not.toBeNull();
    const utterance = synth.speak.mock.calls[0][0] as SpeechSynthesisUtterance;
    expect(utterance.text).toBe("Done");
    utterance.onend?.({} as SpeechSynthesisEvent);
    playback?.cancel();
    expect(settled).toHaveBeenCalledOnce();
    expect(synth.cancel).toHaveBeenCalledOnce();
  });

  it("does not start playback when SpeechSynthesisUtterance is unavailable", () => {
    vi.stubGlobal("SpeechSynthesisUtterance", undefined);
    const synth = { speak: vi.fn(), cancel: vi.fn() };

    expect(speakAssistantFinal(synth, "Done", vi.fn())).toBeNull();
    expect(synth.speak).not.toHaveBeenCalled();
  });
});
