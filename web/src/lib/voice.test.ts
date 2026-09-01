import { describe, expect, it, vi } from "vitest";

import { appendVoiceTranscript, canRecordVoice, chooseRecordingMimeType } from "./voice";

describe("voice composer helpers", () => {
  it("chooses a browser-supported audio MIME type in stable order", () => {
    expect(chooseRecordingMimeType(["audio/ogg", "audio/webm;codecs=opus"])).toBe("audio/webm;codecs=opus");
    expect(chooseRecordingMimeType(["audio/wav"])).toBe("audio/wav");
    expect(chooseRecordingMimeType(["video/mp4"])).toBeUndefined();
  });

  it("appends a non-empty transcript without changing its Unicode content", () => {
    expect(appendVoiceTranscript("เดิม", "  สวัสดี ั  ")).toBe("เดิม สวัสดี ั");
    expect(appendVoiceTranscript("", "  hello ")).toBe("hello");
    expect(appendVoiceTranscript("draft", "  ")).toBe("draft");
  });

  it("reports whether both browser recording capabilities exist", () => {
    expect(canRecordVoice({ getUserMedia: vi.fn() }, { isTypeSupported: vi.fn() })).toBe(true);
    expect(canRecordVoice(undefined, { isTypeSupported: vi.fn() })).toBe(false);
  });
});
