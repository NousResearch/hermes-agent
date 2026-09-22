import { describe, expect, it } from "vitest";
import {
  containsArabic,
  containsLatin,
  detectDominantScript,
  getVoicePauseTimeoutMs,
  normalizeVoicePrompt,
  recognitionTranscript,
} from "./chat-voice";

describe("chat-voice utilities", () => {
  it("detects Arabic script accurately", () => {
    expect(containsArabic("ازيك يا هيرميس")).toBe(true);
    expect(containsArabic("hello world")).toBe(false);
    expect(containsArabic("hello يا صديقي")).toBe(true);
  });

  it("detects Latin script accurately", () => {
    expect(containsLatin("hello world")).toBe(true);
    expect(containsLatin("ازيك")).toBe(false);
    expect(containsLatin("Hermes عربي")).toBe(true);
  });

  it("identifies dominant script for auto-adaptive switching", () => {
    expect(detectDominantScript("صباح الخير يا هيرميس")).toBe("ar");
    expect(detectDominantScript("Good morning Hermes")).toBe("en");
    expect(detectDominantScript("12345 ... !?")).toBe("neutral");
    // Mixed: dominant script wins
    expect(detectDominantScript("اكتبلي كود Python")).toBe("ar");
    expect(detectDominantScript("Please write python code for me")).toBe("en");
  });

  it("calculates adaptive pause timeout with continuation bonus", () => {
    // Normal complete sentences get base timeout
    expect(getVoicePauseTimeoutMs("I want you to fix this")).toBe(1400);
    expect(getVoicePauseTimeoutMs("خلص الشغل ده")).toBe(1400);

    // Sentences ending in Arabic continuation conjunctions get extra pause bonus (+400ms)
    expect(getVoicePauseTimeoutMs("عاوزك تساعدني علشان")).toBe(1800);
    expect(getVoicePauseTimeoutMs("شغل الموسيقى و")).toBe(1800);
    expect(getVoicePauseTimeoutMs("مش عارف لكن")).toBe(1800);
    expect(getVoicePauseTimeoutMs("هو ده يعني")).toBe(1800);

    // Sentences ending in English continuation words get extra pause bonus (+400ms)
    expect(getVoicePauseTimeoutMs("Can you help me because")).toBe(1800);
    expect(getVoicePauseTimeoutMs("Run the command and")).toBe(1800);
    expect(getVoicePauseTimeoutMs("I think that")).toBe(1800);
  });

  it("normalizes prompts and extracts transcripts", () => {
    expect(normalizeVoicePrompt("  hello \n\t  world  ")).toBe("hello world");
    expect(
      recognitionTranscript({
        resultIndex: 0,
        results: [
          { 0: { transcript: "part one" }, isFinal: true },
          { 0: { transcript: "part two" }, isFinal: false },
        ],
      }),
    ).toEqual({
      final: "part one",
      interim: "part two",
    });
  });
});
