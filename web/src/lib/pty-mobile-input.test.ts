import { describe, expect, it } from "vitest";

import {
  isIosLikeUserAgent,
  normalizePtyMobileHangulInput,
  normalizePtyMobileInput,
  shouldSuppressPtyImeData,
  shouldTreatInputAsMobileReplacement,
  updatePtyInputLine,
} from "./pty-mobile-input";

describe("shouldTreatInputAsMobileReplacement", () => {
  it("recognizes explicit browser replacement input", () => {
    expect(
      shouldTreatInputAsMobileReplacement(
        "insertReplacementText",
        "Kain",
        false,
      ),
    ).toBe(true);
    expect(
      shouldTreatInputAsMobileReplacement(
        "insertFromComposition",
        "Kain",
        false,
      ),
    ).toBe(true);
    expect(
      shouldTreatInputAsMobileReplacement(
        "insertCompositionText",
        "Kain",
        false,
      ),
    ).toBe(true);
  });

  it("treats multi-character mobile insertText as replacement-like", () => {
    expect(
      shouldTreatInputAsMobileReplacement("insertText", "Kain", true),
    ).toBe(true);
    expect(shouldTreatInputAsMobileReplacement("insertText", "K", true)).toBe(
      false,
    );
    expect(
      shouldTreatInputAsMobileReplacement("insertText", "Kain", false),
    ).toBe(false);
  });
});

describe("isIosLikeUserAgent", () => {
  it("detects iPhone, iPad, and desktop-mode iPad", () => {
    expect(isIosLikeUserAgent("Mozilla/5.0 (iPhone; CPU iPhone OS 18_0)")).toBe(
      true,
    );
    expect(isIosLikeUserAgent("Mozilla/5.0 (iPad; CPU OS 18_0)")).toBe(true);
    expect(
      isIosLikeUserAgent(
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) AppleWebKit/605.1.15",
        5,
      ),
    ).toBe(true);
  });

  it("does not enable the workaround for Android or desktop macOS", () => {
    expect(
      isIosLikeUserAgent("Mozilla/5.0 (Linux; Android 15; Pixel 9)", 5),
    ).toBe(false);
    expect(
      isIosLikeUserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)", 0),
    ).toBe(false);
  });
});

describe("shouldSuppressPtyImeData", () => {
  it("keeps decomposed Hangul keystrokes out of the PTY until mobile composition ends", () => {
    const composingChunks = ["ㅎ", "ㅏ", "ㄴ"];

    expect(
      composingChunks.filter(
        (data) => !shouldSuppressPtyImeData(data, true, true),
      ),
    ).toEqual([]);
    expect(shouldSuppressPtyImeData("한", false, true)).toBe(false);
  });

  it("does not alter native terminal or non-composing mobile input", () => {
    expect(shouldSuppressPtyImeData("ㅎ", true, false)).toBe(false);
    expect(shouldSuppressPtyImeData("hello", false, true)).toBe(false);
  });

  it("suppresses Enter and delete used by the mobile IME during composition", () => {
    expect(shouldSuppressPtyImeData("\r", true, true)).toBe(true);
    expect(shouldSuppressPtyImeData("\x7f", true, true)).toBe(true);
  });
});

describe("normalizePtyMobileHangulInput", () => {
  const typeJamo = (sequence: string) => {
    let line = "";
    let rawComposition = "";
    const outbound: string[] = [];
    for (const data of sequence) {
      const result = normalizePtyMobileHangulInput(
        data,
        line,
        true,
        rawComposition,
      );
      line = result.nextLine;
      rawComposition = result.nextRawComposition;
      outbound.push(result.data);
    }
    return { line, outbound, rawComposition };
  };

  it("turns decomposed iPhone Korean keystrokes into composed syllables", () => {
    const result = typeJamo("ㅎㅏㄴㄱㅡㄹ");

    expect(result.line).toBe("한글");
    expect(result.outbound).toEqual([
      "ㅎ",
      "\x7f하",
      "\x7f한",
      "ㄱ",
      "\x7f그",
      "\x7f글",
    ]);
  });

  it("moves a final consonant to the next syllable when a vowel follows", () => {
    expect(typeJamo("ㅎㅏㄴㅏ").line).toBe("하나");
  });

  it("supports compound vowels and final consonants", () => {
    expect(typeJamo("ㄱㅘ").line).toBe("과");
    expect(typeJamo("ㄷㅏㄹㄱ").line).toBe("닭");
    expect(typeJamo("ㄷㅏㄹㄱㅏ").line).toBe("달가");
  });

  it("rewinds the active raw composition one jamo at a time", () => {
    const state = typeJamo("ㅎㅏㄴ");
    expect(state.line).toBe("한");

    const first = normalizePtyMobileHangulInput(
      "\x7f",
      state.line,
      true,
      state.rawComposition,
    );
    expect(first).toMatchObject({
      data: "\x7f하",
      nextLine: "하",
      nextRawComposition: "ㅎㅏ",
    });

    const second = normalizePtyMobileHangulInput(
      "\x7f",
      first.nextLine,
      true,
      first.nextRawComposition,
    );
    expect(second).toMatchObject({
      data: "\x7fㅎ",
      nextLine: "ㅎ",
      nextRawComposition: "ㅎ",
    });
  });

  it("rewinds compound vowels and finals without deleting the whole syllable", () => {
    const compoundVowel = typeJamo("ㄱㅗㅏ");
    expect(
      normalizePtyMobileHangulInput(
        "\x7f",
        compoundVowel.line,
        true,
        compoundVowel.rawComposition,
      ),
    ).toMatchObject({ nextLine: "고", nextRawComposition: "ㄱㅗ" });

    const compoundFinal = typeJamo("ㄷㅏㄹㄱ");
    expect(
      normalizePtyMobileHangulInput(
        "\x7f",
        compoundFinal.line,
        true,
        compoundFinal.rawComposition,
      ),
    ).toMatchObject({ nextLine: "달", nextRawComposition: "ㄷㅏㄹ" });
  });

  it("replaces a partially synthesized suffix when compositionstart arrives late", () => {
    const partial = typeJamo("ㅇㅏㄴㄴㅕ");
    expect(partial.line).toBe("안녀");

    const reconciled = normalizePtyMobileHangulInput(
      "녕",
      partial.line,
      true,
      partial.rawComposition,
      true,
    );
    expect(reconciled).toMatchObject({
      data: "\x7f녕",
      nextLine: "안녕",
      nextRawComposition: "",
      normalized: true,
    });
  });

  it("suppresses a late native final that duplicates synthesized raw jamo", () => {
    const synthesized = typeJamo("ㅎㅏㄴ");
    const reconciled = normalizePtyMobileHangulInput(
      "한",
      synthesized.line,
      true,
      synthesized.rawComposition,
    );

    expect(reconciled).toMatchObject({
      data: "",
      nextLine: "한",
      nextRawComposition: "",
      normalized: true,
    });
  });

  it("resets raw composition tracking on cursor controls", () => {
    const state = typeJamo("ㅎㅏㄴ");
    expect(
      normalizePtyMobileHangulInput(
        "\x1b[D",
        state.line,
        true,
        state.rawComposition,
      ),
    ).toMatchObject({
      data: "\x1b[D",
      nextLine: "",
      nextRawComposition: "",
      normalized: false,
    });
  });

  it("does not reopen a finalized or pasted syllable as active composition", () => {
    expect(normalizePtyMobileHangulInput("ㅏ", "한", true, "").nextLine).toBe(
      "한ㅏ",
    );
    expect(normalizePtyMobileHangulInput("ㄴ", "가", true, "").nextLine).toBe(
      "가ㄴ",
    );
  });

  it("leaves desktop and non-jamo text untouched", () => {
    expect(normalizePtyMobileHangulInput("ㅎ", "", false)).toEqual({
      data: "ㅎ",
      nextLine: "ㅎ",
      normalized: false,
      nextRawComposition: "",
    });
    expect(normalizePtyMobileHangulInput("hello", "", true)).toEqual({
      data: "hello",
      nextLine: "hello",
      normalized: false,
      nextRawComposition: "",
    });
  });
});

describe("normalizePtyMobileInput", () => {
  it("turns a Gboard full-line suggestion into a line replacement", () => {
    const result = normalizePtyMobileInput(
      "hello my name is Kain Kain",
      "hello my name is kain",
      true,
    );

    expect(result.normalized).toBe(true);
    expect(result.nextLine).toBe("hello my name is Kain");
    expect(result.data).toBe(
      "\x7f".repeat("hello my name is kain".length) + "hello my name is Kain",
    );
  });

  it("turns a Gboard last-word suggestion into a last-word replacement", () => {
    const result = normalizePtyMobileInput(
      "Kain",
      "hello my name is kain",
      true,
    );

    expect(result.normalized).toBe(true);
    expect(result.nextLine).toBe("hello my name is Kain");
    expect(result.data).toBe(
      "\x7f".repeat("hello my name is kain".length) + "hello my name is Kain",
    );
  });

  it("does not normalize ordinary appends when replacement is not active", () => {
    const result = normalizePtyMobileInput(
      "hello my name is Kain Kain",
      "hello my name is kain",
      false,
    );

    expect(result.normalized).toBe(false);
    expect(result.nextLine).toBe(
      "hello my name is kainhello my name is Kain Kain",
    );
  });

  it("does not normalize control input", () => {
    const result = normalizePtyMobileInput("\r", "hello", true);

    expect(result.normalized).toBe(false);
    expect(result.nextLine).toBe("");
    expect(result.data).toBe("\r");
  });

  it("does not collapse legitimate single-letter reduplication", () => {
    // "a a" is a plausible thing to type; the >=2-char guard keeps the
    // duplicate-final-word collapse from eating it inside the window.
    const result = normalizePtyMobileInput("a a", "a", true);

    expect(result.normalized).toBe(false);
    expect(result.data).toBe("a a");
  });
});

describe("updatePtyInputLine", () => {
  it("tracks printable text, delete, and submit", () => {
    expect(updatePtyInputLine("", "abc")).toBe("abc");
    expect(updatePtyInputLine("abc", "\x7f")).toBe("ab");
    expect(updatePtyInputLine("abc", "\r")).toBe("");
  });

  it("resets tracking on escape sequences instead of appending their payload", () => {
    // Left-arrow arrives as one CSI chunk; the tracker cannot model cursor
    // moves, so it must disarm rather than record "hello[D".
    expect(updatePtyInputLine("hello", "\x1b[D")).toBe("");
    expect(updatePtyInputLine("hello", "\x1b[H")).toBe("");
    expect(updatePtyInputLine("hello", "\x1bOP")).toBe("");
  });
});

describe("normalizePtyMobileInput after cursor movement", () => {
  it("does not emit a replacement against a tracker reset by arrow keys", () => {
    // Simulate: type "hello my name is kain", press left-arrow, then a
    // Gboard suggestion arrives. The tracker reset means no replacement
    // heuristic can fire against a stale line snapshot.
    const afterArrow = updatePtyInputLine("hello my name is kain", "\x1b[D");
    const result = normalizePtyMobileInput("Kain", afterArrow, true);

    expect(result.normalized).toBe(false);
    expect(result.data).toBe("Kain");
  });
});
