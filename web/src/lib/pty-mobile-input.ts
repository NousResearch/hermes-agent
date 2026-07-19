const DELETE = "\x7f";

// How long (ms) after a mobile IME / replacement event we treat subsequent
// terminal input as a candidate line-replacement rather than a plain append.
// Exported so the ChatPage integration and tests share one tunable value.
export const MOBILE_REPLACEMENT_WINDOW_MS = 350;

export function isIosLikeUserAgent(
  userAgent: string,
  maxTouchPoints = 0,
): boolean {
  return (
    /iPhone|iPad|iPod/i.test(userAgent) ||
    (/Macintosh/i.test(userAgent) && maxTouchPoints > 1)
  );
}

function chars(text: string): string[] {
  return Array.from(text);
}

function removeLastChar(text: string): string {
  const c = chars(text);
  c.pop();
  return c.join("");
}

function isPlainText(data: string): boolean {
  return !/[\x00-\x1f\x7f]/.test(data);
}

function lastWordMatch(line: string): RegExpMatchArray | null {
  return line.match(/^(.*?)(\S+)(\s*)$/u);
}

function collapseDuplicatedFinalWord(
  text: string,
  previousLine: string,
): string {
  const match = text.match(/^(.*?)(\S+)(\s+)(\S+)(\s*)$/u);
  if (!match) return text;

  const [, prefix, first, , second, trailing] = match;
  if (first.toLocaleLowerCase() !== second.toLocaleLowerCase()) return text;
  // Only collapse a duplication the tracked line already ended with — i.e.
  // Gboard re-emitted the final word. Requiring a >=2-char word avoids
  // eating legitimate single-letter reduplication ("a a", "i i") that a
  // user may genuinely type inside the replacement window.
  if (first.length < 2) return text;
  if (
    !previousLine
      .trimEnd()
      .toLocaleLowerCase()
      .endsWith(first.toLocaleLowerCase())
  ) {
    return text;
  }
  return `${prefix}${first}${trailing}`;
}

function replacementLineForMobileInput(
  currentLine: string,
  incoming: string,
): string | null {
  if (!currentLine || currentLine.length < 2 || !incoming) return null;

  const currentLower = currentLine.toLocaleLowerCase();
  const incomingLower = incoming.toLocaleLowerCase();

  if (incomingLower.startsWith(currentLower)) {
    return collapseDuplicatedFinalWord(incoming, currentLine);
  }

  const word = lastWordMatch(currentLine);
  if (!word) return null;

  const [, prefix, last, trailing] = word;
  if (trailing) return null;

  const incomingFirst = incoming.trimStart().split(/\s+/u)[0] ?? "";
  if (
    incomingFirst &&
    incomingFirst.toLocaleLowerCase() === last.toLocaleLowerCase()
  ) {
    return `${prefix}${collapseDuplicatedFinalWord(incoming, currentLine)}`;
  }

  return null;
}

const HANGUL_BASE = 0xac00;
const HANGUL_END = 0xd7a3;
const JUNG_COUNT = 21;
const JONG_COUNT = 28;

const CHOSEONG = Array.from("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ");
const JUNGSEONG = Array.from("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ");
const JONGSEONG = [
  "",
  "ㄱ",
  "ㄲ",
  "ㄳ",
  "ㄴ",
  "ㄵ",
  "ㄶ",
  "ㄷ",
  "ㄹ",
  "ㄺ",
  "ㄻ",
  "ㄼ",
  "ㄽ",
  "ㄾ",
  "ㄿ",
  "ㅀ",
  "ㅁ",
  "ㅂ",
  "ㅄ",
  "ㅅ",
  "ㅆ",
  "ㅇ",
  "ㅈ",
  "ㅊ",
  "ㅋ",
  "ㅌ",
  "ㅍ",
  "ㅎ",
];

const COMPOUND_JUNG: Record<string, string> = {
  ㅗㅏ: "ㅘ",
  ㅗㅐ: "ㅙ",
  ㅗㅣ: "ㅚ",
  ㅜㅓ: "ㅝ",
  ㅜㅔ: "ㅞ",
  ㅜㅣ: "ㅟ",
  ㅡㅣ: "ㅢ",
};
const COMPOUND_JONG: Record<string, string> = {
  ㄱㅅ: "ㄳ",
  ㄴㅈ: "ㄵ",
  ㄴㅎ: "ㄶ",
  ㄹㄱ: "ㄺ",
  ㄹㅁ: "ㄻ",
  ㄹㅂ: "ㄼ",
  ㄹㅅ: "ㄽ",
  ㄹㅌ: "ㄾ",
  ㄹㅍ: "ㄿ",
  ㄹㅎ: "ㅀ",
  ㅂㅅ: "ㅄ",
};
const SPLIT_JONG = Object.fromEntries(
  Object.entries(COMPOUND_JONG).map(([pair, compound]) => [
    compound,
    Array.from(pair),
  ]),
) as Record<string, string[]>;

function composeSyllable(cho: number, jung: number, jong = 0): string {
  return String.fromCodePoint(
    HANGUL_BASE + (cho * JUNG_COUNT + jung) * JONG_COUNT + jong,
  );
}

function decomposeSyllable(
  char: string,
): { cho: number; jung: number; jong: number } | null {
  const code = char.codePointAt(0);
  if (code === undefined || code < HANGUL_BASE || code > HANGUL_END)
    return null;
  const offset = code - HANGUL_BASE;
  return {
    cho: Math.floor(offset / (JUNG_COUNT * JONG_COUNT)),
    jung: Math.floor((offset % (JUNG_COUNT * JONG_COUNT)) / JONG_COUNT),
    jong: offset % JONG_COUNT,
  };
}

function composeHangulJamo(line: string, jamo: string): string {
  const lineChars = chars(line);
  const last = lineChars.at(-1);
  if (!last) return jamo;

  const vowel = JUNGSEONG.indexOf(jamo);
  const consonant = CHOSEONG.indexOf(jamo);
  const syllable = decomposeSyllable(last);

  if (vowel >= 0) {
    if (syllable) {
      if (syllable.jong === 0) {
        const combined = COMPOUND_JUNG[`${JUNGSEONG[syllable.jung]}${jamo}`];
        if (!combined) return line + jamo;
        lineChars[lineChars.length - 1] = composeSyllable(
          syllable.cho,
          JUNGSEONG.indexOf(combined),
        );
        return lineChars.join("");
      }

      const final = JONGSEONG[syllable.jong];
      const split = SPLIT_JONG[final] ?? ["", final];
      const nextCho = CHOSEONG.indexOf(split[1]);
      if (nextCho < 0) return line + jamo;
      lineChars[lineChars.length - 1] = composeSyllable(
        syllable.cho,
        syllable.jung,
        JONGSEONG.indexOf(split[0]),
      );
      return lineChars.join("") + composeSyllable(nextCho, vowel);
    }

    const lastCho = CHOSEONG.indexOf(last);
    if (lastCho >= 0) {
      lineChars[lineChars.length - 1] = composeSyllable(lastCho, vowel);
      return lineChars.join("");
    }

    const lastVowel = JUNGSEONG.indexOf(last);
    const combined =
      lastVowel >= 0 ? COMPOUND_JUNG[`${last}${jamo}`] : undefined;
    if (combined) {
      lineChars[lineChars.length - 1] = combined;
      return lineChars.join("");
    }
    return line + jamo;
  }

  if (consonant >= 0 && syllable) {
    const jong = JONGSEONG.indexOf(jamo);
    if (syllable.jong === 0 && jong > 0) {
      lineChars[lineChars.length - 1] = composeSyllable(
        syllable.cho,
        syllable.jung,
        jong,
      );
      return lineChars.join("");
    }
    const combined = COMPOUND_JONG[`${JONGSEONG[syllable.jong]}${jamo}`];
    if (combined) {
      lineChars[lineChars.length - 1] = composeSyllable(
        syllable.cho,
        syllable.jung,
        JONGSEONG.indexOf(combined),
      );
      return lineChars.join("");
    }
  }

  return line + jamo;
}

function composeRawJamo(raw: string): string {
  let output = "";
  for (const jamo of chars(raw)) {
    output = output ? composeHangulJamo(output, jamo) : jamo;
  }
  return output;
}

function rewriteLineData(currentLine: string, nextLine: string): string {
  const before = chars(currentLine);
  const after = chars(nextLine);
  let common = 0;
  while (
    common < before.length &&
    common < after.length &&
    before[common] === after[common]
  ) {
    common += 1;
  }
  return DELETE.repeat(before.length - common) + after.slice(common).join("");
}

function withoutSuffix(text: string, suffix: string): string {
  const textChars = chars(text);
  return textChars.slice(0, textChars.length - chars(suffix).length).join("");
}

function isCompatibilityJamoData(data: string): boolean {
  const incoming = chars(data);
  return (
    incoming.length > 0 &&
    incoming.every(
      (char) => CHOSEONG.includes(char) || JUNGSEONG.includes(char),
    )
  );
}

function isPrecomposedHangulData(data: string): boolean {
  const incoming = chars(data);
  return (
    incoming.length > 0 && incoming.every((char) => decomposeSyllable(char))
  );
}

export function normalizePtyMobileHangulInput(
  data: string,
  currentLine: string,
  isMobileLike: boolean,
  rawComposition = "",
  reconcileNativeFinal = false,
): {
  data: string;
  nextLine: string;
  normalized: boolean;
  nextRawComposition: string;
} {
  const rawDisplay = rawComposition ? composeRawJamo(rawComposition) : "";
  const stateAligned =
    Boolean(rawComposition) &&
    Boolean(rawDisplay) &&
    currentLine.endsWith(rawDisplay);
  const activeRaw = stateAligned ? rawComposition : "";
  const baseLine = stateAligned
    ? withoutSuffix(currentLine, rawDisplay)
    : currentLine;

  if (isMobileLike && activeRaw && (data === DELETE || data === "\b")) {
    const nextRawComposition = chars(activeRaw).slice(0, -1).join("");
    const nextLine = baseLine + composeRawJamo(nextRawComposition);
    return {
      data: rewriteLineData(currentLine, nextLine),
      nextLine,
      normalized: true,
      nextRawComposition,
    };
  }

  if (
    isMobileLike &&
    activeRaw &&
    reconcileNativeFinal &&
    isPrecomposedHangulData(data)
  ) {
    const rawDisplayChars = chars(rawDisplay);
    const reconciledDisplay = rawDisplayChars.slice(0, -1).join("") + data;
    const nextLine = baseLine + reconciledDisplay;
    return {
      data: rewriteLineData(currentLine, nextLine),
      nextLine,
      normalized: true,
      nextRawComposition: "",
    };
  }

  // Some WebKit paths emit raw jamo first and still produce a late finalized
  // syllable. The fallback has already drawn that same suffix, so forwarding
  // the native final would duplicate it ("한한"). Treat it as an acknowledgement
  // of the synthesized text and close the raw-composition state.
  if (
    isMobileLike &&
    activeRaw &&
    isPrecomposedHangulData(data) &&
    rawDisplay.endsWith(data) &&
    currentLine.endsWith(data)
  ) {
    return {
      data: "",
      nextLine: currentLine,
      normalized: true,
      nextRawComposition: "",
    };
  }

  if (!isMobileLike || !isCompatibilityJamoData(data)) {
    return {
      data,
      nextLine: updatePtyInputLine(currentLine, data),
      normalized: false,
      nextRawComposition: "",
    };
  }

  const nextRawComposition = activeRaw + data;
  const nextLine = baseLine + composeRawJamo(nextRawComposition);
  return {
    data: rewriteLineData(currentLine, nextLine),
    nextLine,
    normalized: nextLine !== currentLine + data,
    nextRawComposition,
  };
}

export function shouldSuppressPtyImeData(
  _data: string,
  composing: boolean,
  isMobileLike: boolean,
): boolean {
  // xterm.js can emit each physical Korean key (ㅎ, ㅏ, ㄴ, …) through
  // onData while iOS still owns the same keystrokes as an active IME
  // composition. Forwarding those bytes makes the PTY editor commit jamo
  // before Safari can replace them with the finalized syllable. Keep every
  // composition keystroke browser-side; after compositionend xterm emits the
  // finalized text through its normal onData path.
  return isMobileLike && composing;
}

export function shouldTreatInputAsMobileReplacement(
  inputType: string | undefined,
  data: string | null | undefined,
  isMobileLike: boolean,
): boolean {
  if (
    inputType === "insertReplacementText" ||
    inputType === "insertFromComposition" ||
    inputType === "insertCompositionText"
  ) {
    return true;
  }
  return isMobileLike && inputType === "insertText" && (data?.length ?? 0) > 1;
}

export function updatePtyInputLine(currentLine: string, data: string): string {
  // Escape sequences (arrow keys, home/end, function keys, paste guards)
  // move the cursor or edit the line in ways this flat tracker cannot
  // model — and the per-char loop below would append their printable
  // payload (e.g. the "[D" of a left-arrow) as if it were typed text.
  // Reset instead: an unknown cursor position must disarm replacement
  // normalization until the user starts a fresh, cleanly-tracked line.
  if (data.includes("\x1b")) {
    return "";
  }
  let next = currentLine;
  for (const ch of chars(data)) {
    if (ch === "\r" || ch === "\n") {
      next = "";
    } else if (ch === DELETE || ch === "\b") {
      next = removeLastChar(next);
    } else if (ch === "\x15") {
      next = "";
    } else if (isPlainText(ch)) {
      next += ch;
    }
  }
  return next;
}

export function normalizePtyMobileInput(
  data: string,
  currentLine: string,
  replacementActive: boolean,
): { data: string; nextLine: string; normalized: boolean } {
  if (replacementActive && isPlainText(data)) {
    const replacementLine = replacementLineForMobileInput(currentLine, data);
    if (replacementLine !== null) {
      return {
        data: DELETE.repeat(chars(currentLine).length) + replacementLine,
        nextLine: replacementLine,
        normalized: true,
      };
    }
  }

  return {
    data,
    nextLine: updatePtyInputLine(currentLine, data),
    normalized: false,
  };
}
