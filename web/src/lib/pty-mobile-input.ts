const DELETE = "\x7f";

// How long (ms) after a mobile IME / replacement event we treat subsequent
// terminal input as a candidate line-replacement rather than a plain append.
// Exported so the ChatPage integration and tests share one tunable value.
export const MOBILE_REPLACEMENT_WINDOW_MS = 350;

// Created on first use, not at import: Firefox < 125 has no Intl.Segmenter,
// and a throw while importing ChatPage takes the whole dashboard down.
let graphemeSegmenter: Intl.Segmenter | null | undefined;

// The PTY composer deletes one grapheme per DEL, so every count of DELs and
// every tracked-line deletion is in graphemes. Without Intl.Segmenter,
// fall back to code points.
export function graphemes(text: string): string[] {
  if (graphemeSegmenter === undefined) {
    graphemeSegmenter =
      typeof Intl.Segmenter === "function"
        ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
        : null;
  }
  if (!graphemeSegmenter) {
    return Array.from(text);
  }
  return Array.from(graphemeSegmenter.segment(text), ({ segment }) => segment);
}

function removeLastGrapheme(text: string): string {
  return graphemes(text).slice(0, -1).join("");
}

 
function isPlainText(data: string): boolean {
  // eslint-disable-next-line no-control-regex -- terminal data may contain control chars
  return !/[\x00-\x1f\x7f]/.test(data);
}

function lastWordMatch(line: string): RegExpMatchArray | null {
  return line.match(/^(.*?)(\S+)(\s*)$/u);
}

function collapseDuplicatedFinalWord(text: string, previousLine: string): string {
  const match = text.match(/^(.*?)(\S+)(\s+)(\S+)(\s*)$/u);
  if (!match) return text;

  const [, prefix, first, , second, trailing] = match;
  if (first.toLocaleLowerCase() !== second.toLocaleLowerCase()) return text;
  // Only collapse a duplication the tracked line already ended with — i.e.
  // Gboard re-emitted the final word. Requiring a >=2-char word avoids
  // eating legitimate single-letter reduplication ("a a", "i i") that a
  // user may genuinely type inside the replacement window.
  if (first.length < 2) return text;
  if (!previousLine.trimEnd().toLocaleLowerCase().endsWith(first.toLocaleLowerCase())) {
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

// Applies terminal input bytes to a tracked line. `appendText` is false for
// xterm's hidden textarea, which already holds the text the IME typed and only
// needs the edits xterm sent on its own (DEL, line boundaries).
export function applyPtyLineEdits(
  currentLine: string,
  data: string,
  appendText: boolean,
): string {
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
  // Code points, not graphemes: "\r\n" is one grapheme but two line events.
  for (const ch of Array.from(data)) {
    if (ch === "\r" || ch === "\n" || ch === "\x03" || ch === "\x15") {
      next = "";
    } else if (ch === DELETE || ch === "\b") {
      next = removeLastGrapheme(next);
    } else if (appendText && isPlainText(ch)) {
      next += ch;
    }
  }
  return next;
}

export function updatePtyInputLine(currentLine: string, data: string): string {
  return applyPtyLineEdits(currentLine, data, true);
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
        data: DELETE.repeat(graphemes(currentLine).length) + replacementLine,
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
