const DELETE = "\x7f";
export const CURSOR_LEFT = "\x1b[D";
export const CURSOR_RIGHT = "\x1b[C";

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

 
// Caret positions arrive as UTF-16 offsets; the composer edits whole graphemes,
// so every count below is in graphemes.
export function caretGraphemeIndex(segments: string[], codeUnits: number): number {
  let acc = 0;
  for (let i = 0; i < segments.length; i++) {
    if (acc >= codeUnits) return i;
    acc += segments[i].length;
  }
  return segments.length;
}

// Bytes that replay one textarea edit onto the PTY line from a composer whose
// cursor sits at `caretBefore` (a grapheme index into `before`), landing the
// cursor at `caretAfter`. DEL deletes before the cursor, so a caret edit has
// to walk onto the changed region, delete it, re-insert what the caret edit
// changed, and walk back to where the user sees the caret. With the caret at
// the end of the line the walks are zero and this degenerates to xterm's flat
// keydown-229 diff: one DEL per removed grapheme, then the inserted tail.
// Caret at the line ends: the encoder degenerates to xterm's flat keydown-229
// diff (one DEL per removed grapheme, then the inserted tail) unless a retained
// tail after the caret forces cursor arrows, which the flat tracker disarms.
export function textareaEditBytes(before: string, after: string): string {
  return caretEditBytes(before, before.length, after, after.length);
}

export function caretEditBytes(
  before: string,
  caretBefore: number,
  after: string,
  caretAfter: number,
): string {
  const prev = graphemes(before);
  const next = graphemes(after);
  const prevCaret = caretGraphemeIndex(prev, caretBefore);
  const nextCaret = caretGraphemeIndex(next, caretAfter);
  let common = 0;
  while (common < prev.length && common < next.length && prev[common] === next[common]) {
    common++;
  }
  const bytes: string[] = [];
  const walk = (from: number, to: number) => {
    if (to > from) bytes.push(CURSOR_RIGHT.repeat(to - from));
    else if (to < from) bytes.push(CURSOR_LEFT.repeat(from - to));
  };
  let suffix = 0;
  while (
    common + suffix < prev.length &&
    common + suffix < next.length &&
    prev[prev.length - 1 - suffix] === next[next.length - 1 - suffix]
  ) {
    suffix++;
  }
  const region = prev.length - common - suffix;
  const inserted = next.length - common - suffix;
  if (region === 0 && inserted === 0) {
    // Pure caret move: the arrows are the whole edit, and this is the only
    // way a caret move becomes representable in the byte stream at all.
    walk(prevCaret, nextCaret);
    return bytes.join("");
  }
  // DEL deletes before the cursor, so the cursor has to sit at the end of the
  // changed region. Walk there, delete the region, insert the replacement at
  // the cursor (the retained tail is still ahead of it), then walk back to
  // where the user sees the caret. With the caret at the line end and no
  // common suffix, the walks are zero and this is xterm's flat keydown-229
  // diff: one DEL per removed grapheme, then the inserted tail.
  walk(prevCaret, prev.length - suffix);
  bytes.push(DELETE.repeat(region));
  bytes.push(next.slice(common, common + inserted).join(""));
  walk(common + inserted, nextCaret);
  return bytes.join("");
}

export function isPlainTextData(data: string): boolean {
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

// Some Android keyboards (Gboard on Chrome) deliver plain taps as
// beforeinput/input events only — no keydown and no composition events —
// so neither xterm's onData nor the compositionend fallback ever fires and
// every tap is silently dropped (#137986). Those commits must be forwarded
// through the composition forwarder, whose echo window drops the copies on
// devices where xterm did emit onData. insertCompositionText is excluded on
// purpose: on composition-capable keyboards it fires repeatedly with the
// uncommitted buffer, and only the terminal commit (compositionend /
// insertFromComposition) is authoritative there.
export function mobileBeforeInputCommit(
  inputType: string | undefined,
  data: string | null | undefined,
  isMobileLike: boolean,
): string | null {
  if (!isMobileLike) return null;
  // Browsers report insertLineBreak with a null payload (the line break is
  // the whole event), so Enter cannot be gated on the data check below;
  // xterm sends "\r" for Enter.
  if (inputType === "insertLineBreak") return "\r";
  if (inputType !== "insertText") return null;
  if (typeof data !== "string" || data.length === 0) return null;
  return data;
}

// Applies terminal input bytes to a tracked line. `appendText` is false for
// xterm's hidden textarea, which already holds the text the IME typed and only
// needs the edits xterm sent on its own (DEL, line boundaries).
// `caret` is the grapheme index a composer's cursor sits at. With a caret the
// model is caret-aware: cursor arrows move it, DEL deletes the grapheme before
// it, and plain text inserts at it. Without one the model is flat — every
// escape sequence, including the cursor arrows, is a cursor move it cannot
// model, and appending their printable payload (the "[D" of a left arrow) as
// typed text is what the naive tracker used to do. Such a sequence disarms the
// tracker instead: an unknown cursor position must disarm replacement
// normalization until the user starts a fresh, cleanly-tracked line.
export function applyPtyLineEdits(
  currentLine: string,
  data: string,
  appendText: boolean,
  caret: number | null = null,
): string {
  let next = currentLine;
  let pos = caret;
  let i = 0;
  while (i < data.length) {
    const ch = data[i];
    if (ch === "\r" || ch === "\n" || ch === "\x03" || ch === "\x15") {
      next = "";
      pos = caret === null ? null : 0;
      i += 1;
    } else if (ch === DELETE || ch === "\b") {
      if (pos === null) {
        next = removeLastGrapheme(next);
      } else if (pos > 0) {
        next = graphemes(next).slice(0, pos - 1).concat(graphemes(next).slice(pos)).join("");
        pos -= 1;
      }
      i += 1;
    } else if (ch === "\x1b" && data.startsWith(CURSOR_LEFT, i)) {
      // A cursor arrow is one edit unit — CSI plus the direction letter — so it
      // has to be read as a unit rather than as a lone escape byte. Without a
      // caret the flat tracker cannot model a cursor move and disarms.
      if (pos === null) return "";
      pos = Math.max(0, pos - 1);
      i += 3;
    } else if (ch === "\x1b" && data.startsWith(CURSOR_RIGHT, i)) {
      if (pos === null) return "";
      pos += 1;
      i += 3;
    } else if (ch === "\x1b") {
      return "";
    } else if (appendText && isPlainTextData(ch)) {
      if (pos === null) next += ch;
      else {
        const segs = graphemes(next);
        next = segs.slice(0, pos).concat([ch], segs.slice(pos)).join("");
        pos += 1;
      }
      i += 1;
    } else {
      i += 1;
    }
  }
  return next;
}

export function updatePtyInputLine(currentLine: string, data: string): string {
  return applyPtyLineEdits(currentLine, data, true, null);
}

export function normalizePtyMobileInput(
  data: string,
  currentLine: string,
  replacementActive: boolean,
): { data: string; nextLine: string; normalized: boolean } {
  if (replacementActive && isPlainTextData(data)) {
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
