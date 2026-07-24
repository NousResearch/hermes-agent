/** One physical terminal row, as read from xterm's active buffer. */
export interface TranscriptRow {
  /** The row's text (typically `line.translateToString(true)`, trailing whitespace trimmed). */
  text: string;
  /** True when this physical row is a soft-wrap continuation of the row above it. */
  isWrapped: boolean;
}

/**
 * Reassemble xterm *physical* rows into *logical* lines.
 *
 * xterm stores each visual row separately, and a row that soft-wraps from the
 * one above has `isWrapped === true`. Joining every physical row with "\n"
 * would break a wrapped command, path, or URL across lines when copied — worst
 * on the narrow phones the touch-copy sheet exists for. A newline is inserted
 * only at a real line break: a row that is *not* a wrap continuation starts a
 * fresh logical line; a wrapped row is appended to the current one with no
 * separator. Trailing whitespace across the whole transcript is trimmed to
 * match the copied-buffer behaviour.
 */
export function reassembleWrappedLines(rows: TranscriptRow[]): string {
  const out: string[] = [];
  let logical = "";
  let started = false;
  for (const row of rows) {
    if (row.isWrapped && started) {
      logical += row.text;
    } else {
      if (started) out.push(logical);
      logical = row.text;
      started = true;
    }
  }
  if (started) out.push(logical);
  return out.join("\n").replace(/\s+$/, "");
}
