/**
 * PTY resume output sanitizer — strips pathological ANSI sequences that
 * Ink's two-pass virtual scroll emits during session resume.
 *
 * The PTY is delivered as individual WebSocket frames. A CSI escape
 * sequence may be split across frames, and UTF-8 multi-byte chars may
 * span binary frames. This helper buffers incomplete payloads so the
 * regex transformers always operate on safely-completed input.
 */

const BLANK_LINE_BURST = /\n{50,}/g;
// eslint-disable-next-line no-control-regex -- intentional ESC byte in ANSI sequence parser
const ERASE_LINE = /\x1b\[\d*K/g;
// eslint-disable-next-line no-control-regex
const ERASE_CHAR = /\x1b\[\d*X/g;

/** Regex for a still-incomplete trailing escape: "\x1b", "\x1b[", "\x1b[\d*". */
// eslint-disable-next-line no-control-regex
const PARTIAL_ESC = /^\x1b(?:\[\d*)?$/;

/**
 * Apply all suppression rules to a safely-completed string.
 * Exported for focused unit-testing of filter behaviour.
 */
export function applyPtyFilters(input: string): string {
  return input
    .replace(BLANK_LINE_BURST, "\n\n")
    .replace(ERASE_LINE, "")
    .replace(ERASE_CHAR, "");
}

/** Stateful chunk processor that guards against cross-frame split sequences. */
export class PtyResumeSanitizer {
  #pending = "";

  /** Feed one decoded WebSocket frame payload. Returns the sanitized output. */
  next(chunk: string): string {
    const combined = this.#pending + chunk;
    const lastEsc = combined.lastIndexOf("\x1b");
    if (lastEsc === -1) {
      this.#pending = "";
      return applyPtyFilters(combined);
    }
    const tail = combined.slice(lastEsc);
    if (PARTIAL_ESC.test(tail)) {
      this.#pending = tail;
      return applyPtyFilters(combined.slice(0, lastEsc));
    }
    this.#pending = "";
    return applyPtyFilters(combined);
  }

  /** Drain and sanitize any remaining buffered partial escape. */
  flush(): string {
    const last = this.#pending;
    this.#pending = "";
    return applyPtyFilters(last);
  }
}
