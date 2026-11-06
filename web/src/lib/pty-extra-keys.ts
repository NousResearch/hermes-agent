/**
 * Mobile extra-keys support for the dashboard chat PTY.
 *
 * Soft keyboards have no Esc / Tab / Ctrl / Alt / arrow keys, so the chat
 * renders an on-screen key bar. Plain keys send a fixed byte sequence; Ctrl
 * and Alt are sticky modifiers that transform the NEXT typed input.
 */

export type ExtraKey =
  | "esc"
  | "tab"
  | "shift-tab"
  | "up"
  | "down"
  | "left"
  | "right"
  | "home"
  | "end"
  | "pgup"
  | "pgdn"
  | "ctrl-c"
  | "ctrl-d"
  | "enter";

export const EXTRA_KEY_SEQUENCES: Record<ExtraKey, string> = {
  esc: "\x1b",
  tab: "\t",
  "shift-tab": "\x1b[Z",
  up: "\x1b[A",
  down: "\x1b[B",
  right: "\x1b[C",
  left: "\x1b[D",
  home: "\x1b[H",
  end: "\x1b[F",
  pgup: "\x1b[5~",
  pgdn: "\x1b[6~",
  "ctrl-c": "\x03",
  "ctrl-d": "\x04",
  enter: "\r",
};

// eslint-disable-next-line no-control-regex -- matching terminal ESC sequences
const ARROW_RE = /^\x1b\[[ABCD]$/;

export interface StickyModifiers {
  ctrl: boolean;
  alt: boolean;
}

/** Map one printable character to its Ctrl-chord byte, or null if none. */
export function ctrlByteFor(ch: string): string | null {
  if (ch.length !== 1) return null;
  const code = ch.toUpperCase().charCodeAt(0);
  // @ A-Z [ \ ] ^ _  → 0x00..0x1f
  if (code >= 0x40 && code <= 0x5f) return String.fromCharCode(code - 0x40);
  if (ch === " " || ch === "2") return "\x00";
  if (ch === "?") return "\x7f";
  if (ch === "/") return "\x1f";
  return null;
}

/**
 * Apply sticky modifiers to outgoing PTY data. Only a single typed character
 * (or a single escape sequence for arrows etc.) is transformed; the caller
 * clears the latch afterwards.
 */
export function applyStickyModifiers(
  data: string,
  mods: StickyModifiers,
): { data: string; consumed: boolean } {
  if (!mods.ctrl && !mods.alt) return { data, consumed: false };
  let out = data;
  if (mods.ctrl) {
    const chars = Array.from(data);
    if (chars.length === 1) {
      const b = ctrlByteFor(chars[0]);
      if (b !== null) out = b;
    } else if (ARROW_RE.test(data)) {
      // Ctrl+arrow → xterm "1;5" modified cursor keys (word motion).
      out = `\x1b[1;5${data[2]}`;
    }
  }
  if (mods.alt && !out.startsWith("\x1b")) {
    out = `\x1b${out}`;
  } else if (mods.alt && ARROW_RE.test(out)) {
    out = `\x1b[1;3${out[2]}`;
  }
  return { data: out, consumed: true };
}

/**
 * Plain-text snapshot of the terminal buffer (scrollback + screen) for the
 * mobile "select text" overlay. Trailing blank lines are trimmed; wrapped
 * rows are joined back into their logical line.
 */
export function bufferToPlainText(buffer: {
  length: number;
  getLine(
    y: number,
  ): { isWrapped: boolean; translateToString(trimRight?: boolean): string } | undefined;
}): string {
  const lines: string[] = [];
  for (let y = 0; y < buffer.length; y++) {
    const line = buffer.getLine(y);
    if (!line) continue;
    const text = line.translateToString(true);
    if (line.isWrapped && lines.length) {
      lines[lines.length - 1] += text;
    } else {
      lines.push(text);
    }
  }
  while (lines.length && lines[lines.length - 1].trim() === "") lines.pop();
  return lines.join("\n");
}
