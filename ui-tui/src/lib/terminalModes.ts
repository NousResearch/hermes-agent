import { writeSync } from 'node:fs'

export const TERMINAL_MODE_RESET =
  "\x1b[0'z" + // DEC locator reporting
  "\x1b[0'{" + // selectable locator events
  '\x1b[?2029l' + // passive mouse
  '\x1b[?1016l' + // SGR-pixels mouse
  '\x1b[?1015l' + // urxvt decimal mouse
  '\x1b[?1006l' + // SGR mouse
  '\x1b[?1005l' + // UTF-8 extended mouse
  '\x1b[?1003l' + // any-motion mouse
  '\x1b[?1002l' + // button-motion mouse
  '\x1b[?1001l' + // highlight mouse
  '\x1b[?1000l' + // click mouse
  '\x1b[?9l' + // X10 mouse
  '\x1b[?1004l' + // focus events
  '\x1b[?2004l' + // bracketed paste
  '\x1b[?1049l' + // alternate screen
  '\x1b[<u' + // kitty keyboard
  '\x1b[>4m' + // modifyOtherKeys
  '\x1b[0m' + // attributes
  '\x1b[?25h' // cursor visible

type ResettableStream = Pick<NodeJS.WriteStream, 'isTTY' | 'write'> & {
  fd?: number
}

// OSC 10/11 set the terminal's DEFAULT foreground/background — so every cell,
// including text rendered with no explicit color (markdown body, borders,
// third-party output), takes the skin instead of the host profile's defaults.
// OSC 110/111 restore the terminal's own values. We only reset what we
// actually painted, so a skinless session leaves the terminal untouched.
const HEX_RE = /^#[0-9a-f]{6}$/i

/** True when `hex` is a paintable default (the same bar `set` applies). */
export const isPaintableHex = (hex: string): boolean => HEX_RE.test(hex)

/**
 * A paintable terminal default (fg=10, bg=11). `set(hex)` paints from a skin,
 * or clears back to the terminal's own default when `hex` is empty/invalid
 * (a skin without the key, e.g. reverting to `default`). Runtime writes go
 * through the async stream so they order cleanly with Ink's frames; the
 * exit-time restore rides `resetTerminalModes` (writeSync). No-op off a TTY.
 */
const defaultColorSlot = (osc: 10 | 11) => {
  const restore = `\x1b]1${osc}\x07`
  let painted = false

  const set = (hex: string, stream: ResettableStream = process.stdout): void => {
    if (!stream.isTTY) {
      return
    }

    try {
      if (HEX_RE.test(hex)) {
        stream.write(`\x1b]${osc};${hex}\x07`)
        painted = true
      } else if (painted) {
        stream.write(restore)
        painted = false
      }
    } catch {
      // Terminal that can't take it just keeps its default.
    }
  }

  return { restoreSeq: () => (painted ? restore : ''), set }
}

const foreground = defaultColorSlot(10)
const background = defaultColorSlot(11)

export const setTerminalForeground = foreground.set
export const setTerminalBackground = background.set

export function resetTerminalModes(stream: ResettableStream = process.stdout): boolean {
  if (!stream.isTTY) {
    return false
  }

  // Append default-color restores only for what we painted, so a normal
  // session never resets a terminal it didn't touch.
  const reset = TERMINAL_MODE_RESET + foreground.restoreSeq() + background.restoreSeq()
  const fd = typeof stream.fd === 'number' ? stream.fd : stream === process.stdout ? 1 : undefined

  if (fd !== undefined) {
    try {
      writeSync(fd, reset)

      return true
    } catch {
      // Fall through to stream.write for mocked or unusual TTY streams.
    }
  }

  try {
    stream.write(reset)

    return true
  } catch {
    return false
  }
}
