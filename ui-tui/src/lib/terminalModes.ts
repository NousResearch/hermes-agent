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

// OSC 11 sets the terminal's DEFAULT background — so the whole TUI, not just
// rendered text, takes the skin color. OSC 111 restores the terminal's own
// default. We only reset when we actually painted, so a user who never uses a
// skin background keeps their terminal untouched.
const HEX_RE = /^#[0-9a-f]{6}$/i
const OSC_RESET_BACKGROUND = '\x1b]111\x07'
let _backgroundPainted = false

/**
 * Paint the terminal's default background from a skin (`hex`), or clear it back
 * to the terminal default when `hex` is empty/invalid (a skin with no
 * `background`, e.g. reverting to `default`). Runtime writes go through the async
 * stream so they order cleanly with Ink's frames; the exit-time restore rides
 * `resetTerminalModes` (writeSync). No-op off a TTY.
 */
export function setTerminalBackground(hex: string, stream: ResettableStream = process.stdout): void {
  if (!stream.isTTY) {
    return
  }

  if (HEX_RE.test(hex)) {
    try {
      stream.write(`\x1b]11;${hex}\x07`)
      _backgroundPainted = true
    } catch {
      // Terminal that can't take it just keeps its background.
    }
  } else if (_backgroundPainted) {
    try {
      stream.write(OSC_RESET_BACKGROUND)
      _backgroundPainted = false
    } catch {
      // ignore
    }
  }
}

export function resetTerminalModes(stream: ResettableStream = process.stdout): boolean {
  if (!stream.isTTY) {
    return false
  }

  // Append the background restore only if we painted one, so a normal session
  // never resets a terminal it didn't touch.
  const reset = _backgroundPainted ? TERMINAL_MODE_RESET + OSC_RESET_BACKGROUND : TERMINAL_MODE_RESET
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
