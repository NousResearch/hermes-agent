import { writeSync } from 'node:fs'

export const TERMINAL_MODE_RESET =
  '\x1b[0\'z' + // DEC locator reporting
  '\x1b[0\'{' + // selectable locator events
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

export function resetTerminalModes(stream: ResettableStream = process.stdout): boolean {
  if (!stream.isTTY) {
    return false
  }

  const fd = typeof stream.fd === 'number' ? stream.fd : stream === process.stdout ? 1 : undefined
  if (fd !== undefined) {
    try {
      writeSync(fd, TERMINAL_MODE_RESET)

      return true
    } catch {
      // Fall through to stream.write for mocked or unusual TTY streams.
    }
  }

  try {
    stream.write(TERMINAL_MODE_RESET)

    return true
  } catch {
    return false
  }
}
