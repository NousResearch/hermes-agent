import { describe, expect, it, vi } from 'vitest'

import { resetTerminalModes, TERMINAL_MODE_RESET } from '../lib/terminalModes.js'

describe('terminal mode reset', () => {
  it('includes common sticky input modes', () => {
    expect(TERMINAL_MODE_RESET).toContain('\x1b[0\'z')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[0\'{')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?2029l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1016l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1015l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1006l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1005l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1003l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1002l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1001l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1000l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?9l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1004l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?2004l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1049l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[<u')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[>4m')
  })

  it('writes reset sequence to TTY streams without fds', () => {
    const write = vi.fn()

    expect(resetTerminalModes({ isTTY: true, write } as unknown as NodeJS.WriteStream)).toBe(true)
    expect(write).toHaveBeenCalledWith(TERMINAL_MODE_RESET)
  })

  it('skips non-TTY streams', () => {
    const write = vi.fn()

    expect(resetTerminalModes({ isTTY: false, write } as unknown as NodeJS.WriteStream)).toBe(false)
    expect(write).not.toHaveBeenCalled()
  })

  // entry.tsx installs `process.on('exit', () => resetTerminalModes())` as the
  // final backstop (#28419): /quit, Ctrl+C, Ctrl+D and any process.exit() path
  // must disarm DEC mouse tracking so the parent shell / next TUI doesn't read
  // leaked mouse reports as keystrokes. 'exit' handlers run synchronously only,
  // so the reset must complete via a single synchronous write — verify that an
  // exit-style invocation disables every SGR mouse mode that produced the
  // reported `…;…M` garbage.
  it('disarms mouse tracking from a synchronous exit-style handler', () => {
    const write = vi.fn()
    const stream = { isTTY: true, write } as unknown as NodeJS.WriteStream

    // Mirror entry.tsx's process.on('exit') callback.
    const onExit = () => resetTerminalModes(stream)
    onExit()

    expect(write).toHaveBeenCalledTimes(1)
    const written = write.mock.calls[0]?.[0] as string
    for (const mode of ['\x1b[?1006l', '\x1b[?1003l', '\x1b[?1002l', '\x1b[?1000l']) {
      expect(written).toContain(mode)
    }
  })
})
