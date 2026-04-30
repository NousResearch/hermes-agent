import { describe, expect, it, vi } from 'vitest'

import { resetTerminalModes, TERMINAL_MODE_RESET } from '../lib/terminalModes.js'

describe('terminal mode reset', () => {
  it('includes the sticky input modes Hermes enables', () => {
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1006l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1003l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1002l')
    expect(TERMINAL_MODE_RESET).toContain('\x1b[?1000l')
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
})
