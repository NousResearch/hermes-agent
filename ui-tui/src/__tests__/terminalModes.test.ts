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

  it('saves cursor position before exiting alternate screen', () => {
    const saveIdx = TERMINAL_MODE_RESET.indexOf('\x1b[s')
    const exitAltIdx = TERMINAL_MODE_RESET.indexOf('\x1b[?1049l')

    expect(saveIdx).toBeGreaterThanOrEqual(0)
    expect(exitAltIdx).toBeGreaterThanOrEqual(0)
    expect(saveIdx).toBeLessThan(exitAltIdx)
  })

  it('restores cursor position after exiting alternate screen', () => {
    const exitAltIdx = TERMINAL_MODE_RESET.indexOf('\x1b[?1049l')
    const restoreIdx = TERMINAL_MODE_RESET.indexOf('\x1b[u')

    expect(exitAltIdx).toBeGreaterThanOrEqual(0)
    expect(restoreIdx).toBeGreaterThanOrEqual(0)
    expect(restoreIdx).toBeGreaterThan(exitAltIdx)
  })

  it('brackets alt-screen exit with immediate save/restore (no intervening sequences)', () => {
    const saveIdx = TERMINAL_MODE_RESET.indexOf('\x1b[s')
    const exitAltIdx = TERMINAL_MODE_RESET.indexOf('\x1b[?1049l')
    const restoreIdx = TERMINAL_MODE_RESET.indexOf('\x1b[u')
    const exitLen = '\x1b[?1049l'.length

    // save is immediately before ?1049l
    expect(TERMINAL_MODE_RESET.slice(saveIdx + '\x1b[s'.length, exitAltIdx).trim()).toBe('')
    // restore is immediately after ?1049l
    expect(TERMINAL_MODE_RESET.slice(exitAltIdx + exitLen, restoreIdx).trim()).toBe('')
  })
})
