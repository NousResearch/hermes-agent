import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  isPaintableHex,
  resetTerminalModes,
  setTerminalBackground,
  setTerminalForeground,
  TERMINAL_MODE_RESET
} from '../lib/terminalModes.js'

describe('terminal mode reset', () => {
  it('includes common sticky input modes', () => {
    expect(TERMINAL_MODE_RESET).toContain("\x1b[0'z")
    expect(TERMINAL_MODE_RESET).toContain("\x1b[0'{")
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

// Foreground (OSC 10) and background (OSC 11) are the same slot contract —
// assert it once over both. Painting BOTH is what keeps every default-fg
// token legible when a skin flips the terminal's polarity.
describe.each([
  { name: 'foreground', osc: 10, set: setTerminalForeground },
  { name: 'background', osc: 11, set: setTerminalBackground }
])('terminal default $name (OSC $osc)', ({ osc, set }) => {
  const paint = `\x1b]${osc};`
  const restore = `\x1b]1${osc}\x07`
  const tty = (write: ReturnType<typeof vi.fn>) => ({ isTTY: true, write }) as unknown as NodeJS.WriteStream

  const written = (fn: (s: NodeJS.WriteStream) => void): string => {
    const write = vi.fn()
    fn(tty(write))

    return (write.mock.calls[0]?.[0] as string) ?? ''
  }

  // Leave the module's "painted" flag clean so the exact-match reset test above
  // (and other files) never see a stray restore.
  afterEach(() => set('', tty(vi.fn())))

  it('paints the terminal default from a valid hex', () => {
    expect(written(s => set('#08201F', s))).toBe(`${paint}#08201F\x07`)
  })

  it('ignores an invalid hex and non-TTY streams', () => {
    expect(written(s => set('teal', s))).toBe('')

    const write = vi.fn()
    set('#08201f', { isTTY: false, write } as unknown as NodeJS.WriteStream)
    expect(write).not.toHaveBeenCalled()
  })

  it('appends the restore to the exit reset once painted, not before', () => {
    expect(written(resetTerminalModes)).not.toContain(restore)

    set('#101010', tty(vi.fn()))
    expect(written(resetTerminalModes)).toContain(restore)
  })

  it('clears back to the terminal default when the next skin drops the color', () => {
    set('#123456', tty(vi.fn()))
    expect(written(s => set('', s))).toBe(restore)
    // Cleared: a later reset no longer restores.
    expect(written(resetTerminalModes)).not.toContain(restore)
  })
})

describe('isPaintableHex', () => {
  it('matches exactly what the slot setters paint', () => {
    expect(isPaintableHex('#08201F')).toBe(true)
    expect(isPaintableHex('#08201f')).toBe(true)

    for (const junk of ['', 'teal', '#fff', '#12345', '#1234567']) {
      expect(isPaintableHex(junk)).toBe(false)
    }
  })
})
