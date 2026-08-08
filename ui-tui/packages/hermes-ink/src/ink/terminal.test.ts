import { Writable } from 'node:stream'

import { describe, expect, it } from 'vitest'

import type { Diff } from './frame.js'
import { isSynchronizedOutputSupported, needsAltScreenResizeScrollbackClear, writeDiffToTerminal } from './terminal.js'
import { BSU, ESU } from './termio/dec.js'

describe('terminal resize quirks', () => {
  it('uses a deeper alt-screen resize clear for Apple Terminal', () => {
    expect(needsAltScreenResizeScrollbackClear({ TERM_PROGRAM: 'Apple_Terminal' })).toBe(true)
    expect(needsAltScreenResizeScrollbackClear({ TERM_PROGRAM: ' Apple_Terminal ' })).toBe(true)
  })

  it('keeps the normal resize repaint path for modern terminals', () => {
    expect(needsAltScreenResizeScrollbackClear({ TERM_PROGRAM: 'vscode' })).toBe(false)
    expect(needsAltScreenResizeScrollbackClear({ TERM_PROGRAM: 'iTerm.app' })).toBe(false)
  })
})

describe('synchronized output detection', () => {
  it('does not trust an outer terminal DEC 2026 capability under a multiplexer', () => {
    // Zellij sets ZELLIJ to the session index — "0" for the first session — so
    // the guard keys on presence, not truthiness of the value.
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm', ZELLIJ: '0' })).toBe(false)
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm', ZELLIJ: '1' })).toBe(false)
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm', TMUX: '/tmp/tmux-1/default,1,0' })).toBe(false)
    expect(isSynchronizedOutputSupported({ STY: '4242.pts-0.host', TERM_PROGRAM: 'WezTerm' })).toBe(false)
  })

  it('still reports support for a DEC 2026 terminal with no multiplexer', () => {
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm' })).toBe(true)
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'iTerm.app' })).toBe(true)
    expect(isSynchronizedOutputSupported({ TERM: 'xterm-kitty' })).toBe(true)
  })

  it('reports no support for an unknown terminal', () => {
    expect(isSynchronizedOutputSupported({ TERM: 'xterm-256color' })).toBe(false)
  })

  it('reads the injected env only, never the ambient process env', () => {
    // Guards the injection itself: every branch must consult `env`, so an
    // empty record is "unknown terminal" regardless of the host environment.
    expect(isSynchronizedOutputSupported({})).toBe(false)
  })
})

/** Captures everything written to a Terminal's stdout so a frame can be asserted on. */
const captureTerminal = () => {
  const chunks: string[] = []

  const sink = () =>
    new Writable({
      write(chunk: Buffer | string, _encoding, callback) {
        chunks.push(String(chunk))
        callback()
      }
    })

  return { frame: () => chunks.join(''), terminal: { stderr: sink(), stdout: sink() } }
}

describe('writeDiffToTerminal synchronized-output markers', () => {
  const diff: Diff = [{ content: 'hello', type: 'stdout' }]

  it('omits BSU/ESU from the emitted frame when markers are skipped', () => {
    const { frame, terminal } = captureTerminal()

    writeDiffToTerminal(terminal, diff, true)

    const emitted = frame()

    expect(emitted).toContain('hello')
    expect(emitted).not.toContain(BSU)
    expect(emitted).not.toContain(ESU)
  })

  it('wraps the emitted frame in BSU/ESU when the terminal supports DEC 2026', () => {
    const { frame, terminal } = captureTerminal()

    writeDiffToTerminal(terminal, diff, false)

    const emitted = frame()

    expect(emitted.startsWith(BSU)).toBe(true)
    expect(emitted.endsWith(ESU)).toBe(true)
    expect(emitted).toContain('hello')
  })

  it('emits no markers on a main-screen frame under a multiplexer', () => {
    // The reported bug (#66490): the main-screen renderer passed
    // `altScreenActive && !supported`, which is always false off the alt
    // screen, so BSU/ESU reached Zellij on every frame. The renderer now gates
    // on the capability alone; both flags are computed here exactly as
    // ink.tsx computes them, so the emitted frames show the regression.
    const altScreenActive = false
    const syncSupported = isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm', ZELLIJ: '0' })

    expect(syncSupported).toBe(false)

    const fixed = captureTerminal()

    writeDiffToTerminal(fixed.terminal, diff, !syncSupported)

    expect(fixed.frame()).toBe('hello')

    // Counterexample: the previous gate on this same main-screen frame.
    const previous = captureTerminal()

    writeDiffToTerminal(previous.terminal, diff, altScreenActive && !syncSupported)

    expect(previous.frame()).toBe(`${BSU}hello${ESU}`)
  })
})
