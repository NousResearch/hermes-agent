import { describe, expect, it } from 'vitest'

import { BSU, ESU } from './termio/dec.js'

import { isSynchronizedOutputSupported, needsAltScreenResizeScrollbackClear, writeDiffToTerminal } from './terminal.js'

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
  it('does not trust an outer terminal DEC 2026 capability under Zellij', () => {
    // Zellij (like tmux) proxies/chunks the stream, so the outer WezTerm's
    // DEC 2026 support must not be trusted. Zellij sets ZELLIJ to the session
    // index — "0" for the first session — so the guard keys on presence.
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm', ZELLIJ: '0' })).toBe(false)
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm', ZELLIJ: '1' })).toBe(false)
  })

  it('does not trust an outer terminal DEC 2026 capability under tmux', () => {
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm', TMUX: '/tmp/tmux-1/default,1,0' })).toBe(false)
  })

  it('still reports support for a DEC 2026 terminal with no multiplexer', () => {
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'WezTerm' })).toBe(true)
    expect(isSynchronizedOutputSupported({ TERM_PROGRAM: 'iTerm.app' })).toBe(true)
  })

  it('reports no support for an unknown terminal', () => {
    expect(isSynchronizedOutputSupported({ TERM: 'xterm-256color' })).toBe(false)
  })
})

describe('writeDiffToTerminal sync-marker gating (#66490 main-screen gap)', () => {
  const makeTerminal = () => {
    const writes: string[] = []
    const stdout = {
      write(chunk: string) {
        writes.push(String(chunk))
        return true
      }
    }
    return { terminal: { stderr: stdout, stdout } as never, writes }
  }

  it('wraps the frame in BSU/ESU when sync markers are enabled', () => {
    const { terminal, writes } = makeTerminal()
    writeDiffToTerminal(terminal, [{ content: 'frame', type: 'stdout' }] as never, false)
    const out = writes.join('')
    expect(out).toContain(BSU)
    expect(out).toContain(ESU)
  })

  it('emits no BSU/ESU when skipSyncMarkers is true (unsupported terminal, ANY screen)', () => {
    // The renderer passes !SYNC_OUTPUT_SUPPORTED for every write path — main
    // screen included. Under Zellij the multiplexer re-chunks the stream, so
    // the markers buy no atomicity and stale frames leak into main-screen
    // scrollback as repeated chrome.
    const { terminal, writes } = makeTerminal()
    writeDiffToTerminal(terminal, [{ content: 'frame', type: 'stdout' }] as never, true)
    const out = writes.join('')
    expect(out).not.toContain(BSU)
    expect(out).not.toContain(ESU)
    expect(out).toContain('frame')
  })
})
