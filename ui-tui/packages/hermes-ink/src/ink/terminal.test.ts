import { describe, expect, it } from 'vitest'

import { isSynchronizedOutputSupported, needsAltScreenResizeScrollbackClear } from './terminal.js'

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
