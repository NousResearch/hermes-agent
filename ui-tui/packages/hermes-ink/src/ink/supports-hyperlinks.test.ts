/**
 * Regression tests for Ghostty OSC 8 hyperlink support (issue #22895).
 *
 * Ghostty sets TERM=xterm-ghostty (not TERM_PROGRAM=ghostty), so the
 * supports-hyperlinks detector needs to check TERM in addition to
 * TERM_PROGRAM.
 */

import { supportsHyperlinks, ADDITIONAL_HYPERLINK_TERMINALS } from './supports-hyperlinks'

describe('supportsHyperlinks', () => {
  it('detects ghostty via TERM=xterm-ghostty', () => {
    const result = supportsHyperlinks({
      env: { TERM: 'xterm-ghostty' },
      stdoutSupported: false,
    })
    expect(result).toBe(true)
  })

  it('detects ghostty via TERM_PROGRAM=ghostty', () => {
    const result = supportsHyperlinks({
      env: { TERM_PROGRAM: 'ghostty' },
      stdoutSupported: false,
    })
    expect(result).toBe(true)
  })

  it('detects kitty via TERM=xterm-kitty', () => {
    const result = supportsHyperlinks({
      env: { TERM: 'xterm-kitty' },
      stdoutSupported: false,
    })
    expect(result).toBe(true)
  })

  it('does not detect generic xterm', () => {
    const result = supportsHyperlinks({
      env: { TERM: 'xterm-256color' },
      stdoutSupported: false,
    })
    expect(result).toBe(false)
  })

  it('returns true when stdoutSupported is true regardless of env', () => {
    const result = supportsHyperlinks({
      env: {},
      stdoutSupported: true,
    })
    expect(result).toBe(true)
  })

  it('detects iTerm via TERM_PROGRAM', () => {
    const result = supportsHyperlinks({
      env: { TERM_PROGRAM: 'iTerm.app' },
      stdoutSupported: false,
    })
    expect(result).toBe(true)
  })
})
