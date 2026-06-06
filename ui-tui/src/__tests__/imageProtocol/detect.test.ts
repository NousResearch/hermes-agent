import { describe, it, expect, afterEach } from 'vitest'
import { terminalSupportsImages, preferredImageProtocol } from '../../lib/imageProtocol/detect.js'

describe('terminalSupportsImages', () => {
  const originalEnv = { ...process.env }

  afterEach(() => {
    process.env = { ...originalEnv }
  })

  it('returns true when TERM is xterm-kitty', () => {
    process.env.TERM = 'xterm-kitty'
    process.env.TERM_PROGRAM = ''
    expect(terminalSupportsImages()).toBe(true)
  })

  it('returns true when TERM_PROGRAM is iTerm.app', () => {
    process.env.TERM = 'xterm'
    process.env.TERM_PROGRAM = 'iTerm.app'
    expect(terminalSupportsImages()).toBe(true)
  })

  it('returns true when TERM contains mlterm (Sixel)', () => {
    process.env.TERM = 'mlterm'
    process.env.TERM_PROGRAM = ''
    expect(terminalSupportsImages()).toBe(true)
  })

  it('returns false for plain xterm', () => {
    process.env.TERM = 'xterm'
    process.env.TERM_PROGRAM = ''
    expect(terminalSupportsImages()).toBe(false)
  })
})

describe('preferredImageProtocol', () => {
  it('returns "kitty" when terminal is Kitty', () => {
    process.env.TERM = 'xterm-kitty'
    process.env.TERM_PROGRAM = ''
    expect(preferredImageProtocol()).toBe('kitty')
  })

  it('returns "iterm" when terminal is iTerm2', () => {
    process.env.TERM = 'xterm'
    process.env.TERM_PROGRAM = 'iTerm.app'
    expect(preferredImageProtocol()).toBe('iterm')
  })

  it('returns "sixel" when terminal is mlterm', () => {
    process.env.TERM = 'mlterm'
    process.env.TERM_PROGRAM = ''
    expect(preferredImageProtocol()).toBe('sixel')
  })

  it('returns null when terminal supports no protocol', () => {
    process.env.TERM = 'xterm'
    process.env.TERM_PROGRAM = ''
    expect(preferredImageProtocol()).toBeNull()
  })
})
