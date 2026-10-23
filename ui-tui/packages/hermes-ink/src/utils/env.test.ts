import { describe, expect, it } from 'vitest'

import { detectTerminal } from './env.js'

describe('terminal detection', () => {
  it('treats a Herdr pane as an extended-key multiplexer', () => {
    expect(detectTerminal({ HERDR_ENV: '1', TERM: 'xterm-256color' })).toBe('tmux')
  })

  it('keeps existing direct terminal detection', () => {
    expect(detectTerminal({ TERM: 'xterm-kitty' })).toBe('kitty')
    expect(detectTerminal({ TERM_PROGRAM: 'WezTerm' })).toBe('WezTerm')
  })
})
