import { describe, expect, it } from 'vitest'

import { detectTerminalGraphics } from './terminalGraphics.js'

describe('detectTerminalGraphics', () => {
  it('picks kitty placeholders only for terminals that paint them', () => {
    expect(detectTerminalGraphics({ KITTY_WINDOW_ID: '1', TERM: 'xterm-kitty' })).toBe('kitty')
    expect(detectTerminalGraphics({ TERM: 'xterm-ghostty' })).toBe('kitty')
    expect(detectTerminalGraphics({ TERM_PROGRAM: 'ghostty' })).toBe('kitty')
    expect(detectTerminalGraphics({ TERM: 'xterm-256color', TERM_PROGRAM: 'iTerm.app' })).toBe('unicode')
    expect(detectTerminalGraphics({ TERM_PROGRAM: 'WezTerm', TERM: 'xterm-kitty' })).toBe('unicode')
  })

  it('falls back to half-blocks where the kitty transmit is swallowed or leaked env lies', () => {
    const kitty = { KITTY_WINDOW_ID: '1', TERM: 'xterm-kitty' }

    expect(detectTerminalGraphics({ ...kitty, TMUX: '/tmp/tmux-1000/default,1,0' })).toBe('unicode')
    expect(detectTerminalGraphics({ ...kitty, STY: '1234.pts-0.host' })).toBe('unicode')
    expect(detectTerminalGraphics({ ...kitty, TERM_PROGRAM: 'vscode' })).toBe('unicode')
    expect(detectTerminalGraphics({ ...kitty, HERMES_PTY_HOST: 'dashboard' })).toBe('unicode')
  })
})
