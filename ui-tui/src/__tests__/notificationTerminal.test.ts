import { describe, expect, it } from 'vitest'

import { notificationTerminalForEnv } from '../app/notificationTerminal.js'

describe('notificationTerminalForEnv', () => {
  it.each([
    [{ TERM: 'xterm-ghostty' }, 'ghostty'],
    [{ TERM: 'tmux-256color', TERM_PROGRAM: 'ghostty', TMUX: '/tmp/tmux' }, 'ghostty'],
    [{ GHOSTTY_RESOURCES_DIR: '/Applications/Ghostty.app/Contents/Resources', TERM: 'tmux-256color', TMUX: '/tmp/tmux' }, 'ghostty'],
    [{ KITTY_WINDOW_ID: '1', TERM: 'tmux-256color', TMUX: '/tmp/tmux' }, 'kitty'],
    [{ LC_TERMINAL: 'iTerm2', TERM: 'screen-256color', STY: '1234' }, 'iterm2'],
    [{ TERM_PROGRAM: 'iTerm.app' }, 'iterm2'],
    [{ TERM_PROGRAM: 'WezTerm' }, 'iterm2'],
    [{ TERM: 'screen-256color', STY: '1234', WEZTERM_PANE: '0' }, 'iterm2'],
    [{ TERM: 'xterm-256color' }, null]
  ] as const)('detects terminal from environment %o', (env, expected) => {
    expect(notificationTerminalForEnv(env)).toBe(expected)
  })
})
