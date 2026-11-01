import { describe, expect, it } from 'vitest'

import { notificationTerminalForEnv, warpAgentPayload } from '../app/notificationTerminal.js'

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
    [{ TERM: 'xterm-256color' }, null],
    [
      {
        TERM_PROGRAM: 'WarpTerminal',
        WARP_CLI_AGENT_PROTOCOL_VERSION: '1',
        WARP_CLIENT_VERSION: 'v0.2026.05.01.08.00.stable_01'
      },
      'warp'
    ],
    // A Warp build that advertises the protocol but cannot render it gets no OSC 777.
    [
      {
        TERM_PROGRAM: 'WarpTerminal',
        WARP_CLI_AGENT_PROTOCOL_VERSION: '1',
        WARP_CLIENT_VERSION: 'v0.2026.03.25.08.24.stable_05'
      },
      null
    ]
  ] as const)('detects terminal from environment %o', (env, expected) => {
    expect(notificationTerminalForEnv(env)).toBe(expected)
  })

  it('builds the Warp agent payload the classic CLI sends', () => {
    expect(JSON.parse(warpAgentPayload('prompt', 'Waiting\u0007 for you', { cwd: '/w/proj', sessionId: 's1' }))).toEqual({
      agent: 'hermes',
      cwd: '/w/proj',
      event: 'permission_request',
      project: 'proj',
      session_id: 's1',
      summary: 'Waiting for you',
      v: 1
    })
    expect(JSON.parse(warpAgentPayload('complete', 'done', { cwd: '/w/proj', protocolVersion: '3' }))).toMatchObject({
      event: 'stop',
      response: 'done',
      v: 1
    })
  })
})
