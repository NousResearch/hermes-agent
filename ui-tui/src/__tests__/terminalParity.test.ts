import { describe, expect, it } from 'vitest'

import { terminalParityHints } from '../lib/terminalParity.js'

describe('terminalParityHints', () => {
  it('warns for Apple Terminal and SSH/tmux sessions', async () => {
    const hints = await terminalParityHints({
      TERM_PROGRAM: 'Apple_Terminal',
      TERM_SESSION_ID: 'w0t0p0:123',
      SSH_CONNECTION: '1',
      TMUX: '/tmp/tmux-1/default,1,0'
    } as NodeJS.ProcessEnv)

    expect(hints.map(h => h.key)).toEqual(expect.arrayContaining(['apple-terminal', 'remote', 'tmux']))
  })

  it('suggests IDE setup only for VS Code-family terminals that still need bindings', async () => {
    const hints = await terminalParityHints({ TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv)
    expect(hints.some(h => h.key === 'ide-setup')).toBe(true)
  })
})
