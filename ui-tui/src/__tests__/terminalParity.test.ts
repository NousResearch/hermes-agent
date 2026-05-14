import { describe, expect, it, vi } from 'vitest'

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
    const readFile = vi.fn().mockRejectedValue(Object.assign(new Error('missing'), { code: 'ENOENT' }))

    const hints = await terminalParityHints({ TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv, {
      fileOps: { readFile },
      homeDir: '/tmp/fake-home'
    })

    expect(hints.some(h => h.key === 'ide-setup')).toBe(true)
  })

  it('suppresses IDE setup hint when keybindings are already configured', async () => {
    const readFile = vi.fn().mockResolvedValue(
      JSON.stringify([
        {
          key: 'cmd+c',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus && terminalTextSelected',
          args: { text: '\u001b[99;13u' }
        },
        {
          key: 'shift+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'ctrl+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'cmd+enter',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\\\r\n' }
        },
        {
          key: 'cmd+z',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\u001b[122;9u' }
        },
        {
          key: 'shift+cmd+z',
          command: 'workbench.action.terminal.sendSequence',
          when: 'terminalFocus',
          args: { text: '\u001b[122;10u' }
        }
      ])
    )

    const hints = await terminalParityHints({ TERM_PROGRAM: 'vscode' } as NodeJS.ProcessEnv, {
      fileOps: { readFile },
      homeDir: '/tmp/fake-home'
    })

    expect(hints.some(h => h.key === 'ide-setup')).toBe(false)
  })
})
