import { describe, expect, it } from 'vitest'

import { latestCodeBlocks, latestCommands } from '../domain/copyTargets.js'

describe('copy scopes', () => {
  it('code: raw closed fences of the newest response that has any', () => {
    const blocks = latestCodeBlocks([
      { role: 'assistant', text: '```sh\nold\n```' },
      { role: 'assistant', text: 'Run:\n```bash\n  pip install x\n```\n~~~py\nprint(1)\n~~~\n```unclosed' },
      { role: 'assistant', text: 'no code' }
    ])

    expect(blocks).toEqual([
      { label: 'bash', text: '  pip install x' },
      { label: 'py', text: 'print(1)' }
    ])
  })

  it('cmd: terminal commands of the latest command-running turn only', () => {
    const cmds = latestCommands([
      { role: 'user', text: 'old' },
      { args: { command: 'echo stale' }, name: 'terminal', role: 'tool' },
      { role: 'user', text: 'do it' },
      { args: { path: 'x' }, name: 'read_file', role: 'tool' },
      { args: { command: 'make test' }, name: 'terminal', role: 'tool' },
      { args: { command: 'git status' }, name: 'terminal', role: 'tool' },
      { role: 'assistant', text: 'done' },
      { role: 'user', text: 'thanks' }
    ])

    expect(cmds.map(c => c.text)).toEqual(['make test', 'git status'])
  })
})
