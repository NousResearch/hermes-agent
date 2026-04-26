import { describe, expect, it } from 'vitest'

import { mergeLocalSlashCompletions } from '../lib/localSlashCompletion.js'

const commands = [
  { aliases: ['commands'], help: 'open the Aurora command palette', name: 'palette' },
  { help: 'change or show model', name: 'model' },
  { aliases: ['exit', 'q'], help: 'exit hermes', name: 'quit' }
]

describe('mergeLocalSlashCompletions', () => {
  it('adds local TUI commands missing from gateway slash completion', () => {
    const result = mergeLocalSlashCompletions({ items: [{ display: '/paste', text: '/paste' }], replace_from: 1 }, commands, '/pa')

    expect(result.items?.map(item => item.text)).toEqual(['/paste', '/palette'])
    expect(result.items?.find(item => item.text === '/palette')?.meta).toContain('open the Aurora command palette')
  })

  it('does not duplicate commands already returned by the gateway', () => {
    const result = mergeLocalSlashCompletions(
      { items: [{ display: '/palette', meta: 'remote', text: '/palette' }], replace_from: 1 },
      commands,
      '/pal'
    )

    expect(result.items).toEqual([{ display: '/palette', meta: 'remote', text: '/palette' }])
  })

  it('matches local aliases while completing to canonical commands', () => {
    const result = mergeLocalSlashCompletions({ items: [], replace_from: 1 }, commands, '/comm')

    expect(result.items).toEqual([
      { display: '/palette', meta: 'open the Aurora command palette · /commands', text: '/palette' }
    ])
  })
})
