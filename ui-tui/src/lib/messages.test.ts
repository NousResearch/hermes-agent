import { describe, expect, it } from 'vitest'

import { appendTranscriptMessage } from './messages.js'

describe('appendTranscriptMessage', () => {
  it('merges adjacent tool-only shelves into one transcript row', () => {
    const out = appendTranscriptMessage([{ kind: 'trail', role: 'system', text: '', tools: [{ id: 't1', name: 'Terminal', context: '"one"' }] }], {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [{ id: 't2', name: 'Terminal', context: '"two"' }]
    })

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', tools: [{ id: 't1', name: 'Terminal', context: '"one"' }, { id: 't2', name: 'Terminal', context: '"two"' }] }
    ])
  })

  it('merges tool shelves into the nearest thinking shelf', () => {
    const out = appendTranscriptMessage(
      [{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: [{ id: 't1', name: 'Terminal', context: '"one"' }] }],
      { kind: 'trail', role: 'system', text: '', tools: [{ id: 't2', name: 'Terminal', context: '"two"' }] }
    )

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: [{ id: 't1', name: 'Terminal', context: '"one"' }, { id: 't2', name: 'Terminal', context: '"two"' }] }
    ])
  })
})
