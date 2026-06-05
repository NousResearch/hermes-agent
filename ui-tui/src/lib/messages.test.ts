import { describe, expect, it } from 'vitest'

import { appendTranscriptMessage } from './messages.js'

describe('appendTranscriptMessage', () => {
  it('merges adjacent tool-only shelves into one transcript row', () => {
    const out = appendTranscriptMessage([{ kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓'] }], {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: ['Terminal("two") ✓']
    })

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓', 'Terminal("two") ✓'] }
    ])
  })

  it('merges tool shelves into the nearest thinking shelf', () => {
    const out = appendTranscriptMessage(
      [{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['Terminal("one") ✓'] }],
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("two") ✓'] }
    )

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['Terminal("one") ✓', 'Terminal("two") ✓'] }
    ])
  })
})
