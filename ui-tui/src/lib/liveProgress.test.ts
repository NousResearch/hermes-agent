import { describe, expect, it } from 'vitest'

import { appendToolShelfMessage, canHoldToolShelf, isTodoDone, mergeToolShelfInto } from './liveProgress.js'

describe('isTodoDone', () => {
  it('only treats non-empty all-completed/cancelled lists as done', () => {
    expect(isTodoDone([])).toBe(false)
    expect(isTodoDone([{ content: 'x', id: 'x', status: 'completed' }])).toBe(true)
    expect(isTodoDone([{ content: 'x', id: 'x', status: 'in_progress' }])).toBe(false)
    expect(
      isTodoDone([
        { content: 'x', id: 'x', status: 'completed' },
        { content: 'y', id: 'y', status: 'cancelled' }
      ])
    ).toBe(true)
  })
})

describe('tool shelf helpers', () => {
  it('recognizes contextual thinking shelves as holders', () => {
    expect(canHoldToolShelf({ kind: 'trail', role: 'system', text: '', thinking: 'plan' })).toBe(true)
    expect(canHoldToolShelf({ kind: 'trail', role: 'system', text: '', tools: ['one ✓'] })).toBe(true)
    expect(canHoldToolShelf({ role: 'assistant', text: 'done' })).toBe(false)
  })

  it('merges source rows into an existing shelf', () => {
    expect(
      mergeToolShelfInto(
        { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓'] },
        { kind: 'trail', role: 'system', text: '', tools: ['two ✓'] }
      )
    ).toEqual({ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓', 'two ✓'] })
  })
})

describe('appendToolShelfMessage', () => {
  it('merges adjacent tool shelves into one contextual shelf', () => {
    const merged = appendToolShelfMessage([{ kind: 'trail', role: 'system', text: '', tools: ['one ✓'] }], {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: ['two ✓']
    })

    expect(merged).toEqual([{ kind: 'trail', role: 'system', text: '', tools: ['one ✓', 'two ✓'] }])
  })

  it('adds tools to the nearest contextual thinking shelf', () => {
    const merged = appendToolShelfMessage(
      [{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓'] }],
      { kind: 'trail', role: 'system', text: '', tools: ['two ✓'] }
    )

    expect(merged).toEqual([{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['one ✓', 'two ✓'] }])
  })

  it('starts a new shelf across assistant text boundaries', () => {
    const merged = appendToolShelfMessage(
      [{ kind: 'trail', role: 'system', text: '', tools: ['one ✓'] }, { role: 'assistant', text: 'done' }],
      { kind: 'trail', role: 'system', text: '', tools: ['two ✓'] }
    )

    expect(merged).toHaveLength(3)
  })
})
