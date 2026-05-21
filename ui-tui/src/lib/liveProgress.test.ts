import { describe, expect, it } from 'vitest'

import type { Msg } from '../types.js'

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
    expect(canHoldToolShelf({ kind: 'trail', role: 'system', text: '', tools: [{ id: 't-one', name: 'one', context: '' }] })).toBe(true)
    expect(canHoldToolShelf({ role: 'assistant', text: 'done' })).toBe(false)
  })

  it('merges source rows into an existing shelf', () => {
    expect(
      mergeToolShelfInto(
        { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: [{ id: 't-one', name: 'one', context: '' }] },
        { kind: 'trail', role: 'system', text: '', tools: [{ id: 't-two', name: 'two', context: '' }] }
      )
    ).toEqual({ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: [{ id: 't-one', name: 'one', context: '' }, { id: 't-two', name: 'two', context: '' }] })
  })
})

describe('appendToolShelfMessage', () => {
  it('merges adjacent tool shelves into one contextual shelf', () => {
    const merged = appendToolShelfMessage([{ kind: 'trail', role: 'system', text: '', tools: [{ id: 't-one', name: 'one', context: '' }] }], {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [{ id: 't-two', name: 'two', context: '' }]
    })

    expect(merged).toEqual([{ kind: 'trail', role: 'system', text: '', tools: [{ id: 't-one', name: 'one', context: '' }, { id: 't-two', name: 'two', context: '' }] }])
  })

  it('adds tools to the nearest contextual thinking shelf', () => {
    const merged = appendToolShelfMessage(
      [{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: [{ id: 't-one', name: 'one', context: '' }] }],
      { kind: 'trail', role: 'system', text: '', tools: [{ id: 't-two', name: 'two', context: '' }] }
    )

    expect(merged).toEqual([{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: [{ id: 't-one', name: 'one', context: '' }, { id: 't-two', name: 'two', context: '' }] }])
  })

  it('merges through intervening thinking-only rows back into the nearest holder', () => {
    const prev: Msg[] = [
      { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: [{ id: 't-one', name: 'one', context: '' }] },
      { kind: 'trail', role: 'system', text: '', thinking: 'more plan' }
    ]

    const merged = appendToolShelfMessage(prev, {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: [{ id: 't-two', name: 'two', context: '' }]
    })

    expect(merged).toHaveLength(2)
    expect(merged[0]).toEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: 'plan',
      tools: [{ id: 't-one', name: 'one', context: '' }, { id: 't-two', name: 'two', context: '' }]
    })
    expect(merged[1]).toEqual({ kind: 'trail', role: 'system', text: '', thinking: 'more plan' })
  })

  it('collapses a chronological thinking/tool/thinking/tool stream into one shelf', () => {
    const events: Msg[] = [
      { kind: 'trail', role: 'system', text: '', thinking: 'plan' },
      { kind: 'trail', role: 'system', text: '', tools: [{ id: 't-one', name: 'one', context: '' }] },
      { kind: 'trail', role: 'system', text: '', thinking: 'more plan' },
      { kind: 'trail', role: 'system', text: '', tools: [{ id: 't-two', name: 'two', context: '' }] },
      { kind: 'trail', role: 'system', text: '', tools: [{ id: 't-three', name: 'three', context: '' }] }
    ]

    const reduced = events.reduce<Msg[]>((acc, msg) => appendToolShelfMessage(acc, msg), [])

    expect(reduced).toHaveLength(2)
    expect(reduced[0]).toEqual({
      kind: 'trail',
      role: 'system',
      text: '',
      thinking: 'plan',
      tools: [{ id: 't-one', name: 'one', context: '' }, { id: 't-two', name: 'two', context: '' }, { id: 't-three', name: 'three', context: '' }]
    })
    expect(reduced[1]).toEqual({ kind: 'trail', role: 'system', text: '', thinking: 'more plan' })
  })

  it('starts a new shelf across assistant text boundaries', () => {
    const merged = appendToolShelfMessage(
      [
        { kind: 'trail', role: 'system', text: '', tools: [{ id: 't-one', name: 'one', context: '' }] },
        { role: 'assistant', text: 'done' }
      ],
      { kind: 'trail', role: 'system', text: '', tools: [{ id: 't-two', name: 'two', context: '' }] }
    )

    expect(merged).toHaveLength(3)
  })
})
