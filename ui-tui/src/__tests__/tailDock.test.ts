import { describe, expect, it } from 'vitest'

import { transcriptTailSlots } from '../domain/tailDock.js'

describe('transcriptTailSlots', () => {
  it('keeps queued input above todos and the live assistant summary below todos', () => {
    expect(transcriptTailSlots({ assistant: true, queue: true, todos: true })).toEqual([
      'queue',
      'todos',
      'assistant'
    ])
  })

  it('leaves todos at the bottom when there is no live assistant summary', () => {
    expect(transcriptTailSlots({ assistant: false, queue: true, todos: true })).toEqual(['queue', 'todos'])
  })

  it('keeps the assistant summary at the bottom when it is the only tail item', () => {
    expect(transcriptTailSlots({ assistant: true, queue: false, todos: false })).toEqual(['assistant'])
  })
})
