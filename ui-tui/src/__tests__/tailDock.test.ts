import { describe, expect, it } from 'vitest'

import { transcriptTailSlots } from '../domain/tailDock.js'

describe('transcriptTailSlots', () => {
  it('keeps queued input above todos and the assistant summary below todos', () => {
    expect(transcriptTailSlots({ queue: true, todos: true })).toEqual([
      'queue',
      'todos',
      'assistant'
    ])
  })

  it('keeps todos above the always-mounted assistant area', () => {
    expect(transcriptTailSlots({ queue: false, todos: true })).toEqual(['todos', 'assistant'])
  })

  it('keeps the assistant summary at the bottom when it is the only tail item', () => {
    expect(transcriptTailSlots({ queue: false, todos: false })).toEqual(['assistant'])
  })
})
