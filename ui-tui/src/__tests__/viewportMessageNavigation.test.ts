import { describe, expect, it } from 'vitest'

import { userMessageScrollTarget } from '../domain/viewport.js'
import type { Msg } from '../types.js'

const messages: Msg[] = [
  { role: 'system', text: 'intro' },
  { role: 'user', text: 'first' },
  { role: 'assistant', text: 'long answer' },
  { role: 'tool', text: 'tool output' },
  { role: 'user', text: 'second' },
  { role: 'assistant', text: 'another answer' },
  { role: 'user', text: 'third' }
]

const offsets = [0, 2, 5, 20, 25, 28, 40, 43]

describe('userMessageScrollTarget', () => {
  it('jumps back to the latest user turn above the viewport', () => {
    expect(userMessageScrollTarget(messages, offsets, 39, -1)).toBe(25)
  })

  it('advances through user turns without stopping on assistant or tool rows', () => {
    expect(userMessageScrollTarget(messages, offsets, 2, 1)).toBe(25)
  })

  it('advances past the current user row after a completed jump', () => {
    expect(userMessageScrollTarget(messages, offsets, 25, 1)).toBe(40)
  })

  it('returns null when there is no adjacent user turn', () => {
    expect(userMessageScrollTarget(messages, offsets, 2, -1)).toBeNull()
    expect(userMessageScrollTarget(messages, offsets, 40, 1)).toBeNull()
  })
})
