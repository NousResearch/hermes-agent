import { describe, expect, it } from 'vitest'

import { shouldShowResponseSeparator } from '../components/messageLine.js'

describe('shouldShowResponseSeparator', () => {
  it('separates assistant response text from visible details', () => {
    expect(shouldShowResponseSeparator({ role: 'assistant', text: 'final', thinking: 'plan' }, true)).toBe(true)
  })

  it('does not add a response separator without details or body text', () => {
    expect(shouldShowResponseSeparator({ role: 'assistant', text: 'final' }, false)).toBe(false)
    expect(shouldShowResponseSeparator({ role: 'assistant', text: '   ', thinking: 'plan' }, true)).toBe(false)
  })

  it('suppresses separators in compact mode', () => {
    expect(shouldShowResponseSeparator({ role: 'assistant', text: 'final', thinking: 'plan' }, true, true)).toBe(false)
  })

  it('does not add response separators to non-assistant transcript rows', () => {
    expect(shouldShowResponseSeparator({ role: 'user', text: 'prompt' }, true)).toBe(false)
    expect(shouldShowResponseSeparator({ role: 'system', text: 'note' }, true)).toBe(false)
  })
})
