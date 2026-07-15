import { describe, expect, it } from 'vitest'

import { shouldShowResponseSeparator, shouldShowThinkingTrail } from '../components/messageLine.js'

describe('shouldShowResponseSeparator', () => {
  it('separates assistant response text from visible details', () => {
    expect(shouldShowResponseSeparator({ role: 'assistant', text: 'final', thinking: 'plan' }, true)).toBe(true)
  })

  it('does not add a response separator without details or body text', () => {
    expect(shouldShowResponseSeparator({ role: 'assistant', text: 'final' }, false)).toBe(false)
    expect(shouldShowResponseSeparator({ role: 'assistant', text: '   ', thinking: 'plan' }, true)).toBe(false)
  })

  it('does not add response separators to non-assistant transcript rows', () => {
    expect(shouldShowResponseSeparator({ role: 'user', text: 'prompt' }, true)).toBe(false)
    expect(shouldShowResponseSeparator({ role: 'system', text: 'note' }, true)).toBe(false)
  })
})

describe('shouldShowThinkingTrail', () => {
  it('hides an ordinary reasoning trail when every section is hidden', () => {
    const msg = { role: 'system', text: '', thinking: 'plan' } as const
    expect(shouldShowThinkingTrail(msg, 'hidden', 'hidden', 'hidden')).toBe(false)
  })

  it('shows an ordinary reasoning trail when any section is visible', () => {
    const msg = { role: 'system', text: '', thinking: 'plan' } as const
    expect(shouldShowThinkingTrail(msg, 'collapsed', 'hidden', 'hidden')).toBe(true)
    expect(shouldShowThinkingTrail(msg, 'hidden', 'collapsed', 'hidden')).toBe(true)
    expect(shouldShowThinkingTrail(msg, 'hidden', 'hidden', 'expanded')).toBe(true)
  })

  it('keeps a MoA reference block visible even when every section is hidden (#64657)', () => {
    const msg = {
      role: 'system',
      text: '',
      thinking: '◇ Reference 1/2 — model-a\nadvice-a',
      isMoaReference: true
    } as const
    expect(shouldShowThinkingTrail(msg, 'hidden', 'hidden', 'hidden')).toBe(true)
  })
})
