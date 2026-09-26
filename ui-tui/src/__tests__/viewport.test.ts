import { describe, expect, it } from 'vitest'

import { stickyPromptFromViewport } from '../domain/viewport.js'

describe('stickyPromptFromViewport', () => {
  it('hides the sticky prompt when a newer user message is already visible', () => {
    const messages = [
      { role: 'user' as const, text: 'older prompt' },
      { role: 'assistant' as const, text: 'older answer' },
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    const offsets = [0, 2, 10, 12, 20]

    expect(stickyPromptFromViewport(messages, offsets, 8, 16, false)).toBe('')
  })

  it('shows the latest user message above the viewport when no user message is visible', () => {
    const messages = [
      { role: 'user' as const, text: 'older prompt' },
      { role: 'assistant' as const, text: 'older answer' },
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    const offsets = [0, 2, 10, 12, 20]

    expect(stickyPromptFromViewport(messages, offsets, 16, 20, false)).toBe('current prompt')
  })

  it('shows the last prompt once the viewport starts after the history tail', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'completed answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 5], 8, 14, false)).toBe('current prompt')
  })

  it('shows a prompt as soon as its full row is above the viewport', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 10], 2, 8, false)).toBe('current prompt')
  })

  it('hides the sticky prompt at the bottom', () => {
    const messages = [
      { role: 'user' as const, text: 'current prompt' },
      { role: 'assistant' as const, text: 'current answer' }
    ]

    expect(stickyPromptFromViewport(messages, [0, 2, 10], 8, 10, true)).toBe('')
  })
})
